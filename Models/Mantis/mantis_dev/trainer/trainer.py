import torch
import numpy as np

from tqdm import tqdm
from copy import deepcopy
from itertools import chain
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import torch.distributed as dist
import os
# from mantis.architecture import Mantis8M
from mantis_dev.architecture import Mantis8M
from torch.utils.data.distributed import DistributedSampler
from .trainer_utils.architecture import FineTuningNetwork
from .trainer_utils.dataset import LabeledDataset, UnlabeledDataset
from .trainer_utils.scheduling import adjust_learning_rate
from .trainer_utils.pretraining import ContrastiveLoss, RandomCropResize, TensorboardLogger

import torch.nn.functional as F
# def random_double_crop(x_batch, crop_ratio=0.45, num_patches=32, min_length=16):
#     """
#     对输入 x_batch (shape: [B, 1, L]) 进行双段随机裁剪：
#       - 第一段：随机裁剪出长度为 int(L * crop_ratio) 的片段；
#       - 第二段：在剩余部分 [end_1, L] 中随机裁剪出长度同样为 int(L * crop_ratio)（或剩余部分允许的最大长度）。
    
#     为保证后续 reshape 和卷积操作，返回的每个片段必须：
#       1. 长度至少 >= min_length（比如 17）；
#       2. 长度能被 num_patches 整除。
    
#     因此，我们先对裁剪得到的片段进行零填充，使其长度达到最小满足条件的数，再截断到恰好是 num_patches 的整数倍。
#     """
#     B, C, L = x_batch.shape
#     crop_len = int(L * crop_ratio)

#     # 第一段裁剪
#     start_1 = np.random.randint(0, L - crop_len + 1)
#     end_1 = start_1 + crop_len
#     x_augmented_1 = x_batch[:, :, start_1:end_1]

#     # 第二段裁剪：从剩余部分 [end_1, L]
#     remaining_len = L - end_1
#     second_len = min(crop_len, remaining_len)
#     start_2 = np.random.randint(end_1, end_1 + remaining_len - second_len + 1)
#     end_2 = start_2 + second_len
#     x_augmented_2 = x_batch[:, :, start_2:end_2]

#     def pad_to_multiple(x_tensor, multiple, min_length):
#         """
#         将 x_tensor (shape: [B, C, Lp]) 用零填充到最小满足：
#           - Lp >= min_length
#           - Lp 是 multiple 的整数倍
#         """
#         Lp = x_tensor.shape[2]
#         # 首先确保至少满足 min_length
#         desired = max(Lp, min_length)
#         # 计算大于等于 desired 且能被 multiple 整除的最小数
#         target = ((desired + multiple - 1) // multiple) * multiple
#         pad_size = target - Lp
#         if pad_size > 0:
#             x_tensor = F.pad(x_tensor, (0, pad_size), mode='constant', value=0)
#         return x_tensor

#     def truncate_to_multiple(x_tensor, multiple):
#         """
#         如果 x_tensor 的长度超过能被 multiple 整除的部分，则截断多余部分。
#         """
#         Lp = x_tensor.shape[2]
#         remainder = Lp % multiple
#         if remainder != 0:
#             x_tensor = x_tensor[:, :, :Lp - remainder]
#         return x_tensor

#     # 对两个片段分别进行填充和截断
#     x_augmented_1 = pad_to_multiple(x_augmented_1, num_patches, min_length)
#     x_augmented_1 = truncate_to_multiple(x_augmented_1, num_patches)
    
#     x_augmented_2 = pad_to_multiple(x_augmented_2, num_patches, min_length)
#     x_augmented_2 = truncate_to_multiple(x_augmented_2, num_patches)

#     return x_augmented_1, x_augmented_2



class MantisTrainer:
    """
    A scikit-learn-like wrapper to use Mantis as a feature extractor or fine-tune it to the downstream task.

    Parameters
    ----------
    device: {'cpu', 'cuda'}
        On which device the model is located and trained.
    network: Mantis, default=None
        The foundation model. If None, the class initializes a Mantis object by itself (so weights are randomly
        initialized). Otherwise, pass a pre-trained model.
    """
    def __init__(self, device, network=None):
        self.device = device
        if network is None:
            network = Mantis8M(seq_len=512, hidden_dim=256, num_patches=32, scalar_scales=None, hidden_dim_scalar_enc=32,
                             epsilon_scalar_enc=1.1, transf_depth=6, transf_num_heads=8, transf_mlp_dim=512,
                             transf_dim_head=128, transf_dropout=0.1, device=device, pre_training=False)
        self.network = network.to(device)

    
    def pretrain(self, x, num_epochs=100, batch_size=512, learning_rate=2e-3, 
             crop_rate_range=[0, 0.2], temperature=0.1, data_parallel=True,
             checkpoint_path='./checkpoint/', experiment_name=None):
    # ① 导入DistributedSampler（用于多GPU数据采样）
    

    # ② 初始化对比损失函数
        criterion = ContrastiveLoss(temperature=temperature, device=self.device)

    # ③ 深拷贝基础模型，避免直接修改原模型
        network = deepcopy(self.network)

    # ④ 如果启用多GPU，则转换为同步 BN，并用DistributedDataParallel封装
        if data_parallel:
            network = nn.SyncBatchNorm.convert_sync_batchnorm(network)
        # 这里device_ids使用当前进程所使用的GPU索引（假设self.device格式为 "cuda:索引"）
            gpu_index = self.device.index if self.device.type == "cuda" else None
            network = nn.parallel.DistributedDataParallel(network, device_ids=[gpu_index] if gpu_index is not None else None,
                                                      find_unused_parameters=True)
        network.train()

    # ⑤ 构造无标签数据集，并使用DistributedSampler进行采样（确保每个进程获取不同数据）
        train_dataset = UnlabeledDataset(x)
        sampler = DistributedSampler(train_dataset) if data_parallel else None
        data_loader = DataLoader(train_dataset, sampler=sampler, batch_size=batch_size)

    # ⑥ 根据总进程数（world_size）调整学习率缩放
        world_size = dist.get_world_size() if (data_parallel and dist.is_initialized()) else 1
        scaled_lr = batch_size * world_size * learning_rate / 2048
        optimizer = torch.optim.AdamW(network.parameters(), lr=scaled_lr, betas=(0.9, 0.999), weight_decay=0.05)

    # ⑦ 获取当前进程的rank（只允许rank 0负责日志记录和模型保存）
        rank = dist.get_rank() if (data_parallel and dist.is_initialized()) else 0
        best_loss = 1e+10
        if rank == 0:
            logger = TensorboardLogger({}, base_path=checkpoint_path, folder_name=experiment_name)
            best_model_filename = os.path.join(logger.base_path, 'best_epoch.pth')
            last_model_filename = os.path.join(logger.base_path, 'last_epoch.pth')

    # ⑧ 训练循环，使用tqdm显示进度，并在每个epoch前调用sampler.set_epoch以确保随机性
        progress_bar = tqdm(range(num_epochs))
        step = 0
        for epoch in progress_bar:
            if sampler is not None:
                sampler.set_epoch(epoch)  # 更新采样器状态以保证不同进程每个epoch都shuffle数据
            loss_list = []
            for x_batch in data_loader:
            # 动态调整学习率（如果有实现该函数，确保其适用于多GPU设置）
                adjust_learning_rate(num_epochs, optimizer, data_loader, step, scaled_lr)
                x_batch = x_batch.to(self.device)
                step += 1

            # 随机采样两个裁剪比例以生成两种数据增强
                crop_rate_1 = torch.empty(1).uniform_(crop_rate_range[0], crop_rate_range[1]).item()
                crop_rate_2 = torch.empty(1).uniform_(crop_rate_range[0], crop_rate_range[1]).item()
                x_augmented_1 = RandomCropResize(x_batch, crop_rate=crop_rate_1).to(self.device)
                x_augmented_2 = RandomCropResize(x_batch, crop_rate=crop_rate_2).to(self.device)

                # x_augmented_1, x_augmented_2 = random_double_crop(x_batch, crop_ratio=0.45)

            # 前向传播计算对比损失
                out_1 = network(x_augmented_1)
                out_2 = network(x_augmented_2)
                loss = criterion(out_1, out_2)

            # 反向传播和优化更新
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())

            # ⑨ 仅在rank 0上进行日志记录与模型保存
            if rank == 0:
                train_loss = np.mean(loss_list)
                logger.update(epoch=epoch, train_loss=train_loss)
                progress_bar.set_description("Epoch {:d}: Train Loss {:.4f}".format(epoch, train_loss), refresh=True)
            # 如果当前epoch的loss更低，则保存模型
                if train_loss < best_loss:
                    best_loss = train_loss
                    self.save(best_model_filename, network, data_parallel=data_parallel)
    
    # ⑩ 保存最后一个epoch的模型，并加载最优模型（同样只在rank 0操作）
        if rank == 0:
            self.save(last_model_filename, network, data_parallel=data_parallel)
            self.load(best_model_filename)
            logger.finish(best_loss=best_loss)


    # def pretrain(self, x, num_epochs=100, batch_size=512, learning_rate=2e-3, crop_rate_range=[0, 0.2], temperature=0.1, data_parallel=True,
    #              checkpoint_path='./checkpoint/', experiment_name=None):
    #     criterion = ContrastiveLoss(temperature=temperature, device=self.device)
        
    #     network = deepcopy(self.network)
    #     if data_parallel:
    #         network = nn.SyncBatchNorm.convert_sync_batchnorm(network)
    #         network = nn.parallel.DistributedDataParallel(network, device_ids=[self.device], find_unused_parameters=True)
    #     network.train()

    #     train_dataset = UnlabeledDataset(x)
    #     sampler = DistributedSampler(train_dataset) if data_parallel else None
    #     data_loader = DataLoader(train_dataset, sampler=sampler, batch_size=batch_size)
    
    #     learning_rate = batch_size * torch.cuda.device_count() * learning_rate / 2048
    #     optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=0.05)

    #     best_loss = 1e+10
    #     # if self.local_rank == 0:
    #     if True:
    #         # logger = TensorboardLogger(self.args_dict, base_path=checkpoint_path, folder_name=experiment_name)
    #         logger = TensorboardLogger({}, base_path=checkpoint_path, folder_name=experiment_name)
    #         best_model_filename = logger.base_path + 'best_epoch.pth'
    #         last_model_filename = logger.base_path + 'last_epoch.pth'

    #     progress_bar = tqdm(range(num_epochs))
    #     step = 0
    #     for epoch in progress_bar:
    #         loss_list = []
    #         for x_batch in data_loader:
    #             adjust_learning_rate(num_epochs, optimizer, data_loader, step, learning_rate)
    #             x_batch = x_batch.to(self.device)
    #             step += 1
    #             # sample crop scales
    #             crop_rate_1 = torch.empty(1).uniform_(crop_rate_range[0], crop_rate_range[1]).item()
    #             crop_rate_2 = torch.empty(1).uniform_(crop_rate_range[0], crop_rate_range[1]).item()
    #             # create 2 augmentations of the batch
    #             x_augmented_1 = RandomCropResize(x_batch, crop_rate=crop_rate_1).to(self.device)
    #             x_augmented_2 = RandomCropResize(x_batch, crop_rate=crop_rate_2).to(self.device)
    #             # =============== forward ===============
    #             out_1 = network(x_augmented_1)
    #             out_2 = network(x_augmented_2)
    #             loss = criterion(out_1, out_2)
    #             # =============== backward ===============
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #             loss_list.append(loss.item())
    #         # =============== save model / update log ===============
    #         # if self.local_rank == 0:
    #         if True:
    #             train_loss = np.mean(loss_list)
    #             logger.update(epoch=epoch, train_loss=train_loss)
    #             progress_bar.set_description("Epoch {:d}: Train Loss {:.4f}".format(epoch, train_loss), refresh=True)
    #             reference_loss = train_loss
    #             # update the best training loss model
    #             if reference_loss < best_loss:
    #                 best_loss = reference_loss
    #                 self.save(best_model_filename, network, data_parallel=data_parallel)
    #     # save the last epoch model
    #     self.save(last_model_filename, network, data_parallel=data_parallel)
    #     # load the best epoch
    #     # if self.local_rank == 0:
    #     if True:
    #         self.load(last_model_filename)
    #         logger.finish(best_loss=best_loss)

    def fit(self, x, y, fine_tuning_type='full', adapter=None, head=None, num_epochs=500, batch_size=256,
            base_learning_rate=2e-4, init_optimizer=None, criterion=None, learning_rate_adjusting=True):
        """
        Fit (fine-tune) the foundation model to the downstream task.

        Parameters
        ----------
        x: array-like of shape (n_samples, n_channels, seq_len)
            The input samples. If data is univariate case, the shape should be ``(n_samples, 1, seq_len)``.
            ``seq_len`` should correspond to ``self.network.seq_len``.
        y: array-like of shape (n_samples,)
            The class labels with the following unique_values: ``[i for i in range(n_classes)]``
        fine_tuning_type: {'full', 'adapter_head', 'head', 'scratch'}, default='full'
            fine-tuning type
        adapter: nn.Module, default=None
            Adapter is a part of the network that precedes the foundation model and reduces the original data matrix
            of shape ``(n_samples, n_channels, seq_len)`` to ``(n_samples, new_n_channels, seq_len)``. By default,
            adapter is not used.
        head: nn.Module, default=None
            Head is a part of the network that follows the foundation model and projects from the embedding space
            to the probability matrix of shape ``(n_samples, n_classes)``. By default, head is a linear layer ``Linear``
            preceded by the layer normalization ``LayerNorm``.
        num_epochs: int, default=500
            Number of training epochs.
        batch_size: int, default=256
            Batch size.
        base_learning_rate: float, default=2e-4
            Learning rate that optimizer starts from. If ``learning_rate_adjusting`` is ``False``,
            it remains to be fixed
        init_optimizer: callable, default=None
            Function that initializes the optimizer. By default, ``AdamW`` 
            with pre-defined hyperparameters (except the learning rate) is used.
        criterion: nn.Module, default=None
            Learning criterion. By default, ``CrossEntropyLoss`` is used. 
        learning_rate_adjusting: bool, default=True
            Whether to use the implemented scheduling scheme.
        
        Returns
        -------
        self.fine_tuned_model: nn.Module
            Network fine-tuned to the downstream task.
        """
        
        self.fine_tuning_type = fine_tuning_type
        # ==== get the whole fine-tuning architecture ====
        # init head
        if head is None:
            num_channels = x.shape[1] if adapter is None else adapter.new_num_channels
            head = nn.Sequential(
                nn.LayerNorm(self.network.hidden_dim * num_channels),
                nn.Linear(self.network.hidden_dim *
                          num_channels, np.unique(y).shape[0])
            ).to(self.device)
        else:
            head = head.to(self.device)
        # init adapter
        if adapter is not None:
            adapter = adapter.to(self.device)
        else:
            adapter = None
        # when fine-tuning head, the forward pass over the encoder will be done only once (see init data_loader below)
        self.fine_tuned_model = FineTuningNetwork(
            deepcopy(self.network), head, adapter).to(self.device)

        # ==== get params to fine-tune and set them into the training model ====
        parameters = self._get_fine_tuning_params(
            fine_tuning_type=fine_tuning_type)
        self.fine_tuned_model.eval()
        self._set_train(fine_tuning_type=fine_tuning_type)

        # ==== init criterion, optimizer and dataloader ====
        # init criterion if None
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        # init optimizer by init_optimizer
        if init_optimizer is None:
            optimizer = torch.optim.AdamW(
                parameters, lr=base_learning_rate, betas=(0.9, 0.999), weight_decay=0.05)
        else:
            optimizer = init_optimizer(parameters)

        # init dataloader, for the head fine-tuning we directly load the embeddings
        train_dataset = LabeledDataset(x, y)
        data_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)

        # ==== training loop ====
        progress_bar = tqdm(range(num_epochs))
        step = 1
        for epoch in progress_bar:
            loss_list = []
            for (x_batch, y_batch) in data_loader:
                # adjust learning rate
                if learning_rate_adjusting:
                    adjust_learning_rate(
                        num_epochs, optimizer, data_loader, step, base_learning_rate)
                # read data
                x_batch, y_batch = x_batch.to(
                    self.device), y_batch.to(self.device)
                step += 1
                # forward
                output = self.fine_tuned_model(x_batch)
                loss = criterion(output, y_batch)
                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())
            # report average loss over all batches
            avg_loss_in_epoch = np.mean(loss_list)
            progress_bar.set_description("Epoch {:d}: Train Loss {:.4f}".format(
                epoch, avg_loss_in_epoch), refresh=True)
        return self.fine_tuned_model

    def transform(self, x, batch_size=256, three_dim=False, to_numpy=True):
        """
        Projects to the embedding space using self.network.

        Parameters
        ----------
        x: array-like of shape (n_samples, n_channels, seq_len)
            The input samples. If data is univariate case, the shape should be (n_samples, 1, seq_len).
            ``seq_len`` should correspond to ``self.network.seq_len``. In the multivariate case, each channel is sent
            independently to the foundation model.
        batch_size: int, default=256
            To fit memory, the data matrix is split into the chunks for inference. ``batch_size`` corresponds to
            the chunk size.
        three_dim: bool, default=False
            Whether the output should be two- or three-dimensional. By default, the embeddings of all channels are
            concatenated along the same axis, so the output is of shape (n_samples, n_channels * hidden_dim). When
            three_dim is set to True, the output is of shape (n_samples, n_channels, hidden_dim).
        to_numpy: bool, default=True
            Whether to convert the output to a numpy array.

        Returns
        -------
        z: array-like of shape (n_samples, n_channels * hidden_dim) or (n_samples, n_channels, hidden_dim)
            Embeddings.
        """
        concat = np.concatenate if to_numpy else torch.cat
        # apply network to each channel
        if three_dim:
            return concat([
                self._transform(x[:, [i], :], batch_size=batch_size, to_numpy=to_numpy)[
                    :, None, :]
                for i in range(x.shape[1])
            ], axis=1)
        else:
            return concat([
                self._transform(x[:, [i], :], batch_size=batch_size, to_numpy=to_numpy)
                for i in range(x.shape[1])
            ], axis=1)

    def _transform(self, x, batch_size=256, to_numpy=True):
        self.network.eval()
        dataloader = self._prepare_dataloader_for_inference(x, batch_size)
        outs = []
        for _, batch in enumerate(dataloader):
            x = batch[0].to(self.device)
            with torch.no_grad():
                out = self.network(x)
            outs.append(out)
        outs = torch.cat(outs)
        self.network.train()
        if to_numpy:
            return outs.cpu().numpy()
        else:
            return outs

    def predict_proba(self, x, batch_size=256, to_numpy=True):
        """
        Predicts the class probability matrix using self.fine_tuned_model.

        Parameters
        ----------
        x: array-like of shape (n_samples, n_channels, seq_len)
            The input samples. If data is univariate case, the shape should be (n_samples, 1, seq_len).
            ``seq_len`` should correspond to ``self.network.seq_len``.
        batch_size: int, default=256
            To fit memory, the data matrix is split into the chunks for inference. ``batch_size`` corresponds to
            the chunk size.
        to_numpy: bool, default=True
            Whether to convert the output to a numpy array.
        
        Returns
        -------
        probs: array_like of shape (n_samples, n_classes)
            Class probability matrix.
        """

        self.fine_tuned_model.eval()
        dataloader = self._prepare_dataloader_for_inference(x, batch_size)
        outs = []
        for _, batch in enumerate(dataloader):
            x = batch[0].to(self.device)
            with torch.no_grad():
                out = torch.softmax(self.fine_tuned_model(x), dim=-1)
            outs.append(out.cpu())
        outs = torch.cat(outs)
        if to_numpy:
            return outs.cpu().numpy()
        else:
            return outs

    def predict(self, x, batch_size=256, to_numpy=True):
        """
        Predicts the class labels using self.fine_tuned_model.

        Parameters
        ----------
        x: array-like of shape (n_samples, n_channels, seq_len)
            The input samples. If data is univariate case, the shape should be (n_samples, 1, seq_len).
            ``seq_len`` should correspond to ``self.network.seq_len``.
        batch_size: int, default=256
            To fit memory, the data matrix is split into the chunks for inference. ``batch_size`` corresponds to
            the chunk size.
        to_numpy: bool, default=True
            Whether to convert the output to a numpy array.
        
        Returns
        -------
        y: array_like of shape (n_samples,)
            Class labels.
        """
        probs = self.predict_proba(x, batch_size=batch_size, to_numpy=to_numpy)
        return probs.argmax(axis=1)

    def save(self, file_path, network, data_parallel=True):
        """
        Save the trained model to a file. 

        Parameters
        ----------
        file_path : 
            str model file path to save
        network: 
            trained model
        data_parallel: 
            whether the network is wrapped into DistributedDataParallel
        """
        checkpoints = dict()
        if data_parallel:
            checkpoints['net_param'] = network.module.state_dict()
        else:
            checkpoints['net_param'] = network.state_dict()
        # checkpoints['other_param'] = self.args_dict
        checkpoints['other_param'] = {}
        torch.save(checkpoints, file_path)
    
    def load(self, file_path):
        """
        Load the model from a file.

        Parameters
        ----------
        file_path : str
            model file path to load

        Returns
        -------
        self after loading param
        """
        model_params = torch.load(file_path)
        self.network.load_state_dict(model_params['net_param'])
        return self

    def _prepare_dataloader_for_inference(self, x, batch_size):
        if isinstance(x, torch.Tensor):
            dataset = TensorDataset(x.type(torch.float))
        else:
            dataset = TensorDataset(torch.tensor(x, dtype=torch.float))
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def __repr__(self):
        class_name = self.__class__.__name__
        return f"{class_name}({', '.join(f'{k}={v}' for k, v in vars(self).items() if k not in ['network'])})"

    def _get_fine_tuning_params(self, fine_tuning_type):
        tune_params_dict = {
            "full": [
                self.fine_tuned_model.parameters()
            ],
            "scratch": [
                self.fine_tuned_model.parameters()
            ],
            "head": [
                self.fine_tuned_model.head.parameters()
            ],
            "adapter_head": [
                [] if self.fine_tuned_model.adapter is None else self.fine_tuned_model.adapter.parameters(),
                self.fine_tuned_model.head.parameters()
            ]
        }
        params_list = list(chain(*tune_params_dict[fine_tuning_type]))
        return params_list

    def _set_train(self, fine_tuning_type):
        if fine_tuning_type in ["full", "scratch"]:
            self.fine_tuned_model.train()
        elif fine_tuning_type == "head":
            self.fine_tuned_model.head.train()
        elif fine_tuning_type == "adapter_head":
            self.fine_tuned_model.adapter.train()
            self.fine_tuned_model.head.train()
        else:
            raise KeyError("Unknown fine_tuning_type")

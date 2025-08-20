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
        criterion = ContrastiveLoss(temperature=temperature, device=self.device)

        network = deepcopy(self.network)

        if data_parallel:
            network = nn.SyncBatchNorm.convert_sync_batchnorm(network)
            gpu_index = self.device.index if self.device.type == "cuda" else None
            network = nn.parallel.DistributedDataParallel(network, device_ids=[gpu_index] if gpu_index is not None else None,
                                                      find_unused_parameters=True)
        network.train()

        train_dataset = UnlabeledDataset(x)
        sampler = DistributedSampler(train_dataset) if data_parallel else None
        data_loader = DataLoader(train_dataset, sampler=sampler, batch_size=batch_size)

        world_size = dist.get_world_size() if (data_parallel and dist.is_initialized()) else 1
        scaled_lr = batch_size * world_size * learning_rate / 2048
        optimizer = torch.optim.AdamW(network.parameters(), lr=scaled_lr, betas=(0.9, 0.999), weight_decay=0.05)

        rank = dist.get_rank() if (data_parallel and dist.is_initialized()) else 0
        best_loss = 1e+10
        if rank == 0:
            logger = TensorboardLogger({}, base_path=checkpoint_path, folder_name=experiment_name)
            best_model_filename = os.path.join(logger.base_path, 'best_epoch.pth')
            last_model_filename = os.path.join(logger.base_path, 'last_epoch.pth')

        progress_bar = tqdm(range(num_epochs))
        step = 0
        for epoch in progress_bar:
            if sampler is not None:
                sampler.set_epoch(epoch)  
            loss_list = []
            for x_batch in data_loader:
            
                adjust_learning_rate(num_epochs, optimizer, data_loader, step, scaled_lr)
                x_batch = x_batch.to(self.device)
                step += 1

                crop_rate_1 = torch.empty(1).uniform_(crop_rate_range[0], crop_rate_range[1]).item()
                crop_rate_2 = torch.empty(1).uniform_(crop_rate_range[0], crop_rate_range[1]).item()
                x_augmented_1 = RandomCropResize(x_batch, crop_rate=crop_rate_1).to(self.device)
                x_augmented_2 = RandomCropResize(x_batch, crop_rate=crop_rate_2).to(self.device)

                # x_augmented_1, x_augmented_2 = random_double_crop(x_batch, crop_ratio=0.45)

                out_1 = network(x_augmented_1)
                out_2 = network(x_augmented_2)
                loss = criterion(out_1, out_2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())

            if rank == 0:
                train_loss = np.mean(loss_list)
                logger.update(epoch=epoch, train_loss=train_loss)
                progress_bar.set_description("Epoch {:d}: Train Loss {:.4f}".format(epoch, train_loss), refresh=True)
                if train_loss < best_loss:
                    best_loss = train_loss
                    self.save(best_model_filename, network, data_parallel=data_parallel)
    
        if rank == 0:
            self.save(last_model_filename, network, data_parallel=data_parallel)
            self.load(best_model_filename)
            logger.finish(best_loss=best_loss)


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

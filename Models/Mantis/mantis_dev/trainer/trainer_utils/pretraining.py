import torch
import os

import torch.nn.functional as F

from torch import nn
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature, device):
        super().__init__()
        self.temperature = temperature
        self.device = device
    
    def forward(self, q, k):
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        logits = torch.einsum('nc,ck->nk', [q, k.t()])
        logits /= self.temperature
        labels = torch.arange(q.shape[0], dtype=torch.long).to(self.device)
        return nn.CrossEntropyLoss()(logits, labels)


def RandomCropResize(x, crop_rate, size=None):
    """
    RandomCropResize augmentation.

    Parameters
    ----------
        x: 
            Original time series with shape [n_samples, n_channels, seq_len].
        crop_rate, float in [0, 1):
            How much (in %) of a time series will be cropped.
        size:
            To which size the input will be interpolated. By default, to the original sequence length.
    """
    # Determine the sequence length (time dimension)
    seq_len = x.shape[-1]
    size = seq_len if size is None else size
    cropped_seq_len = int(seq_len * (1-crop_rate))  # Calculate the length of the cropped sequence

    # Generate a random starting index for the crop
    start_idx = torch.randint(0, seq_len - cropped_seq_len + 1, (1,)).item()

    # Perform the crop on the time dimension (last dimension)
    x_cropped = x[:, :, start_idx:start_idx+cropped_seq_len]

    # Resize the cropped sequence to the target size
    # We only need to resize along the time dimension (last dimension)
    x_resized = F.interpolate(x_cropped, size=size, mode='linear', align_corners=False)

    return x_resized


class TensorboardLogger:
    def __init__(self, args_dict, base_path='.checkpoint/', folder_name=None):
        self.args_dict = args_dict
        # name of folder for this session
        if folder_name is None:
            now = datetime.now()
            folder_name = now.strftime("%Y-%m-%d_%H-%M-%S")
        self.base_path = base_path + folder_name + '/'
        # create folder
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
        self.writer = SummaryWriter(self.base_path)
    
    def update(self, train_loss, epoch):
        self.writer.add_scalar('Train Loss', train_loss, epoch)
    
    def finish(self, best_loss):
        self.writer.add_hparams(self.args_dict, {'best loss': best_loss})
        self.writer.close()

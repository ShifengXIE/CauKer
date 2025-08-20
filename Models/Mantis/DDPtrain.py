import os
import random
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

import pyarrow as pa
import pyarrow.dataset as ds

from mantis_dev.architecture import Mantis8M
from mantis_dev.trainer import MantisTrainer

def main():
    ###############################################################################
    ###############################################################################
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl", init_method="env://")
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    ###############################################################################
    ###############################################################################
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ###############################################################################
    ###############################################################################
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    ###############################################################################
    ###############################################################################
    arrow_file_path = "/home/sxie/Curriculum/CauKer100K_Kernel6_Parents5.arrow"  
    dataset_arrow = ds.dataset(source=arrow_file_path, format="arrow")

    num_total_rows = dataset_arrow.count_rows()
    if rank == 0:
        print(f"[INFO] Total rows in arrow: {num_total_rows}")

    # Randomly select 100,000 rows
    num_select = min(100000, num_total_rows)
    indices = random.sample(range(num_total_rows), num_select)
    indices.sort()

    table = dataset_arrow.take(indices)
    df_selected = table.to_pandas()  

    # # if you want to load the entire dataset instead of a subset, uncomment the following lines:
    # table = dataset_arrow.to_table()  
    # df_selected = table.to_pandas()  


    ts_list = df_selected["target"].to_list()  
    X_data = np.stack(ts_list, axis=0)         

    # mask_good = ~np.isnan(X_data).any(axis=1)  
    # X_data = X_data[mask_good]  
    # print(f"[INFO] Removed {len(ts_list)-len(X_data)} bad series; remaining {len(X_data)}")

    
    X_data_torch = torch.tensor(X_data, dtype=torch.float)

    # x = X_data_torch  # [N, 1, 512] float32
    # print(torch.isnan(x).any(), torch.isinf(x).any())
    # print(x.min().item(), x.max().item(), x.std().item())
    # print("TESTED")


    old_len = X_data_torch.shape[-1]
    target_len = 512
    X_data_torch = X_data_torch.unsqueeze(1)

    if old_len != target_len:
        X_data_torch = F.interpolate(
            X_data_torch,
            size=target_len,
            mode='linear',
            align_corners=False
        )
        if rank == 0:
            print(f"[INFO] Interpolated from length {old_len} to {target_len}.")

    if rank == 0:
        print(f"[INFO] Final shape for training data: {X_data_torch.shape}")
        

    ###############################################################################
    # Trainer
    ###############################################################################
    mantis_model = Mantis8M(
        seq_len=512,
        hidden_dim=256,
        num_patches=32,
        scalar_scales=None,
        hidden_dim_scalar_enc=32,
        epsilon_scalar_enc=1.1,
        transf_depth=6,
        transf_num_heads=8,
        transf_mlp_dim=512,
        transf_dim_head=128,
        transf_dropout=0.1,
        device=device,
        pre_training=False
    )


    trainer = MantisTrainer(device=device, network=mantis_model)
    # Load a pretrained model if needed
    #trainer.load("/home/sxie/scaling_tsfm/checkpoint/mean_kernel100K/last_epoch.pth")

    ###############################################################################
    # Self-supervised pretraining
    ###############################################################################
    if rank == 0:
        print("[INFO] Starting self-supervised pretraining ...")

    X_for_trainer = X_data_torch.cpu().numpy()

    trainer.pretrain(
        x=X_for_trainer,
        num_epochs=100,
        batch_size=256,
        learning_rate=2e-3,
        crop_rate_range=[0, 0.2],
        temperature=0.1,
        data_parallel=True,
        checkpoint_path='./checkpoint/',
        experiment_name='Graph100K-k6P5',
    )

    if rank == 0:
        print("[INFO] Pretraining completed! Checkpoints saved in ./checkpoint/ directory.")


if __name__ == "__main__":
    main()

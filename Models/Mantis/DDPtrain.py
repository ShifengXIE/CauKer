import argparse
import random
import torch
import os

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import torch.nn.functional as F
import torch.distributed as dist

from sklearn.ensemble import RandomForestClassifier
from importlib.metadata import version, PackageNotFoundError
from packaging import version as v

from mantis.architecture import Mantis8M
from mantis.trainer import MantisTrainer


def check_installed_version(package_name, threshold):
    try:
        installed_version = version(package_name)
        if v.parse(installed_version) < v.parse(threshold):
            raise RuntimeError(
                f"{package_name} version {threshold} or higher required, "
                f"but {installed_version} is installed."
            )
        else:
            print(f"{package_name} {installed_version} is OK")
    except PackageNotFoundError:
        raise RuntimeError(f"{package_name} is not installed.")


def main(cauker_data_path, seed, file_name):
    # ========= Initialize Distributed Environment =========
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        # NCCL backend for distributed initialization
        dist.init_process_group(backend="nccl", init_method="env://")
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    # ========= Set Random Seed =========
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ========= Set Device =========
    # each process corresponds to one GPU, use local_rank to specify which GPU to use for the current process
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # ========= Check Version of mantis =========
    check_installed_version(package_name="mantis", threshold="0.2.0")

    # ========= Load Data =========
    dataset_arrow = ds.dataset(source=cauker_data_path, format="arrow")
    num_total_rows = dataset_arrow.count_rows()
    if rank == 0:
        print(f"Total rows in arrow: {num_total_rows}")

    # randomly select 100,000 rows
    num_select = min(100000, num_total_rows)
    indices = random.sample(range(num_total_rows), num_select)
    indices.sort()
    table = dataset_arrow.take(indices)
    df_selected = table.to_pandas()  

    # if you want to load the entire dataset instead of a subset, uncomment the following lines:
    # table = dataset_arrow.to_table()  
    # df_selected = table.to_pandas()  

    ts_list = df_selected["target"].to_list()
    X_data = np.stack(ts_list, axis=0)

    if rank == 0:
        print("Pretraining data dims: ", X_data.shape)

    # ========= Initialize Mantis Model =========
    network = Mantis8M(
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
        pre_training=False # when this argument is True, a projection head for contrastive learning is used at the forward step.
        # we found that without this projector the performance is comparable.
    )

    trainer = MantisTrainer(device=device, network=network)

    # ========= Pre-training =========
    if rank == 0:
        print("Starting to pre-train")
        # if file_name is not None, create absent folders in the path
        if file_name is not None:
            dir_path = os.path.dirname(file_name)
            if dir_path != "":
                os.makedirs(dir_path, exist_ok=True)

    trainer.pretrain(
        X_data,
        num_epochs=100,                     # Adjust the number of epochs as needed
        batch_size=256,                     # Adjust batch size based on GPU memory
        base_learning_rate=2e-3,            # Initial learning rate
        data_parallel=True,                 # Enable distributed data parallelism
        learning_rate_adjusting=True,       # Cosine annealing
        file_name=file_name                 # Where to save the final checkpoint
    )
    # if the checkpoint was saved, you can load it later by ``trainer.load('file_name')``

    if rank == 0:
        print("Pre-training is finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cauker_data_path', type=str, help='Where generated CauKer data is located.')
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility of a single experiment. Note that it is reproducible only on the same machine.")
    parser.add_argument('--file_name', type=str, default='./checkpoint/Graph100K-k6P5', help="Where to save the final checkpoint. By default, the checkpoint is not saved anywhere.")
    
    args = parser.parse_args()
    cauker_data_path = args.cauker_data_path
    seed = args.seed
    file_name = args.file_name

    main(cauker_data_path, seed, file_name)

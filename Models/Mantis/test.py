import os
import random
import logging
import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier

# 1. Import from custom modules
from data_reader import DataReader                 # Your DataReader
from mantis_dev.architecture import Mantis8M       # Your Mantis8M class
from mantis_dev.trainer import MantisTrainer       # Your MantisTrainer class

#############################
# Logging Configuration
#############################
logging.basicConfig(level=logging.INFO,filename='test.log', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------
# 2. Fix random seed for reproducibility
# ---------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------------------
# 3. Read UCR data
# ---------------------------
DATA_PATH = "/home/data/"  # Make sure it contains subdirectory UCRArchive_2018/
reader = DataReader(data_path=DATA_PATH)  

# Store train/test data in a dict
all_data = {}

for ds_name in reader.dataset_list_ucr:
    try:
        X_train, y_train = reader.read_dataset(ds_name, which_set='train')
        X_test,  y_test  = reader.read_dataset(ds_name, which_set='test')

        logger.info(f"Successfully read dataset: {ds_name}")
        logger.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

        # Save to dict
        all_data[ds_name] = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test":  X_test,
            "y_test":  y_test
        }

    except Exception as e:
        logger.warning(f"Failed to read dataset {ds_name}: {e}")
        continue

if len(all_data) == 0:
    raise RuntimeError("No dataset was loaded from UCRArchive_2018. Please check your path or data format.")

# ---------------------------
# 4. Load pretrained Mantis model
# ---------------------------
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

mantis_model = Mantis8M(
    seq_len=512,       # Must match pretraining
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

MODEL_PATH = "/home/sxie/scaling_tsfm/checkpoint/Graph100K-k6P5/best_epoch.pth"
trainer.load(MODEL_PATH)
logger.info("Pretrained model loaded successfully.")

# If you need to resize sequences to 512, define a function (commented out by default):
# def resize_to_512(X):
#     # X: shape (N, C, seq_len), C=1
#     X_torch = torch.tensor(X, dtype=torch.float)
#     X_resized = F.interpolate(X_torch, size=512, mode='linear', align_corners=False)
#     return X_resized.numpy()

# ---------------------------
# 5. Per-dataset feature extraction and classification
# ---------------------------
logger.info("Start feature extraction and classification for each UCR dataset...")

results = []  # collect (ds_name, train_size, test_size, accuracy)

for ds_name, data_dict in all_data.items():
    X_train = data_dict["X_train"]
    y_train = data_dict["y_train"]
    X_test  = data_dict["X_test"]
    y_test  = data_dict["y_test"]

    # # Optional resizing:
    # X_train = resize_to_512(X_train)
    # X_test  = resize_to_512(X_test)

    # Extract features
    Z_train = trainer.transform(X_train, batch_size=256, to_numpy=True)
    Z_test  = trainer.transform(X_test,  batch_size=256, to_numpy=True)
    

    # Train and evaluate RandomForest
    rf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    rf.fit(Z_train, y_train)
    y_pred = rf.predict(Z_test)
    accuracy = (y_pred == y_test).mean()

    results.append((ds_name, len(X_train), len(X_test), accuracy))

# ---------------------------
# 6. Summary
# ---------------------------
logger.info("Classification summary for all UCR subsets:")
for ds_name, train_size, test_size, acc in results:
    logger.info(f"Dataset: {ds_name}, Train size: {train_size}, "
                f"Test size: {test_size}, Accuracy: {acc:.4f}")

# Calculate the average accuracy over all datasets
if results:
    average_accuracy = sum(acc for _, _, _, acc in results) / len(results)
    logger.info(f"Average accuracy over all datasets: {average_accuracy:.4f}")
else:
    logger.warning("No valid datasets to compute average accuracy.")
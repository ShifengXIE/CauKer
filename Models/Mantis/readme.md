
### Training Mantis with Our Code

> **Note**  
> The implementation of Mantis originates from the official repository by [Huawei Noah's Ark Lab](https://huggingface.co/paris-noah).  
> All credit for the model architecture and pre-trained checkpoints belongs to the original authors of  
> *"Mantis: Lightweight Calibrated Foundation Model for User-Friendly Time Series Classification"*.  
>  
> In this repository, we only provide additional training scripts and instructions to facilitate reproducible fine-tuning and pretraining of Mantis on CauKer synthetic data.  
> Please refer to the [official Hugging Face model card](https://huggingface.co/paris-noah/Mantis-8M) and the [paper](https://arxiv.org/abs/2502.15637) for more details about the model itself.
> We would also like to thank **Vasilii Feofanov** for his support.

### Requirements
> mantis-tsfm >= 0.2.0

### How to Use Our Training Code

After setting up the environment, Mantis can be trained in distributed mode with PyTorch DDP using the following command:

```bash
/Path/envs/Mantis/bin/python \
    -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_port=29503 \
    DDPtrain.py \
    --cauker_data_path /path/CauKer.arrow
````

* `--nproc_per_node=4`: number of GPUs to use (modify according to your setup).
* `--master_port`: communication port for distributed training.
* `DDPtrain.py`: the training script we provide, which supports multi-GPU training.

### Configuration

* The model architecture is Mantis-8M.
* Data preprocessing should follow the guidelines in the original Mantis paper (sequence length proportional to 32, with 512 as the default interpolation length).
* Hyperparameters (batch size, learning rate, optimizer, etc.) can be set in the training script or config files.


### Citing Mantis 

If you use Mantis in your work, please cite this technical report:

```bibtex
@article{feofanov2025mantis,
  title={Mantis: Lightweight Calibrated Foundation Model for User-Friendly Time Series Classification},
  author={Vasilii Feofanov and Songkang Wen and Marius Alonso and Romain Ilbert and Hongbo Guo and Malik Tiomoko and Lujia Pan and Jianfeng Zhang and Ievgen Redko},
  journal={arXiv preprint arXiv:2502.15637},
  year={2025},
}
```


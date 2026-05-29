# Unsupervised StrainNet Implementation in PyTorch

## Training the model

This runs on Python 3.11 and requires the following packages:

```
torch
torchvision
tensorboardX
pandas
numpy
```
This work is based on [Boukhtache et al., "When Deep Learning Meets Digital Image Correlation", *Optics and Lasers in Engineering*, 2021](https://www.sciencedirect.com/science/article/pii/S0143816620306588).

The dataset is expected as two CSV annotation files (`Train_annotations.csv` and `Test_annotations.csv`) pointing to pairs of reference/deformed speckle images and their corresponding displacement fields (CSV format). These annotation files and the associated speckle image data can be generated using the dataset generator provided in the [StrainNet repository](https://github.com/DreamIP/StrainNet).

To run training, run:
```bash
python Train.py --arch StrainNet_f --epochs 300 --batch-size 16
```

Three architectures are available via `--arch`: `StrainNet_f` and `StrainNet_h` and `StrainNet_l`.


### Unsupervised loss

The training objective follows the unsupervised formulation of [Yu et al. (ECCV 2016)](https://arxiv.org/abs/1608.05842) and combines:

- **Photometric loss** – Charbonnier penalty on the difference between the reference image and the deformed image warped by the predicted displacement field.
- **Smoothness loss** – Charbonnier penalty on the first-order spatial gradients of the predicted flow, weighted by `--lambda-smooth`.


### Outputs

Training metrics are logged with TensorboardX under a folder named after the run configuration. At each epoch the best checkpoint (lowest validation EPE) is saved as `model_best.pth.tar`, and the latest checkpoint as `checkpoint.pth.tar`.

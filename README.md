# Unsupervised learning for the estimation of deformation fields on the surface of materials subjected to mechanical stress. 

## Motivation

Quantifying surface deformation fields is a core problem in solid mechanics and materials science. Classical approaches such as Digital Image Correlation (DIC) are accurate but slow. This project investigates whether unsupervised deep learning methods, inspired by optical flow architectures like FlowNet, can serve as a faster alternative, without requiring labeled ground-truth data (which is difficult to obtain experimentally).


## Implementations

### 1. Supervised FlowNet (TensorFlow)

A TensorFlow implementation of [FlowNet](https://arxiv.org/abs/1504.06852) (Dosovitskiy et al., 2015), an encoder-decoder CNN for optical flow estimation. Trained with a multi-scale EPE loss on the [FlyingChairs dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html).

**Requirements:** Python 3.11, `tensorflow`, `tqdm`, `matplotlib`

```
python train.py --data_path /your/data/path --epochs 50 --batch_size 8
```

Learning rate is fixed at `1e-4`. Weights are saved as `flownet.weights.h5` and losses logged to `training_log.txt`.



### 2. Unsupervised FlowNet (TensorFlow)

An unsupervised variant of FlowNet based on [Yu et al. (ECCV 2016)](https://arxiv.org/abs/1608.05842), combining a photometric consistency loss with a smoothness regularizer — no ground-truth flow required.

**Requirements:** Python 3, `tensorflow`, `tqdm`, `matplotlib`

```
python train.py --data_path /your/data/path --epochs 50 --batch_size 4 --lr 1.6e-5
```

Can also be trained on the KITTI dataset. Results were not fully satisfactory; hyperparameter sensitivity is a known difficulty.



### 3. Unsupervised StrainNet (PyTorch)

A PyTorch re-implementation of [StrainNet](https://www.sciencedirect.com/science/article/pii/S0143816620306588) (Boukhtache et al., 2021) adapted for unsupervised training. StrainNet shares the FlowNetS architecture but is tailored for speckle-pattern images used in DIC. Three variants are available: `StrainNet_f`, `StrainNet_h`, and `StrainNet_l`.

The unsupervised loss combines:
- **Photometric loss** — Charbonnier penalty on the difference between the reference image and the warped deformed image.
- **Smoothness loss** — Charbonnier penalty on spatial gradients of the predicted flow, weighted by `--lambda-smooth`.

**Requirements:** Python 3.11, `torch`, `torchvision`, `tensorboardX`, `pandas`, `numpy`

Dataset annotation files (`Train_annotations.csv`, `Test_annotations.csv`) can be generated using the [official StrainNet dataset generator](https://github.com/DreamIP/StrainNet).

```
python Train.py --arch StrainNet_f --epochs 300 --batch-size 16
```

Checkpoints are saved as `model_best.pth.tar` (best validation EPE) and `checkpoint.pth.tar` (latest epoch). Metrics are logged via TensorboardX.



### 4. Flow Viewer

A visualization utility for optical flow stored in `.flo` files. Inspired by [flow-code-python](https://github.com/Johswald/flow-code-python).

**Requirements:** Python 3.11, `tensorflow`, `cv2`, `numpy`

**Write a `.flo` file from model weights:**
```
python write.py --frame1 path/to/frame1 --frame2 path/to/frame2 \
    --weights_path path/to/weights --model_path path/to/model.py
```

**Visualize a `.flo` file** (using the Middlebury colour wheel):
```bash
python computeColor.py --flowfile output.flo --write True
```

## Report

The full written report covers:
- Theoretical foundations of optical flow (Horn-Schunck, Lucas-Kanade)
- FlowNet architecture and training
- StrainNet
- Unsupervised learning strategies and their challenges



## References

- Stuart, A. L. (2020). *TensorFlow/Keras implementation of the original FlowNet model*. [GitHub Repository](https://github.com/andrewlstewart/FlowNet_v1_TF2).
- Dosovitskiy, A., Fischer, P., Ilg, E., Häusser, P., Hazırbaş, C., Golkov, V., van der Smagt, P., Cremers, D., & Brox, T. (2015). *FlowNet: Learning Optical Flow with Convolutional Networks*. IEEE International Conference on Computer Vision (ICCV). [arXiv:1504.06852](https://arxiv.org/abs/1504.06852) / [Freiburg Publication](http://lmb.informatik.uni-freiburg.de/Publications/2015/DFIB15).
- Yu, J. J., Harley, A. W., & Derpanis, K. G. (2016). *Back to Basics: Unsupervised Learning of Optical Flow via Brightness Constancy and Motion Smoothness*. European Conference on Computer Vision (ECCV). [arXiv:1608.05842](https://arxiv.org/abs/1608.05842).
- Boukhtache, S., Abdelouahab, K., Berry, F., Blaysat, B., Grédiac, M., & Sur, F. (2021). *When Deep Learning Meets Digital Image Correlation*. *Optics and Lasers in Engineering*, 136, 106308. [DOI: 10.1016/j.optlaseng.2020.106308](https://doi.org/10.1016/j.optlaseng.2020.106308).
- Boukhtache, S., Abdelouahab, K., Bahou, A., Berry, F., Blaysat, B., Grédiac, M., & Sur, F. (2022). *A lightweight convolutional neural network as an alternative to DIC to measure in-plane displacement fields*. *Optics and Lasers in Engineering*.

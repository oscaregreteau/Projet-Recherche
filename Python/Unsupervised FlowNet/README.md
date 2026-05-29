# Unsupervised FlowNet Implementation in TensorFlow

## Training the model

This runs on Python 3. and requires the following packages :
```
tensorflow
tqdm
matplotlib
```

To run the file, just open a terminal in the ```src``` folder and run the following command :
```bash
python train.py --data_path /yourdatapath --epochs 50 --batch_size 4 --lr 1.6e-5
```
The batch size and the learning rate are defined as defined in the [Back to Basics: Unsupervised Learning of Optical Flow via Brightness Constancy and Motion Smoothness](https://arxiv.org/abs/1608.05842) paper.

The loss will be written in a file called ```training_log.txt```. At the end of training, the weights will be saved as ```flownet.weights.h5```.

## Results

After running on the [FlyingChairs Dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html) for 50 epochs, we obtain the following loss curves : 

(insert curves)

This can also be trained on the Kitti Dataset.

The results obtained are not what was expected.
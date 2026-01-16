from typing import Dict, Tuple, Optional, Union
from pathlib import Path
from copy import deepcopy
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from config import CONFIG_FLOWNET, CONFIG_TRAINING
from utils.warp import warp_image
from utils.utils import get_train_val_test, normalize_images
import utils.utils_io as uio
import matplotlib.pyplot as plt

class FlowNet:
    def __init__(self, config):
        self.config = config
        self.model = self._construct_network(config)

    def __getattr__(self, attr):
        return getattr(self.model, attr)

    @staticmethod
    def get_simple_model(config: Dict) -> tf.keras.Model:
        inputs = tf.keras.Input(shape=(384, 512, 6))

        conv_1 = tf.keras.layers.Conv2D(name='conv1', filters=64, kernel_size=7, strides=2, padding='same', activation=tf.keras.activations.relu)(inputs)
        conv_2 = tf.keras.layers.Conv2D(name='conv2', filters=128, kernel_size=5, strides=2, padding='same', activation=tf.keras.activations.relu)(conv_1)
        conv_3 = tf.keras.layers.Conv2D(name='conv3', filters=256, kernel_size=5, strides=2, padding='same', activation=tf.keras.activations.relu)(conv_2)
        conv_3_1 = tf.keras.layers.Conv2D(name='conv3_1', filters=256, kernel_size=3, strides=1, padding='same', activation=tf.keras.activations.relu)(conv_3)
        conv_4 = tf.keras.layers.Conv2D(name='conv4', filters=512, kernel_size=3, strides=2, padding='same', activation=tf.keras.activations.relu)(conv_3_1)
        conv_4_1 = tf.keras.layers.Conv2D(name='conv4_1', filters=512, kernel_size=3, strides=1, padding='same', activation=tf.keras.activations.relu)(conv_4)
        conv_5 = tf.keras.layers.Conv2D(name='conv5', filters=512, kernel_size=3, strides=2, padding='same', activation=tf.keras.activations.relu)(conv_4_1)
        conv_5_1 = tf.keras.layers.Conv2D(name='conv5_1', filters=512, kernel_size=3, strides=1, padding='same', activation=tf.keras.activations.relu)(conv_5)
        conv_6 = tf.keras.layers.Conv2D(name='conv6', filters=1024, kernel_size=3, strides=2, padding='same', activation=tf.keras.activations.relu)(conv_5_1)
        conv_6_1 = tf.keras.layers.Conv2D(name='conv6_1', filters=1024, kernel_size=3, strides=1, padding='same', activation=tf.keras.activations.relu)(conv_6)

        predict_6 = tf.keras.layers.Conv2D(name='predict_6', filters=2, kernel_size=3, strides=1, padding='same', activation=None)(conv_6_1)

        upconv_5 = tf.keras.layers.Conv2DTranspose(name='upconv_5', filters=512, kernel_size=(4, 4), strides=2, padding='same', activation=tf.keras.activations.relu)(conv_6)
        flow_6 = tf.keras.layers.Conv2DTranspose(name='flow_6', filters=2, kernel_size=(4, 4), strides=2, padding='same', activation=tf.keras.activations.relu)(predict_6)
        concat_5 = tf.keras.layers.Concatenate(name='concat_5', axis=-1)([upconv_5, conv_5_1, flow_6])
        predict_5 = tf.keras.layers.Conv2D(name='predict_5', filters=2, kernel_size=3, strides=1, padding='same', activation=None)(concat_5)

        upconv_4 = tf.keras.layers.Conv2DTranspose(name='upconv_4', filters=256, kernel_size=(4, 4), strides=2, padding='same', activation=tf.keras.activations.relu)(concat_5)
        flow_5 = tf.keras.layers.Conv2DTranspose(name='flow_5', filters=2, kernel_size=(4, 4), strides=2, padding='same', activation=tf.keras.activations.relu)(predict_5)
        concat_4 = tf.keras.layers.Concatenate(name='concat_4', axis=-1)([upconv_4, conv_4_1, flow_5])
        predict_4 = tf.keras.layers.Conv2D(name='predict_4', filters=2, kernel_size=3, strides=1, padding='same', activation=None)(concat_4)

        upconv_3 = tf.keras.layers.Conv2DTranspose(name='upconv_3', filters=128, kernel_size=(4, 4), strides=2, padding='same', activation=tf.keras.activations.relu)(concat_4)
        flow_4 = tf.keras.layers.Conv2DTranspose(name='flow_4', filters=2, kernel_size=(4, 4), strides=2, padding='same', activation=tf.keras.activations.relu)(predict_4)
        concat_3 = tf.keras.layers.Concatenate(name='concat_3', axis=-1)([upconv_3, conv_3_1, flow_4])
        predict_3 = tf.keras.layers.Conv2D(name='predict_3', filters=2, kernel_size=3, strides=1, padding='same', activation=None)(concat_3)

        upconv_2 = tf.keras.layers.Conv2DTranspose(name='upconv_2', filters=64, kernel_size=(4, 4), strides=2, padding='same', activation=tf.keras.activations.relu)(concat_3)
        flow_3 = tf.keras.layers.Conv2DTranspose(name='flow_3', filters=2, kernel_size=(4, 4), strides=2, padding='same', activation=tf.keras.activations.relu)(predict_3)
        concat_2 = tf.keras.layers.Concatenate(name='concat_2', axis=-1)([upconv_2, conv_2, flow_3])
        predict_2 = tf.keras.layers.Conv2D(name='predict_2', filters=2, kernel_size=3, strides=1, padding='same', activation=None)(concat_2)

        upconv_1 = tf.keras.layers.Conv2DTranspose(name='upconv_1', filters=64, kernel_size=(4, 4), strides=2, padding='same', activation=tf.keras.activations.relu)(concat_2)
        flow_2 = tf.keras.layers.Conv2DTranspose(name='flow_2', filters=2, kernel_size=(4, 4), strides=2, padding='same', activation=tf.keras.activations.relu)(predict_2)
        concat_1 = tf.keras.layers.Concatenate(name='concat_1', axis=-1)([upconv_1, conv_1, flow_2])
        predict_1 = tf.keras.layers.Conv2D(name='predict_1', filters=2, kernel_size=3, strides=1, padding='same', activation=None)(concat_1)

        if config['training']:
            return tf.keras.Model(inputs=inputs, outputs=[predict_6, predict_5, predict_4, predict_3, predict_2, predict_1])

        return tf.keras.Model(inputs=inputs, outputs=predict_1)

    @staticmethod
    def _construct_network(config: Dict):
        return FlowNet.get_simple_model(config)

class DataGenerator:
    def __init__(self,
                 network_type: str,
                 flo_normalization: Tuple[float, float],
                 root_path: Path,
                 batch_size: int,
                 validation_batch_size: int,
                 train_ratio: Union[float, int] = 1,
                 test_ratio: Union[float, int] = 0,
                 shuffle: bool = False,
                 augmentations: Optional[Dict] = None):
        self.network_type = network_type
        images = list(root_path.glob('*1.ppm'))
        self.train, self.val, self.test = get_train_val_test(images, train_ratio, test_ratio, shuffle)
        self.batch_size = batch_size
        self.validation_batch_size = validation_batch_size
        self.replace = True
        self.flo_normalization = flo_normalization
        self.augmentations = augmentations
        self.output_shape = (384, 512, 2)
    def next_train(self):

        while True:
            images = np.random.choice(self.train, self.batch_size, replace=self.replace)
            img1 = [uio.read(str(img)) for img in images]
            img2 = [uio.read(str(img).replace('1.ppm', '2.ppm')) for img in images]

            img1 = normalize_images(img1)
            img2 = normalize_images(img2)

            # if not self.augmentations is None:
            #     img1, img2 = self._augment(img1, img2)

            if self.network_type == 'simple':
                images = np.concatenate([img1, img2], axis=-1)
            elif self.network_type == 'correlation':
                raise NotImplementedError()
            else:
                raise MalformedNetworkType(f'{self.network_type}: {MalformedNetworkType.__doc__}')
            dummy_targets = np.zeros((self.batch_size, *self.output_shape))
            dummy_targets_list = tuple([dummy_targets] * 6 ) 
            yield (images, dummy_targets_list)


    def next_val(self):
        while True:
            images = np.random.choice(self.val, self.validation_batch_size, replace=False)
            img1 = [uio.read(str(img)) for img in images]
            img2 = [uio.read(str(img).replace('1.ppm', '2.ppm')) for img in images]

            img1 = normalize_images(img1)
            img2 = normalize_images(img2)

            if self.network_type == 'simple':
                images = np.concatenate([img1, img2], axis=-1)
            elif self.network_type == 'correlation':
                raise NotImplementedError()
            else:
                raise MalformedNetworkType(f'{self.network_type}: {MalformedNetworkType.__doc__}')

            yield (images)

class lossphoto(tf.keras.losses.Loss):

    def warp_image(img, flow):
        B,H,W,_ = tf.shape(img)
        grid_x, grid_y = tf.meshgrid(tf.range(W), tf.range(H))
        coords = tf.stack([grid_x, grid_y], axis=-1) 
        coords = tf.cast(coords, tf.float32)
        coords = coords[None] + flow                  
        coords_x = 2.0 * coords[...,0] / (W-1) - 1.0
        coords_y = 2.0 * coords[...,1] / (H-1) - 1.0
        grid = tf.stack([coords_x, coords_y], axis=-1)
        warped = tfa.image.resampler(img, grid)
        return warped


    def call(self, y_true, y_pred):
        image1, image2 = tf.split(y_true, num_or_size_splits=2, axis=-1)
        H_pred, W_pred = tf.shape(y_pred)[1], tf.shape(y_pred)[2]
        image1_small = tf.image.resize(image1, [H_pred, W_pred])
        image2_small = tf.image.resize(image2, [H_pred, W_pred])

        warped_I2 = warp_image(image2_small, y_pred)
        photometric_loss = tf.reduce_mean(tf.abs(image1_small - warped_I2))

        dx = tf.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dy = tf.abs(y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :])
        smoothness_loss = tf.reduce_mean(dx) + tf.reduce_mean(dy)
        lambda_smooth = 0.5
        return photometric_loss + lambda_smooth * smoothness_loss



def main():
    config_network = deepcopy(CONFIG_FLOWNET)
    config_training = deepcopy(CONFIG_TRAINING)
    flownet = FlowNet(config_network)
    loss=lossphoto()
    flownet.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=[loss,loss,loss,loss,loss,loss],
        loss_weights=config_training['loss_weights'][::-1]
    )
    data_generator = DataGenerator(config_network['architecture'],
                                   config_network['flo_normalization'],
                                   Path(r'/Users/oscar/Downloads/FlyingChairs_release/data'),
                                   config_training['batch_size'],
                                   config_training['validation_batch_size'],
                                   config_training['train_ratio'],
                                   config_training['test_ratio'],
                                   config_training['shuffle'],
                                   config_training['augmentations'])
    history = flownet.fit(
        data_generator.next_train(),
        steps_per_epoch=200//config_training['batch_size'],
        epochs=1
    )
    flownet.save_weights('unsupflow.weights.h5')
    for key in history.history:
        if 'loss' in key and key != 'loss':  # skip overall loss
            plt.plot(history.history[key], label=key)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Per-output Loss per Epoch')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

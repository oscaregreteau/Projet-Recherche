from typing import Dict, Tuple, Optional, Union
from pathlib import Path
from copy import deepcopy
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from config import CONFIG_FLOWNET, CONFIG_TRAINING
from utils.warp import warp_image as warp
from utils.utils import get_train_val_test, normalize_images
import utils.utils_io as uio

class FlowNet:
    def __init__(self, config):
        self.config = config
        self.model = self._construct_network(config)

    # def call(self, inputs, training=False):
    #     return self.backbone(inputs, training=training)
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
    """ Instantiate then call instance.next_train() to get a generator for training images/labels
            call instance.next_val() to get a generator for validation images/labels
    """

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

    # def _augment(self, img1, img2):
    #     img1 = tf.convert_to_tensor(img1, dtype=tf.float32)
    #     img2 = tf.convert_to_tensor(img2, dtype=tf.float32)
    #     # Augmentations are more awkward because of the Siamese architecture, I can't justify applying different color transforms to each image independently
    #     # I'm 100 certain there is a better way to do this as this is extremely inefficient with each call likely containing some portion of each other call.
    #     r = np.random.rand(len(self.augmentations))
    #     r_inc = 0  # This, with r, are used to randomly turn on/off augmentations so that not every augmentation is applied each time
    #     r_onoff = 2/5
    #     if 'brightness' in self.augmentations and r[r_inc] <= r_onoff:
    #         rdm = np.random.rand(self.batch_size) * self.augmentations['brightness']
    #         def brt(x, idx): return tf.image.adjust_brightness(x, rdm[idx])
    #         img1 = tf.stack([brt(im, idx) for idx, im in enumerate(img1)], axis=0)
    #         img2 = tf.stack([brt(im, idx) for idx, im in enumerate(img2)], axis=0)
    #         r_inc += 1
    #     if 'multiplicative_colour' in self.augmentations and r[r_inc] <= r_onoff:
    #         rdm = np.random.rand(self.batch_size, 3) * (self.augmentations['multiplicative_colour'][1] -
    #                                                     self.augmentations['multiplicative_colour'][0]) + self.augmentations['multiplicative_colour'][0]

    #         def mc(x, idx): return x * rdm[idx]
    #         img1 = tf.clip_by_value(tf.stack([mc(im, idx) for idx, im in enumerate(img1)], axis=0), clip_value_min=0, clip_value_max=1)
    #         img2 = tf.clip_by_value(tf.stack([mc(im, idx) for idx, im in enumerate(img2)], axis=0), clip_value_min=0, clip_value_max=1)
    #         r_inc += 1
    #     if 'gamma' in self.augmentations and r[r_inc] <= r_onoff:
    #         rdm = np.random.rand(self.batch_size) * (self.augmentations['gamma'][1] - self.augmentations['gamma'][0]) + self.augmentations['gamma'][0]
    #         def gam(x, idx): return tf.image.adjust_gamma(x, gamma=rdm[idx])
    #         img1 = tf.stack([gam(im, idx) for idx, im in enumerate(img1)], axis=0)
    #         img2 = tf.stack([gam(im, idx) for idx, im in enumerate(img2)], axis=0)
    #         r_inc += 1
    #     if 'contrast' in self.augmentations and r[r_inc] <= r_onoff:
    #         rdm = np.random.rand(self.batch_size) * (self.augmentations['contrast'][1] - self.augmentations['contrast'][0]) + self.augmentations['contrast'][0]
    #         def cts(x, idx): return tf.image.adjust_contrast(x, contrast_factor=rdm[idx])
    #         img1 = tf.stack([cts(im, idx) for idx, im in enumerate(img1)], axis=0)
    #         img2 = tf.stack([cts(im, idx) for idx, im in enumerate(img2)], axis=0)
    #         r_inc += 1
    #     if 'gaussian_noise' in self.augmentations and r[r_inc] <= r_onoff:
    #         rdm = np.random.rand(self.batch_size) * self.augmentations['gaussian_noise']
    #         def gau(x, idx): return x + tf.random.normal(x.shape, mean=0.0, stddev=rdm[idx], dtype=x.dtype)
    #         img1 = tf.clip_by_value(tf.stack([gau(im, idx) for idx, im in enumerate(img1)], axis=0), clip_value_min=0, clip_value_max=1)
    #         img2 = tf.clip_by_value(tf.stack([gau(im, idx) for idx, im in enumerate(img2)], axis=0), clip_value_min=0, clip_value_max=1)
    #         r_inc += 1

    #     return img1, img2

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
        flow_pred = y_pred  # [B,H,W,2]
        warped_I2 = warp.warp_image(image2, flow_pred)
        photometric_loss = tf.reduce_mean(tf.abs(image1 - warped_I2))
        dx = tf.abs(flow_pred[:, :, 1:, :] - flow_pred[:, :, :-1, :])
        dy = tf.abs(flow_pred[:, 1:, :, :] - flow_pred[:, :-1, :, :])
        smoothness_loss = tf.reduce_mean(dx) + tf.reduce_mean(dy)
        return photometric_loss + self.lambda_smooth * smoothness_loss


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
    flownet.fit(
        data_generator.next_train(),
        steps_per_epoch=200 // config_training['batch_size'],
        epochs=10
    )


if __name__ == "__main__":
    main()

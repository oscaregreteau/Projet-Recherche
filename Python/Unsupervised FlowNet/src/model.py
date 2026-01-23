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
from huggingface_hub import HfApi, login
import os


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

        upconv_5 = tf.keras.layers.Conv2DTranspose(name='upconv_5', filters=512, kernel_size=(4, 4), strides=2, padding='same', activation=tf.keras.activations.relu)(conv_6_1)
        flow_6 = tf.keras.layers.Conv2DTranspose(name='flow_6', filters=2, kernel_size=(4, 4), strides=2, padding='same', activation=None)(predict_6)#tf.keras.activations.relu
        concat_5 = tf.keras.layers.Concatenate(name='concat_5', axis=-1)([upconv_5, conv_5_1, flow_6])
        predict_5 = tf.keras.layers.Conv2D(name='predict_5', filters=2, kernel_size=3, strides=1, padding='same', activation=None)(concat_5)

        upconv_4 = tf.keras.layers.Conv2DTranspose(name='upconv_4', filters=256, kernel_size=(4, 4), strides=2, padding='same', activation=tf.keras.activations.relu)(concat_5)
        flow_5 = tf.keras.layers.Conv2DTranspose(name='flow_5', filters=2, kernel_size=(4, 4), strides=2, padding='same', activation=None)(predict_5)#tf.keras.activations.relu
        concat_4 = tf.keras.layers.Concatenate(name='concat_4', axis=-1)([upconv_4, conv_4_1, flow_5])
        predict_4 = tf.keras.layers.Conv2D(name='predict_4', filters=2, kernel_size=3, strides=1, padding='same', activation=None)(concat_4)

        upconv_3 = tf.keras.layers.Conv2DTranspose(name='upconv_3', filters=128, kernel_size=(4, 4), strides=2, padding='same', activation=tf.keras.activations.relu)(concat_4)
        flow_4 = tf.keras.layers.Conv2DTranspose(name='flow_4', filters=2, kernel_size=(4, 4), strides=2, padding='same', activation=None)(predict_4)#tf.keras.activations.relu
        concat_3 = tf.keras.layers.Concatenate(name='concat_3', axis=-1)([upconv_3, conv_3_1, flow_4])
        predict_3 = tf.keras.layers.Conv2D(name='predict_3', filters=2, kernel_size=3, strides=1, padding='same', activation=None)(concat_3)

        upconv_2 = tf.keras.layers.Conv2DTranspose(name='upconv_2', filters=64, kernel_size=(4, 4), strides=2, padding='same', activation=tf.keras.activations.relu)(concat_3)
        flow_3 = tf.keras.layers.Conv2DTranspose(name='flow_3', filters=2, kernel_size=(4, 4), strides=2, padding='same', activation=None)(predict_3)#tf.keras.activations.relu
        concat_2 = tf.keras.layers.Concatenate(name='concat_2', axis=-1)([upconv_2, conv_2, flow_3])
        predict_2 = tf.keras.layers.Conv2D(name='predict_2', filters=2, kernel_size=3, strides=1, padding='same', activation=None)(concat_2)

        upconv_1 = tf.keras.layers.Conv2DTranspose(name='upconv_1', filters=64, kernel_size=(4, 4), strides=2, padding='same', activation=tf.keras.activations.relu)(concat_2)
        flow_2 = tf.keras.layers.Conv2DTranspose(name='flow_2', filters=2, kernel_size=(4, 4), strides=2, padding='same', activation=None)(predict_2)#tf.keras.activations.relu
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

            if self.network_type == 'simple':
                images = np.concatenate([img1, img2], axis=-1)
            elif self.network_type == 'correlation':
                raise NotImplementedError()
            else:
                raise MalformedNetworkType(f'{self.network_type}: {MalformedNetworkType.__doc__}')
            targets = np.concatenate([img1, img2], axis=-1)  # (B,H,W,6)
            targets_list = tuple([targets] * 6)
            yield images, targets_list  



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

    @staticmethod
    def charbonnier(x, alpha=0.25):
        eps = 1e-5
        return tf.pow(x * x + eps * eps, alpha)

    def call(self, y_true, y_pred):
        image1, image2 = tf.split(y_true, 2, axis=-1)

        H = tf.shape(image1)[1]
        W = tf.shape(image1)[2]
        h = tf.shape(y_pred)[1]
        w = tf.shape(y_pred)[2]

        image1_s = tf.image.resize(image1, [h, w])
        image2_s = tf.image.resize(image2, [h, w])

        scale_x = tf.cast(W, tf.float32) / tf.cast(w, tf.float32)
        scale_y = tf.cast(H, tf.float32) / tf.cast(h, tf.float32)

        flow_scaled = tf.stack([
            y_pred[..., 0] * scale_x,
            y_pred[..., 1] * scale_y
        ], axis=-1)

        warped_I2 = warp_image(image2_s, flow_scaled)

        photo = tf.reduce_mean(self.charbonnier(image1_s - warped_I2, 0.25))

        dx = self.charbonnier(flow_scaled[:, :, 1:, :] - flow_scaled[:, :, :-1, :], 0.37)
        dy = self.charbonnier(flow_scaled[:, 1:, :, :] - flow_scaled[:, :-1, :, :], 0.37)

        smooth = tf.reduce_mean(dx) + tf.reduce_mean(dy)

        return photo + smooth



# login(token="")

REPO_ID = "oscaregreteau/flow"
api = HfApi()

class HuggingFaceCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, log_file="loss_log.txt"):
        super().__init__()
        self.last_weight_file = None
        self.log_file = log_file

        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                f.write(
                    "epoch,loss," +
                    ",".join([f"predict_{i}_loss" for i in range(1, 7)]) +
                    "\n"
                )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        epoch_num = epoch + 1

        weight_file = f"weights_epoch_{epoch_num}.weights.h5"
        self.model.save_weights(weight_file)

        api.upload_file(
            path_or_fileobj=weight_file,
            path_in_repo=weight_file,
            repo_id=REPO_ID,
            repo_type="model",
            commit_message=f"Update weights (epoch {epoch_num})"
        )

        if self.last_weight_file is not None:
            try:
                api.delete_file(
                    repo_id=REPO_ID,
                    path_in_repo=self.last_weight_file,
                    repo_type="model",
                    commit_message="Remove previous weights"
                )
            except Exception as e:
                print(f"Warning: could not delete previous weights: {e}")

        os.remove(weight_file)
        self.last_weight_file = weight_file

        with open(self.log_file, 'a') as f:
            overall_loss = logs.get('loss', 0)
            per_output_losses = [logs.get(f'predict_{i}_loss', 0) for i in range(1, 7)]
            f.write(
                f"{epoch_num},{overall_loss:.6f}," +
                ",".join(f"{v:.6f}" for v in per_output_losses) +
                "\n"
            )

        api.upload_file(
            path_or_fileobj=self.log_file,
            path_in_repo=self.log_file,
            repo_id=REPO_ID,
            repo_type="model",
            commit_message=f"Update loss log after epoch {epoch_num}"
        )

        print(f"Epoch {epoch_num}: latest weights + loss log uploaded")




def main():
    config_network = deepcopy(CONFIG_FLOWNET)
    config_training = deepcopy(CONFIG_TRAINING)
    flownet = FlowNet(config_network)
    loss=lossphoto()

    data_generator = DataGenerator(config_network['architecture'],
                                   config_network['flo_normalization'],
                                   Path(r'/Users/oscar/Downloads/FlyingChairs_release/data'),
                                   config_training['batch_size'],
                                   config_training['validation_batch_size'],
                                   config_training['train_ratio'],
                                   config_training['test_ratio'],
                                   config_training['shuffle'],
                                   config_training['augmentations'])
    checkpoint_cb = HuggingFaceCheckpoint()
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1.6e-5, decay_steps=100000, decay_rate=0.5, staircase=True)
    flownet.compile(
        optimizer=tf.keras.optimizers.Adam(lr_schedule), 
        loss=[loss] * 6,
        loss_weights=config_training['loss_weights'][::-1]
    )

    history = flownet.fit(
        data_generator.next_train(),
        steps_per_epoch=22572 // config_training['batch_size'],
        epochs=5#,
        #callbacks=[checkpoint_cb]
    )

    flownet.save_weights("flownet.weights.h5")
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

from typing import Dict, Tuple, Optional, Union
from pathlib import Path
from copy import deepcopy
from datetime import datetime
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

import utils_io as uio
import utils
from config import CONFIG_FLOWNET, CONFIG_TRAINING
import matplotlib.pyplot as plt
from huggingface_hub import HfApi, login



class FlowNet:
    def __init__(self, config: Dict):
        self.config = config
        self.model = self.get_simple_model(config)

    def __getattr__(self, attr):
        return getattr(self.model, attr)

    @staticmethod
    def get_simple_model(config: Dict) -> tf.keras.Model:
        inputs = tf.keras.Input(shape=(384, 512, 6))

        conv_1 = tf.keras.layers.Conv2D(32, 7, strides=2, padding='same', activation='relu', name='conv1')(inputs)
        conv_2 = tf.keras.layers.Conv2D(64, 5, strides=2, padding='same', activation='relu', name='conv2')(conv_1)
        conv_3 = tf.keras.layers.Conv2D(128, 5, strides=2, padding='same', activation='relu', name='conv3')(conv_2)

        predict_3 = tf.keras.layers.Conv2D(2, 3, strides=1, padding='same', activation=None, name='predict_3')(conv_3)

        upconv_2 = tf.keras.layers.Conv2DTranspose(64, 4, strides=2, padding='same', activation='relu', name='upconv_2')(conv_3)
        flow_3 = tf.keras.layers.Conv2DTranspose(2, 4, strides=2, padding='same', activation='relu', name='flow_3')(predict_3)
        concat_2 = tf.keras.layers.Concatenate(axis=-1, name='concat_2')([upconv_2, conv_2, flow_3])
        predict_2 = tf.keras.layers.Conv2D(2, 3, strides=1, padding='same', activation=None, name='predict_2')(concat_2)

        upconv_1 = tf.keras.layers.Conv2DTranspose(32, 4, strides=2, padding='same', activation='relu', name='upconv_1')(concat_2)
        flow_2 = tf.keras.layers.Conv2DTranspose(2, 4, strides=2, padding='same', activation='relu', name='flow_2')(predict_2)
        concat_1 = tf.keras.layers.Concatenate(axis=-1, name='concat_1')([upconv_1, conv_1, flow_2])
        predict_1 = tf.keras.layers.Conv2D(2, 3, strides=1, padding='same', activation=None, name='predict_1')(concat_1)

        if config['training']:
            return tf.keras.Model(inputs=inputs, outputs=[predict_3, predict_2, predict_1])

        return tf.keras.Model(inputs=inputs, outputs=predict_1)


    def disable_training(self):
        self.model = tf.keras.Model(
            inputs=self.model.input,
            outputs=self.model.output[-1]
        )

    def enable_training(self):
        output_layers = [layer.output for layer in self.model.layers if 'predict' in layer.name]
        self.model = tf.keras.Model(inputs=self.model.layers[0].input, outputs=output_layers)



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
        self.flo_normalization = flo_normalization
        self.batch_size = batch_size
        self.validation_batch_size = validation_batch_size
        self.augmentations = augmentations
        self.replace = True

        # Load and split image paths
        images = list(root_path.glob('*1.ppm'))
        self.train, self.val, self.test = utils.get_train_val_test(
            images, train_ratio, test_ratio, shuffle
        )

    def _load_batch(self, image_paths, batch_size):
        """Load a batch of image pairs and flow labels."""
        images = np.random.choice(
            image_paths, 
            batch_size, 
            replace=(batch_size > len(image_paths)) or self.replace
        )
        
        # Read images and flow
        img1 = [uio.read(str(img)) for img in images]
        img2 = [uio.read(str(img).replace('1.ppm', '2.ppm')) for img in images]
        label = [uio.read(str(img).replace('img1.ppm', 'flow.flo')) for img in images]

        # Normalize
        img1 = utils.normalize_images(img1)
        img2 = utils.normalize_images(img2)
        label = utils.normalize_flo(label, self.flo_normalization)

        return img1, img2, label

    def _prepare_inputs(self, img1, img2):
        """Prepare network inputs based on architecture type."""
        if self.network_type == 'simple':
            return np.concatenate([img1, img2], axis=-1)
        elif self.network_type == 'correlation':
            raise NotImplementedError('Correlation network type not yet implemented')
        else:
            raise ValueError(f'Unknown network type: {self.network_type}')

    def _augment(self, img1, img2, label):
        """Apply data augmentations to image pairs."""
        if self.augmentations is None:
            return img1, img2, label

        # Probability for each augmentation
        r = np.random.rand(len(self.augmentations))
        r_threshold = 0.4  # 40% chance for each augmentation
        r_idx = 0

        # Brightness augmentation
        if 'brightness' in self.augmentations and r[r_idx] <= r_threshold:
            rdm = np.random.rand(self.batch_size) * self.augmentations['brightness']
            img1 = tf.stack([
                tf.image.adjust_brightness(im, rdm[idx]) 
                for idx, im in enumerate(img1)
            ], axis=0)
            img2 = tf.stack([
                tf.image.adjust_brightness(im, rdm[idx]) 
                for idx, im in enumerate(img2)
            ], axis=0)
            r_idx += 1

        # Multiplicative color augmentation
        if 'multiplicative_colour' in self.augmentations and r[r_idx] <= r_threshold:
            min_val, max_val = self.augmentations['multiplicative_colour']
            rdm = np.random.rand(self.batch_size, 3) * (max_val - min_val) + min_val
            img1 = tf.clip_by_value(
                tf.stack([im * rdm[idx] for idx, im in enumerate(img1)], axis=0),
                clip_value_min=0, 
                clip_value_max=1
            )
            img2 = tf.clip_by_value(
                tf.stack([im * rdm[idx] for idx, im in enumerate(img2)], axis=0),
                clip_value_min=0, 
                clip_value_max=1
            )
            r_idx += 1

        # Gamma augmentation
        if 'gamma' in self.augmentations and r[r_idx] <= r_threshold:
            min_val, max_val = self.augmentations['gamma']
            rdm = np.random.rand(self.batch_size) * (max_val - min_val) + min_val
            img1 = tf.stack([
                tf.image.adjust_gamma(im, gamma=rdm[idx]) 
                for idx, im in enumerate(img1)
            ], axis=0)
            img2 = tf.stack([
                tf.image.adjust_gamma(im, gamma=rdm[idx]) 
                for idx, im in enumerate(img2)
            ], axis=0)
            r_idx += 1

        # Contrast augmentation
        if 'contrast' in self.augmentations and r[r_idx] <= r_threshold:
            min_val, max_val = self.augmentations['contrast']
            rdm = np.random.rand(self.batch_size) * (max_val - min_val) + min_val
            img1 = tf.stack([
                tf.image.adjust_contrast(im, contrast_factor=rdm[idx]) 
                for idx, im in enumerate(img1)
            ], axis=0)
            img2 = tf.stack([
                tf.image.adjust_contrast(im, contrast_factor=rdm[idx]) 
                for idx, im in enumerate(img2)
            ], axis=0)
            r_idx += 1

        # Gaussian noise augmentation
        if 'gaussian_noise' in self.augmentations and r[r_idx] <= r_threshold:
            rdm = np.random.rand(self.batch_size) * self.augmentations['gaussian_noise']
            img1 = tf.clip_by_value(
                tf.stack([
                    im + tf.random.normal(im.shape, mean=0.0, stddev=rdm[idx], dtype=im.dtype)
                    for idx, im in enumerate(img1)
                ], axis=0),
                clip_value_min=0, 
                clip_value_max=1
            )
            img2 = tf.clip_by_value(
                tf.stack([
                    im + tf.random.normal(im.shape, mean=0.0, stddev=rdm[idx], dtype=im.dtype)
                    for idx, im in enumerate(img2)
                ], axis=0),
                clip_value_min=0, 
                clip_value_max=1
            )
            r_idx += 1

        return img1, img2, label

    def get_train_dataset(self):
        def train_generator():
            while True:
                # Load batch
                img1, img2, label = self._load_batch(self.train, self.batch_size)
                
                # Apply augmentations
                img1, img2, label = self._augment(img1, img2, label)
                
                # Prepare network inputs
                inputs = self._prepare_inputs(img1, img2)
                labels = np.array(label)
                
                # Resize labels to match each output resolution
                # predict_3: 48×64, predict_2: 96×128, predict_1: 192×256
                labels_3 = tf.image.resize(labels, [48, 64]).numpy()
                labels_2 = tf.image.resize(labels, [96, 128]).numpy()
                labels_1 = tf.image.resize(labels, [192, 256]).numpy()
                
                # Yield with multi-scale outputs
                yield (inputs, (labels_3, labels_2, labels_1))
    
        return tf.data.Dataset.from_generator(
            train_generator,
            output_signature=(
                tf.TensorSpec(shape=(self.batch_size, 384, 512, 6), dtype=tf.float32),
                (
                    tf.TensorSpec(shape=(self.batch_size, 48, 64, 2), dtype=tf.float32),   # predict_3
                    tf.TensorSpec(shape=(self.batch_size, 96, 128, 2), dtype=tf.float32),  # predict_2
                    tf.TensorSpec(shape=(self.batch_size, 192, 256, 2), dtype=tf.float32)  # predict_1
                )
            )
        )

    def get_val_dataset(self):
        """Create a tf.data.Dataset for validation."""
        def val_generator():
            while True:
                # Load batch (no replacement for validation)
                img1, img2, label = self._load_batch(self.val, self.validation_batch_size)
                
                # No augmentations for validation
                inputs = self._prepare_inputs(img1, img2)
                labels = np.array(label)
                
                yield (inputs, labels)
        
        return tf.data.Dataset.from_generator(
            val_generator,
            output_signature=(
                tf.TensorSpec(shape=(self.validation_batch_size, 384, 512, 6), dtype=tf.float32),
                tf.TensorSpec(shape=(self.validation_batch_size, 192, 256, 2), dtype=tf.float32)
            )
        )

    def next_train(self):
        """Legacy generator for training (kept for backwards compatibility)."""
        while True:
            img1, img2, label = self._load_batch(self.train, self.batch_size)
            
            if self.augmentations is not None:
                img1, img2, label = self._augment(img1, img2, label)
            
            inputs = self._prepare_inputs(img1, img2)
            labels = np.array(label)
            
            yield (inputs, (labels, labels, labels))

    def next_val(self):
        """Legacy generator for validation (kept for backwards compatibility)."""
        while True:
            img1, img2, label = self._load_batch(self.val, self.validation_batch_size)
            inputs = self._prepare_inputs(img1, img2)
            labels = np.array(label)
            
            yield (inputs, labels)

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


class EndPointError(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return K.mean(K.sqrt(K.sum(K.square(tf.image.resize(y_true, y_pred.shape[1:3]) - y_pred), axis=-1)))
    

def main():
    config_network = deepcopy(CONFIG_FLOWNET)
    config_training = deepcopy(CONFIG_TRAINING)
    flownet = FlowNet(config_network)
    
    data_generator = DataGenerator(
        config_network['architecture'],
        config_network['flo_normalization'],
        Path(r'/Users/oscar/Downloads/FlyingChairs_release/data'),
        config_training['batch_size'],
        config_training['validation_batch_size'],
        config_training['train_ratio'],
        config_training['test_ratio'],
        config_training['shuffle'],
        config_training['augmentations']
    )
    
    checkpoint_cb = HuggingFaceCheckpoint()
    
    # FIX: Create separate loss instances for each output
    # This prevents TensorFlow from trying to batch the resize operations
    # which causes shape conflicts between the three different resolutions
    flownet.compile(
        optimizer=tf.keras.optimizers.Adam(1.6e-5), 
        loss=[EndPointError(), EndPointError(), EndPointError()],  # 3 separate instances
        loss_weights=config_training['loss_weights'][:3]  # First 3 weights
    )

    history = flownet.fit(
        data_generator.get_train_dataset(),
        steps_per_epoch=200 // config_training['batch_size'],
        epochs=5,
        callbacks=[checkpoint_cb]
    )

    flownet.save_weights("flownet.weights.h5")
    
    # Plot losses
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

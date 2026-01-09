""" FlowNet model written in TF2/Keras
    https://arxiv.org/pdf/1504.06852.pdf
"""

from typing import Dict, Tuple, Optional, Union
from pathlib import Path
from copy import deepcopy
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

import utils_io as uio
import utils
# from utils import visualize_flownet_prediction
from config import CONFIG_FLOWNET, CONFIG_TRAINING
import matplotlib.pyplot as plt


class MalformedNetworkType(Exception):
    """The provided network type doesn't match one of 'simple' or 'correlation'."""


class FlowNet:
    """ FlowNetSimple model from the Computer Vision Group of Freiburg.
        https://lmb.informatik.uni-freiburg.de/
        https://lmb.informatik.uni-freiburg.de/Publications/2015/DFIB15/flownet.pdf
    """

    def __init__(self, config: Dict):
        self.config = config

        self.model = self._construct_network(config)

    def __getattr__(self, attr):
        """ Rather than potentially override any of the tf.keras.Model methods by subclassing and defining new methods,
            create a composition class with self.model:tf.keras.Model and allow attribute calls directly against self.model
        """
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
        """Switch model to single-output inference mode"""
        self.model = tf.keras.Model(
            inputs=self.model.input,
            outputs=self.model.output[-1]
        )

    def enable_training(self):
        """ If you need to re-enable training, run this method to have self.model predict the list of 6 predictions
        """
        output_layers = [layer.output for layer in self.model.layers if 'predict' in layer.name]
        self.model = tf.keras.Model(inputs=self.model.layers[0].input, outputs=output_layers)

    @staticmethod
    def get_corr_model(config: Dict) -> tf.keras.Model:
        raise NotImplementedError("The correlation model hasn't been implemented.")

    @staticmethod
    def _construct_network(config: Dict) -> tf.keras.Model:
        if config['architecture'] == 'simple':
            return FlowNet.get_simple_model(config)
        if config['architecture'] == 'corr':
            return FlowNet.get_corr_model(config)

        raise MalformedNetworkType(f"{config['architecture']}: {MalformedNetworkType.__doc__}")


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
        self.train, self.val, self.test = utils.get_train_val_test(images, train_ratio, test_ratio, shuffle)
        self.batch_size = batch_size
        self.validation_batch_size = validation_batch_size
        self.replace = True
        self.flo_normalization = flo_normalization
        self.augmentations = augmentations
        
    def next_train(self):

        while True:
            images = np.random.choice(self.train, self.batch_size, replace=self.replace)
            img1 = [uio.read(str(img)) for img in images]
            img2 = [uio.read(str(img).replace('1.ppm', '2.ppm')) for img in images]
            label = [uio.read(str(img).replace('img1.ppm', 'flow.flo')) for img in images]

            img1 = utils.normalize_images(img1)
            img2 = utils.normalize_images(img2)
            label = utils.normalize_flo(label, self.flo_normalization)

            if not self.augmentations is None:
                img1, img2, label = self._augment(img1, img2, label)

            if self.network_type == 'simple':
                images = np.concatenate([img1, img2], axis=-1)
            elif self.network_type == 'correlation':
                raise NotImplementedError()
            else:
                raise MalformedNetworkType(f'{self.network_type}: {MalformedNetworkType.__doc__}')

            yield (images, np.array(label))

    def next_val(self):

        while True:
            images = np.random.choice(self.val, self.validation_batch_size, replace=False)
            img1 = [uio.read(str(img)) for img in images]
            img2 = [uio.read(str(img).replace('1.ppm', '2.ppm')) for img in images]
            label = [uio.read(str(img).replace('img1.ppm', 'flow.flo')) for img in images]

            img1 = utils.normalize_images(img1)
            img2 = utils.normalize_images(img2)
            label = utils.normalize_flo(label, self.flo_normalization)

            if self.network_type == 'simple':
                images = np.concatenate([img1, img2], axis=-1)
            elif self.network_type == 'correlation':
                raise NotImplementedError()
            else:
                raise MalformedNetworkType(f'{self.network_type}: {MalformedNetworkType.__doc__}')

            yield (images, np.array(label))

    def _augment(self, img1, img2, label):
        # Augmentations are more awkward because of the Siamese architecture, I can't justify applying different color transforms to each image independently
        # I'm 100 certain there is a better way to do this as this is extremely inefficient with each call likely containing some portion of each other call.
        r = np.random.rand(len(self.augmentations))
        r_inc = 0  # This, with r, are used to randomly turn on/off augmentations so that not every augmentation is applied each time
        r_onoff = 2/5
        if 'brightness' in self.augmentations and r[r_inc] <= r_onoff:
            rdm = np.random.rand(self.batch_size) * self.augmentations['brightness']
            def brt(x, idx): return tf.image.adjust_brightness(x, rdm[idx])
            img1 = tf.stack([brt(im, idx) for idx, im in enumerate(img1)], axis=0)
            img2 = tf.stack([brt(im, idx) for idx, im in enumerate(img2)], axis=0)
            r_inc += 1
        if 'multiplicative_colour' in self.augmentations and r[r_inc] <= r_onoff:
            rdm = np.random.rand(self.batch_size, 3) * (self.augmentations['multiplicative_colour'][1] -
                                                        self.augmentations['multiplicative_colour'][0]) + self.augmentations['multiplicative_colour'][0]

            def mc(x, idx): return x * rdm[idx]
            img1 = tf.clip_by_value(tf.stack([mc(im, idx) for idx, im in enumerate(img1)], axis=0), clip_value_min=0, clip_value_max=1)
            img2 = tf.clip_by_value(tf.stack([mc(im, idx) for idx, im in enumerate(img2)], axis=0), clip_value_min=0, clip_value_max=1)
            r_inc += 1
        if 'gamma' in self.augmentations and r[r_inc] <= r_onoff:
            rdm = np.random.rand(self.batch_size) * (self.augmentations['gamma'][1] - self.augmentations['gamma'][0]) + self.augmentations['gamma'][0]
            def gam(x, idx): return tf.image.adjust_gamma(x, gamma=rdm[idx])
            img1 = tf.stack([gam(im, idx) for idx, im in enumerate(img1)], axis=0)
            img2 = tf.stack([gam(im, idx) for idx, im in enumerate(img2)], axis=0)
            r_inc += 1
        if 'contrast' in self.augmentations and r[r_inc] <= r_onoff:
            rdm = np.random.rand(self.batch_size) * (self.augmentations['contrast'][1] - self.augmentations['contrast'][0]) + self.augmentations['contrast'][0]
            def cts(x, idx): return tf.image.adjust_contrast(x, contrast_factor=rdm[idx])
            img1 = tf.stack([cts(im, idx) for idx, im in enumerate(img1)], axis=0)
            img2 = tf.stack([cts(im, idx) for idx, im in enumerate(img2)], axis=0)
            r_inc += 1
        if 'gaussian_noise' in self.augmentations and r[r_inc] <= r_onoff:
            rdm = np.random.rand(self.batch_size) * self.augmentations['gaussian_noise']
            def gau(x, idx): return x + tf.random.normal(x.shape, mean=0.0, stddev=rdm[idx], dtype=x.dtype)
            img1 = tf.clip_by_value(tf.stack([gau(im, idx) for idx, im in enumerate(img1)], axis=0), clip_value_min=0, clip_value_max=1)
            img2 = tf.clip_by_value(tf.stack([gau(im, idx) for idx, im in enumerate(img2)], axis=0), clip_value_min=0, clip_value_max=1)
            r_inc += 1

        return img1, img2, label


class EndPointError(tf.keras.losses.Loss):
    """ EndPointError is the Euclidean distance between the predicted flow vector and the ground truth averaged over all pixels.
        The resizing is required because the loss is calculated for each flow prediction which occur at different stride levels,
        resizing effectively averages at that scale.
    """

    def call(self, y_true, y_pred):
        return K.sqrt(K.sum(K.square(tf.image.resize(y_true, y_pred.shape[1:3]) - y_pred), axis=1, keepdims=True))
    




def show_images(simple_images, label):
    """
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(ncols=2, nrows=2)
    ax[0, 0].imshow(simple_images[..., :3])
    ax[0, 1].imshow(simple_images[..., 3:])
    ax[1, 0].imshow(label[..., 0])
    ax[1, 1].imshow(label[..., 1])
    plt.show()


# def main():
#     config_network = deepcopy(CONFIG_FLOWNET)
#     config_training = deepcopy(CONFIG_TRAINING)

#     # On first run, populate the min, max scaling values for the flo dataset
#     # min, max = utils.get_training_min_max(config_training['img_path'])

#     flownet = FlowNet(config_network)

#     loss = EndPointError()

#     flownet.compile(optimizer=tf.keras.optimizers.Adam(),
#                     loss=[loss, loss, loss, loss, loss, loss],
#                     loss_weights=config_training['loss_weights'][::-1])

#     data_generator = DataGenerator(config_network['architecture'],
#                                    config_network['flo_normalization'],
#                                    config_training['img_path'],
#                                    config_training['batch_size'],
#                                    config_training['validation_batch_size'],
#                                    config_training['train_ratio'],
#                                    config_training['test_ratio'],
#                                    config_training['shuffle'],
#                                    config_training['augmentations'])

#     log_dir = f"logs/fit/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
#     tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

#     checkpoint_filepath = f"checkpoint/{datetime.now().strftime('%Y%m%d-%H%M%S')}.keras"
#     model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
#                                                                    save_weights_only=False,
#                                                                    monitor='val_loss',
#                                                                    mode='min',
#                                                                    save_best_only=True)

#     # if not config_training['pretrained_path'] is None:
#     #     # flownet.model = tf.keras.models.load_model(config_training['pretrained_path'], custom_objects={'EndPointError': EndPointError})
#     # flownet.fit(x=data_generator.next_train(),
#     #             epochs=1,
#     #             verbose=1,
#     #             steps_per_epoch=200 // config_training['batch_size'],
#     #             validation_data=data_generator.next_val(),
#     #             validation_steps=4,
#     #             validation_batch_size=config_training['validation_batch_size'],
#     #             callbacks=[tensorboard_callback, model_checkpoint_callback],
#     #             # use_multiprocessing=True
#     #             )
#     flownet.load_weights("/Users/oscar/Etudes/Ecole/PR/FlowNet_v1_TF2-master/src/flow.weights.h5")
#     # flownet.save_weights('flow.weights.h5')
#     # flownet.disable_training()
#     # IMPORTANT: switch to inference mode (single output)
#     flownet.disable_training()

#     # Visualize one FlyingChairs sample (change ID if needed)
#     visualize_prediction(
#         flownet,
#         image_name="00001",
#         config=config_network
#     )
#     # #
#     # Temporary debugging and visualization
#     # #
#     # img, flo = load_images(image_name="20000")
#     # norm_img = utils.normalize_images(img)
#     # predicted_flo = flownet.predict(norm_img)
#     # predicted_flo = utils.denormalize_flo(predicted_flo, config_network['flo_normalization'])
#     # predicted_flo = tf.image.resize(predicted_flo, (384, 512)).numpy()

#     # import matplotlib.pyplot as plt
#     # scale_min = np.min([np.min(flo), np.min(predicted_flo)])
#     # scale_max = np.max([np.max(flo), np.max(predicted_flo)])
#     # fig, ax = plt.subplots(ncols=2, nrows=3)
#     # ax[0, 0].imshow(img[0, ..., :3])
#     # ax[0, 1].imshow(img[0, ..., 3:])
#     # ax[0, 0].set_ylabel('Input images')
#     # ax[1, 0].imshow(flo[0, ..., 0], vmin=scale_min, vmax=scale_max)
#     # ax[1, 1].imshow(flo[0, ..., 1], vmin=scale_min, vmax=scale_max)
#     # ax[1, 0].set_ylabel('Ground truth flows')
#     # ax[2, 0].imshow(predicted_flo[0, ..., 0], vmin=scale_min, vmax=scale_max)
#     # ax[2, 1].imshow(predicted_flo[0, ..., 1], vmin=scale_min, vmax=scale_max)
#     # ax[2, 0].set_ylabel('Predicted flows')
#     # plt.show()

#     # # print('stall')
def visualize_flownet_prediction(flownet, image_id, config, data_path):
    """
    Visualize FlowNet prediction with input images, predicted flow, and ground truth.
    
    Args:
        flownet: Trained FlowNet model
        image_id: Image identifier (e.g., "00001")
        config: Configuration dictionary
        data_path: Path to the dataset directory
    """
    # Construct file paths
    img1_path = Path(data_path) / f"{image_id}_img1.ppm"
    img2_path = Path(data_path) / f"{image_id}_img2.ppm"
    flow_path = Path(data_path) / f"{image_id}_flow.flo"
    
    # Load images and ground truth flow
    img1 = uio.read(str(img1_path))
    img2 = uio.read(str(img2_path))
    flow_gt = uio.read(str(flow_path))
    
    # Normalize images for prediction
    img1_norm = utils.normalize_images([img1])[0]
    img2_norm = utils.normalize_images([img2])[0]
    
    # Concatenate images for FlowNet input
    network_input = np.concatenate([img1_norm, img2_norm], axis=-1)
    network_input = np.expand_dims(network_input, axis=0)  # Add batch dimension
    
    # Get prediction
    flow_pred = flownet.predict(network_input)
    
    # Handle different output shapes
    if isinstance(flow_pred, list):
        flow_pred = flow_pred[-1]  # Get the final prediction if multiple outputs
    
    # Remove batch dimension if present
    if len(flow_pred.shape) == 4:
        flow_pred = flow_pred[0]
    
    # Debug: print shapes
    print(f"Flow prediction shape: {flow_pred.shape}")
    print(f"Ground truth flow shape: {flow_gt.shape}")
    
    # Denormalize predicted flow
    # Assuming utils has denormalize_flo function, otherwise use manual method
    try:
        flow_pred_denorm = utils.denormalize_flo(flow_pred, config['flo_normalization'])
    except (AttributeError, NameError):
        # Manual denormalization
        flo_min, flo_max = config['flo_normalization']
        flow_pred_denorm = flow_pred * (flo_max - flo_min) + flo_min
    
    # Ensure flow_pred_denorm has correct shape (H, W, 2)
    if len(flow_pred_denorm.shape) == 2:
        print("Warning: Flow prediction is 2D, reshaping to (H, W, 2)")
        h, w = flow_pred_denorm.shape
        flow_pred_denorm = flow_pred_denorm.reshape(h, w, 1)
        flow_pred_denorm = np.concatenate([flow_pred_denorm, np.zeros_like(flow_pred_denorm)], axis=-1)
    
    # Resize predicted flow to match ground truth size if needed
    if flow_pred_denorm.shape[:2] != flow_gt.shape[:2]:
        print(f"Resizing flow from {flow_pred_denorm.shape[:2]} to {flow_gt.shape[:2]}")
        import cv2
        flow_pred_denorm = cv2.resize(flow_pred_denorm, 
                                      (flow_gt.shape[1], flow_gt.shape[0]), 
                                      interpolation=cv2.INTER_LINEAR)
    
    # Create visualization with white background
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), facecolor='white')
    fig.patch.set_facecolor('white')
    
    # Row 1: Input images
    axes[0, 0].imshow(img1.astype(np.uint8))
    axes[0, 0].set_title('Image 1', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img2.astype(np.uint8))
    axes[0, 1].set_title('Image 2', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Row 2: Ground truth flow and predicted flow
    # Ground truth flow visualization
    flow_gt_vis = flow_to_color(flow_gt)
    axes[1, 0].imshow(flow_gt_vis)
    axes[1, 0].set_title('Flot « GroundTruth »', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Predicted flow visualization
    flow_pred_vis = flow_to_color(flow_pred_denorm)
    axes[1, 1].imshow(flow_pred_vis)
    
    # Calculate EPE for title
    flow_error = np.sqrt(np.sum((flow_gt - flow_pred_denorm)**2, axis=-1))
    axes[1, 1].set_title(f'Flot Prédit', fontsize=14, fontweight='bold') #(EPE: {flow_error.mean():.2f})
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'flownet_visualization_{image_id}.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # Print statistics
    print(f"\n=== FlowNet Visualization Statistics ===")
    print(f"Image ID: {image_id}")
    print(f"Image shape: {img1.shape}")
    print(f"Flow shape: {flow_gt.shape}")
    print(f"Ground truth flow range: [{flow_gt.min():.2f}, {flow_gt.max():.2f}]")
    print(f"Predicted flow range: [{flow_pred_denorm.min():.2f}, {flow_pred_denorm.max():.2f}]")
    print(f"Average EPE (End-Point Error): {flow_error.mean():.3f}")
    print(f"Max EPE: {flow_error.max():.3f}")


def flow_to_color(flow, max_flow=None):
    """
    Convert optical flow to RGB color representation using Middlebury color scheme.
    
    Args:
        flow: Flow field (H, W, 2) with (u, v) components
        max_flow: Maximum flow magnitude for normalization. If None, uses max from flow.
    
    Returns:
        RGB image (H, W, 3) representing the flow
    """
    # Handle edge cases
    if flow is None:
        raise ValueError("Flow is None")
    
    # Ensure flow has correct shape
    if len(flow.shape) == 2:
        # If 2D, assume it's a single channel and create dummy second channel
        flow = np.stack([flow, np.zeros_like(flow)], axis=-1)
    
    if len(flow.shape) != 3 or flow.shape[2] != 2:
        raise ValueError(f"Flow must have shape (H, W, 2), got {flow.shape}")
    
    u = flow[:, :, 0]
    v = flow[:, :, 1]
    
    # Calculate flow magnitude and angle
    mag = np.sqrt(u**2 + v**2)
    angle = np.arctan2(-v, -u) / np.pi
    
    # Normalize magnitude
    if max_flow is None:
        max_flow = np.max(mag)
    
    if max_flow > 0:
        mag = mag / max_flow
    
    # Create Middlebury color wheel
    fk = (angle + 1) / 2 * (ncols - 1)  # -1~1 mapped to 0~ncols-1
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    
    ncolors = colorwheel.shape[0]
    
    img = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.float32)
    
    for i in range(3):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1
        
        # Increase saturation with magnitude
        idx = mag <= 1
        col[idx] = 1 - mag[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75  # Out of range
        
        img[:, :, i] = col
    
    return img


def make_colorwheel():
    """
    Generate color wheel according to Middlebury color code
    for optical flow visualization.
    
    Returns:
        colorwheel: Color wheel with shape (ncols, 3)
    """
    # Relative lengths of color transitions
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6
    
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3), dtype=np.uint8)
    
    col = 0
    
    # RY
    colorwheel[col:col+RY, 0] = 255
    colorwheel[col:col+RY, 1] = np.floor(255 * np.arange(RY) / RY)
    col += RY
    
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255 * np.arange(YG) / YG)
    colorwheel[col:col+YG, 1] = 255
    col += YG
    
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255 * np.arange(GC) / GC)
    col += GC
    
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col:col+CB, 2] = 255
    col += CB
    
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255 * np.arange(BM) / BM)
    col += BM
    
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    colorwheel[col:col+MR, 0] = 255
    
    return colorwheel


# Generate the Middlebury color wheel
colorwheel = make_colorwheel()
ncols = colorwheel.shape[0]


def visualize_multiple_samples(flownet, image_ids, config, data_path):
    """
    Visualize multiple samples in a grid.
    
    Args:
        flownet: Trained FlowNet model
        image_ids: List of image identifiers
        config: Configuration dictionary
        data_path: Path to the dataset directory
    """
    n_samples = len(image_ids)
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4*n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx, image_id in enumerate(image_ids):
        # Load data
        img1_path = Path(data_path) / f"{image_id}_img1.ppm"
        img2_path = Path(data_path) / f"{image_id}_img2.ppm"
        flow_path = Path(data_path) / f"{image_id}_flow.flo"
        
        img1 = uio.read(str(img1_path))
        img2 = uio.read(str(img2_path))
        flow_gt = uio.read(str(flow_path))
        
        # Get prediction
        flow_pred = flownet.predict(network_input, verbose=0)
        
        # Handle different output shapes
        if isinstance(flow_pred, list):
            flow_pred = flow_pred[-1]
        
        # Remove batch dimension if present
        if len(flow_pred.shape) == 4:
            flow_pred = flow_pred[0]
        
        # Denormalize
        try:
            flow_pred_denorm = utils.denormalize_flo(flow_pred, config['flo_normalization'])
        except (AttributeError, NameError):
            flo_min, flo_max = config['flo_normalization']
            flow_pred_denorm = flow_pred * (flo_max - flo_min) + flo_min
        
        # Ensure correct shape and size
        if len(flow_pred_denorm.shape) == 2:
            h, w = flow_pred_denorm.shape
            flow_pred_denorm = flow_pred_denorm.reshape(h, w, 1)
            flow_pred_denorm = np.concatenate([flow_pred_denorm, np.zeros_like(flow_pred_denorm)], axis=-1)
        
        if flow_pred_denorm.shape[:2] != flow_gt.shape[:2]:
            import cv2
            flow_pred_denorm = cv2.resize(flow_pred_denorm, 
                                          (flow_gt.shape[1], flow_gt.shape[0]), 
                                          interpolation=cv2.INTER_LINEAR)
        
        # Visualize
        axes[idx, 0].imshow(img1.astype(np.uint8))
        axes[idx, 0].set_title(f'{image_id} - Image 1')
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(img2.astype(np.uint8))
        axes[idx, 1].set_title(f'{image_id} - Image 2')
        axes[idx, 1].axis('off')
        
        axes[idx, 2].imshow(flow_to_color(flow_gt))
        axes[idx, 2].set_title('Ground Truth')
        axes[idx, 2].axis('off')
        
        axes[idx, 3].imshow(flow_to_color(flow_pred_denorm))
        epe = np.sqrt(np.sum((flow_gt - flow_pred_denorm)**2, axis=-1)).mean()
        axes[idx, 3].set_title(f'Prediction (EPE: {epe:.2f})')
        axes[idx, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('flownet_multiple_samples.png', dpi=150, bbox_inches='tight')
    plt.show()
def main():
    config_network = deepcopy(CONFIG_FLOWNET)
    config_training = deepcopy(CONFIG_TRAINING)
    
    flownet = FlowNet(config_network)
    flownet.load_weights("/Users/oscar/Etudes/Ecole/PR/FlowNet_v1_TF2-master/src/flow.weights.h5")
    
    # IMPORTANT: switch to inference mode (single output)
    flownet.disable_training()
    
    # Visualize a single sample with detailed view
    visualize_flownet_prediction(
        flownet=flownet,
        image_id="00001",  # Change this to any image ID
        config=config_network,
        data_path=config_training['img_path']
    )
    
    # Optional: Visualize multiple samples in a grid
    # visualize_multiple_samples(
    #     flownet=flownet,
    #     image_ids=["00001", "00002", "00003"],
    #     config=config_network,
    #     data_path=config_training['img_path']
    # )

if __name__ == "__main__":
    main()
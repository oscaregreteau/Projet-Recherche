"""
"""


from typing import Tuple, List, Union
from pathlib import Path

import numpy as np
import tensorflow as tf
import cv2
import utils_io as uio
import matplotlib.pyplot as plt

def get_training_min_max(root_path: Path) -> List[np.ndarray]:
    flo_list = []
    for idx, flo in enumerate(root_path.glob('*.flo')):
        # idx isn't required but makes troubleshooting easier
        v = uio.read(str(flo))
        flo_list.append([tf.math.reduce_min(v), tf.math.reduce_max(v)])
    flo_list_stacked = tf.stack(flo_list, axis=0)
    return [tf.math.reduce_min(flo_list_stacked), tf.math.reduce_max(flo_list_stacked)]


def normalize_images(img: np.array) -> np.ndarray:
    # this, tf.image.convert_image_dtype(img, tf.float16), is producing strange results, ie all zero values...
    return tf.cast(img, tf.float16) / 255  # Casting to float16 may reduce precision but hopefully it increases speed


def normalize_flo(flo: np.array, scale_factors: Tuple[float, float]) -> np.ndarray:
    # range -> [-1, 1]
    if isinstance(flo, list):
        flo = tf.stack(flo, axis=0)
    return ((flo - scale_factors[0]) / (scale_factors[1] - scale_factors[0]) - 0.5) * 2


def denormalize_flo(flo: np.ndarray, scale_factors: Tuple[float, float]) -> np.ndarray:
    return (scale_factors[1] - scale_factors[0]) * (0.5 + flo/2) + scale_factors[0]


def get_train_val_test(image_names: List[Path],
                       train_ratio: Union[float, int],
                       test_ratio: Union[float, int],
                       shuffle: bool = True) -> Tuple[List[Path], List[Path], List[Path]]:
    """ Get the train, val, and test sets from a list of all image paths.
        The test set is the last block and shouldn't be handled until after hyperparameter tuning.
        This function is sloppy and can easily be broken.  Reasonable values, such as train_ratio=0.7 and test_ratio=0.1
        will return a train_ratio of 0.7, a validation_ratio of 0.2, and a test_ratio of 0.1 and work fine.
    """
    if (not 0 < train_ratio < 1) or (not 0 < test_ratio < 1) or (train_ratio + test_ratio >= 1):
        raise Exception(f"Why have you done this. Train ratio: {train_ratio}, val ratio: {1-train_ratio-test_ratio}, Test ratio: {test_ratio}.")

    n_images = len(image_names)
    test = image_names[int(-test_ratio*n_images):]  # Don't use the last set of images until done hyperparameter tuning

    image_names = image_names[:int(-test_ratio*n_images)]

    n_train = int(train_ratio * n_images)
    if shuffle:
        np.random.shuffle(image_names)
    train = image_names[:n_train]
    val = image_names[n_train:]

    return train, val, test

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
    img1_norm = normalize_images([img1])[0]
    img2_norm = normalize_images([img2])[0]
    
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
    
    # Denormalize predicted flow if needed
    if hasattr('denormalize_flo'):
        flow_pred_denorm = denormalize_flo(flow_pred, config['flo_normalization'])
    else:
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
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Input images and their difference
    axes[0, 0].imshow(img1.astype(np.uint8))
    axes[0, 0].set_title('Image 1 (t)', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img2.astype(np.uint8))
    axes[0, 1].set_title('Image 2 (t+1)', fontsize=12)
    axes[0, 1].axis('off')
    
    # Show difference between images
    img_diff = np.abs(img1.astype(float) - img2.astype(float))
    axes[0, 2].imshow(img_diff.astype(np.uint8))
    axes[0, 2].set_title('Absolute Difference', fontsize=12)
    axes[0, 2].axis('off')
    
    # Row 2: Ground truth flow and predicted flow
    # Ground truth flow visualization
    flow_gt_vis = flow_to_color(flow_gt)
    axes[1, 0].imshow(flow_gt_vis)
    axes[1, 0].set_title('Ground Truth Flow', fontsize=12)
    axes[1, 0].axis('off')
    
    # Predicted flow visualization
    flow_pred_vis = flow_to_color(flow_pred_denorm)
    axes[1, 1].imshow(flow_pred_vis)
    axes[1, 1].set_title('Predicted Flow', fontsize=12)
    axes[1, 1].axis('off')
    
    # Flow error visualization
    flow_error = np.sqrt(np.sum((flow_gt - flow_pred_denorm)**2, axis=-1))
    im = axes[1, 2].imshow(flow_error, cmap='hot')
    axes[1, 2].set_title(f'Flow Error (EPE: {flow_error.mean():.2f})', fontsize=12)
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(f'flownet_visualization_{image_id}.png', dpi=150, bbox_inches='tight')
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
    Convert optical flow to RGB color representation.
    Uses the color wheel representation commonly used in optical flow visualization.
    
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
    angle = np.arctan2(v, u)
    
    # Normalize
    if max_flow is None:
        max_flow = mag.max()
    
    if max_flow > 0:
        mag = mag / max_flow
    
    # Convert to HSV then to RGB
    # Hue represents direction, Saturation is always 1, Value represents magnitude
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.float32)
    hsv[:, :, 0] = (angle + np.pi) / (2 * np.pi)  # Hue (normalized angle)
    hsv[:, :, 1] = 1.0  # Saturation
    hsv[:, :, 2] = np.clip(mag, 0, 1)  # Value (normalized magnitude)
    
    # Convert HSV to RGB
    from matplotlib.colors import hsv_to_rgb
    rgb = hsv_to_rgb(hsv)
    
    return rgb


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
        if hasattr('denormalize_flo'):
            flow_pred_denorm = denormalize_flo(flow_pred, config['flo_normalization'])
        else:
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
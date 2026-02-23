import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def warp_image(img, flow):
    """
    Fully vectorized backward warp using only TensorFlow.
    img: [B, H, W, C]
    flow: [B, H, W, 2]
    returns: [B, H, W, C] warped image
    """
    batch_size, height, width, channels = tf.unstack(tf.shape(img))
    
    grid_y, grid_x = tf.meshgrid(tf.range(height), tf.range(width), indexing='ij')
    grid_x = tf.cast(grid_x, tf.float32)
    grid_y = tf.cast(grid_y, tf.float32)
    
    sampling_x = grid_x[tf.newaxis, ...] + flow[..., 0]
    sampling_y = grid_y[tf.newaxis, ...] + flow[..., 1]
    
    x0 = tf.cast(tf.floor(sampling_x), tf.int32)
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(sampling_y), tf.int32)
    y1 = y0 + 1
    
    x0 = tf.clip_by_value(x0, 0, width-1)
    x1 = tf.clip_by_value(x1, 0, width-1)
    y0 = tf.clip_by_value(y0, 0, height-1)
    y1 = tf.clip_by_value(y1, 0, height-1)
    
    def get_pixel(img, x, y):
        batch_idx = tf.range(batch_size)[:, tf.newaxis, tf.newaxis]
        batch_idx = tf.tile(batch_idx, [1, height, width])
        indices = tf.stack([batch_idx, y, x], axis=-1)  # [B,H,W,3]
        return tf.gather_nd(img, indices)
    
    Ia = get_pixel(img, x0, y0)
    Ib = get_pixel(img, x0, y1)
    Ic = get_pixel(img, x1, y0)
    Id = get_pixel(img, x1, y1)
    
    wa = tf.cast(x1, tf.float32) - sampling_x
    wa *= tf.cast(y1, tf.float32) - sampling_y
    wb = tf.cast(x1, tf.float32) - sampling_x
    wb *= sampling_y - tf.cast(y0, tf.float32)
    wc = sampling_x - tf.cast(x0, tf.float32)
    wc *= tf.cast(y1, tf.float32) - sampling_y
    wd = sampling_x - tf.cast(x0, tf.float32)
    wd *= sampling_y - tf.cast(y0, tf.float32)
    

    warped = wa[..., tf.newaxis]*Ia + wb[..., tf.newaxis]*Ib + wc[..., tf.newaxis]*Ic + wd[..., tf.newaxis]*Id
    return warped

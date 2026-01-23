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
    
    # 1. Create a regular grid
    grid_y, grid_x = tf.meshgrid(tf.range(height), tf.range(width), indexing='ij')
    grid_x = tf.cast(grid_x, tf.float32)
    grid_y = tf.cast(grid_y, tf.float32)
    
    # 2. Apply flow
    sampling_x = grid_x[tf.newaxis, ...] + flow[..., 0]
    sampling_y = grid_y[tf.newaxis, ...] + flow[..., 1]
    
    # 3. Get 4 neighboring indices for bilinear interpolation
    x0 = tf.cast(tf.floor(sampling_x), tf.int32)
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(sampling_y), tf.int32)
    y1 = y0 + 1
    
    # 4. Clip coordinates to image boundaries
    x0 = tf.clip_by_value(x0, 0, width-1)
    x1 = tf.clip_by_value(x1, 0, width-1)
    y0 = tf.clip_by_value(y0, 0, height-1)
    y1 = tf.clip_by_value(y1, 0, height-1)
    
    # 5. Gather pixel values
    def get_pixel(img, x, y):
        batch_idx = tf.range(batch_size)[:, tf.newaxis, tf.newaxis]
        batch_idx = tf.tile(batch_idx, [1, height, width])
        indices = tf.stack([batch_idx, y, x], axis=-1)  # [B,H,W,3]
        return tf.gather_nd(img, indices)
    
    Ia = get_pixel(img, x0, y0)
    Ib = get_pixel(img, x0, y1)
    Ic = get_pixel(img, x1, y0)
    Id = get_pixel(img, x1, y1)
    
    # 6. Compute bilinear weights
    wa = tf.cast(x1, tf.float32) - sampling_x
    wa *= tf.cast(y1, tf.float32) - sampling_y
    wb = tf.cast(x1, tf.float32) - sampling_x
    wb *= sampling_y - tf.cast(y0, tf.float32)
    wc = sampling_x - tf.cast(x0, tf.float32)
    wc *= tf.cast(y1, tf.float32) - sampling_y
    wd = sampling_x - tf.cast(x0, tf.float32)
    wd *= sampling_y - tf.cast(y0, tf.float32)
    
    # 7. Combine
    warped = wa[..., tf.newaxis]*Ia + wb[..., tf.newaxis]*Ib + wc[..., tf.newaxis]*Ic + wd[..., tf.newaxis]*Id
    return warped

# def read_flo(filename):
#     """
#     Reads a .flo file (Middlebury format)
#     Returns a numpy array of shape (H, W, 2)
#     """
#     with open(filename, 'rb') as f:
#         magic = np.fromfile(f, np.float32, count=1)
#         if magic != 202021.25:
#             raise ValueError('Magic number incorrect. Invalid .flo file')
        
#         w = np.fromfile(f, np.int32, count=1)[0]
#         h = np.fromfile(f, np.int32, count=1)[0]
#         data = np.fromfile(f, np.float32, count=2*w*h)
        
#         # ✅ Use reshape, NOT resize
#         flow = np.reshape(data, (h, w, 2))
        
#         # Optional: flip vertical flow if residual is high
#         # flow[..., 1] = -flow[..., 1]
        
#         return flow

# # --- Paths for pair 00001 ---
# img1_path = "/Users/oscar/Downloads/FlyingChairs_release/data/00001_img1.ppm"
# img2_path = "/Users/oscar/Downloads/FlyingChairs_release/data/00001_img2.ppm"
# flo_path  = "/Users/oscar/Downloads/FlyingChairs_release/data/00001_flow.flo"

# # Load images
# img1 = np.array(Image.open(img1_path)).astype(np.float32) / 255.0
# img2 = np.array(Image.open(img2_path)).astype(np.float32) / 255.0
# flow_np = read_flo(flo_path)

# # Convert to tensors
# img2_tf = tf.expand_dims(tf.convert_to_tensor(img2), axis=0) # [1,H,W,3]
# flow_tf = tf.expand_dims(tf.convert_to_tensor(flow_np), axis=0) # [1,H,W,2]

# # --- Backward warp ---
# warped_img1 = backward_warp(img2_tf, flow_tf)

# # Compute residual
# residual = np.abs(img1 - warped_img1[0].numpy())

# # Visualize
# fig, axes = plt.subplots(1, 3, figsize=(18,5))
# axes[0].imshow(img1); axes[0].set_title("Target: Image 1")
# axes[1].imshow(warped_img1[0].numpy()); axes[1].set_title("Warped: Image 2 -> Image 1")
# axes[2].imshow(np.sum(residual, axis=-1), cmap='hot'); axes[2].set_title("Error (should be black)")
# plt.show()

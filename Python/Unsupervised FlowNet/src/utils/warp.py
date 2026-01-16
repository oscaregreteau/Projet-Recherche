import tensorflow as tf

def warp_image(image, flow):
    """
    Warp `image` using `flow` (optical flow) with bilinear interpolation.
    Works with both eager tensors and symbolic Keras tensors.
    """
    B, H, W, C = tf.unstack(tf.shape(image))  # <-- use tf.shape + unstack

    # Create a grid
    grid_x, grid_y = tf.meshgrid(
        tf.linspace(0.0, tf.cast(W - 1, tf.float32), W),
        tf.linspace(0.0, tf.cast(H - 1, tf.float32), H)
    )

    grid = tf.stack([grid_x, grid_y], axis=-1)
    grid = tf.expand_dims(grid, 0)
    grid = tf.tile(grid, [B, 1, 1, 1])

    coords = grid + flow
    x, y = coords[..., 0], coords[..., 1]

    x0 = tf.clip_by_value(tf.floor(x), 0, tf.cast(W - 1, tf.float32))
    x1 = tf.clip_by_value(x0 + 1, 0, tf.cast(W - 1, tf.float32))
    y0 = tf.clip_by_value(tf.floor(y), 0, tf.cast(H - 1, tf.float32))
    y1 = tf.clip_by_value(y0 + 1, 0, tf.cast(H - 1, tf.float32))

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    x0 = tf.cast(x0, tf.int32)
    x1 = tf.cast(x1, tf.int32)
    y0 = tf.cast(y0, tf.int32)
    y1 = tf.cast(y1, tf.int32)

    batch_idx = tf.reshape(tf.range(B), (B, 1, 1))
    batch_idx = tf.tile(batch_idx, [1, H, W])

    def gather(ix, iy):
        idx = tf.stack([batch_idx, iy, ix], axis=-1)
        return tf.gather_nd(image, idx)

    Ia = gather(x0, y0)
    Ib = gather(x0, y1)
    Ic = gather(x1, y0)
    Id = gather(x1, y1)

    warped = wa[..., None] * Ia + wb[..., None] * Ib + wc[..., None] * Ic + wd[..., None] * Id
    return warped

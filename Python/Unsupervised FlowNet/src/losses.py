import tensorflow as tf

def charbonnier(x, alpha: float = 0.25, epsilon: float = 1e-9):
    return tf.pow(tf.square(x) + epsilon ** 2, alpha)

def smoothness_loss(flow):
    b = tf.cast(tf.shape(flow)[0], tf.float32)
    h = tf.shape(flow)[1]
    w = tf.shape(flow)[2]

    v_shift = tf.concat(
        [flow[:, 1:, :, :],
         tf.zeros([tf.shape(flow)[0], 1, w, 2], dtype=flow.dtype)],
        axis=1
    )
    h_shift = tf.concat(
        [flow[:, :, 1:, :],
         tf.zeros([tf.shape(flow)[0], h, 1, 2], dtype=flow.dtype)],
        axis=2
    )

    s_loss = charbonnier(flow - v_shift) + charbonnier(flow - h_shift)
    s_loss = tf.reduce_sum(s_loss, axis=-1) / 2.0

    return tf.reduce_sum(s_loss) / b

def photometric_loss(warped, frame1):
    h_w = tf.shape(warped)[1]
    w_w = tf.shape(warped)[2]

    frame1_r = tf.image.resize(frame1, [h_w, w_w], method="bilinear")

    p_loss = charbonnier(warped - frame1_r)
    p_loss = tf.reduce_sum(p_loss, axis=-1) / 3.0  

    b = tf.cast(tf.shape(frame1)[0], tf.float32)
    return tf.reduce_sum(p_loss) / b

_DEFAULT_WEIGHTS = (0.32, 0.08, 0.02, 0.01, 0.005)


def unsup_loss(pred_flows, warped_imgs, frame1,
               weights=_DEFAULT_WEIGHTS):
    n = len(pred_flows)
    if n < len(weights):
        w = [0.005] * n
    else:
        w = list(weights[:n])

    photo_total  = tf.constant(0.0, dtype=tf.float32)
    smooth_total = tf.constant(0.0, dtype=tf.float32)

    for i in range(n):
        photo_total  += w[i] * photometric_loss(warped_imgs[i], frame1)
        smooth_total += w[i] * smoothness_loss(pred_flows[i])

    total = photo_total + smooth_total
    return total, photo_total, smooth_total


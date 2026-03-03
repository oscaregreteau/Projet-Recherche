import tensorflow as tf
from warp import warp_image

def local_response_norm(img, kernel_size=9):
    avg = tf.nn.avg_pool2d(img, ksize=kernel_size, strides=1, padding='SAME')
    return img - avg

def charbonnier(x, alpha):
    eps = 1e-5
    return tf.pow(x**2 + eps**2, alpha)

def photo_loss(image1, image2_warped):
    image1 = local_response_norm(image1)
    image2_warped = local_response_norm(image2_warped)
    return tf.reduce_mean(charbonnier(image1 - image2_warped, alpha=0.25))

def smooth_loss(flow):
    dx = charbonnier(flow[:, :, 1:, :] - flow[:, :, :-1, :], alpha=0.37)
    dy = charbonnier(flow[:, 1:, :, :] - flow[:, :-1, :, :], alpha=0.37)
    return tf.reduce_mean(dx) + tf.reduce_mean(dy)

def multiscale_loss(flows, frame1, frame2, smooth_weight=1.0, weights=None):
    if weights is None:
        weights = [0.125, 0.25, 0.5, 1.0]

    total = 0.0
    scale_losses = []

    for flow, w in zip(flows, weights):
        h = tf.shape(flow)[1]
        w_dim = tf.shape(flow)[2]
        frame1_s = tf.image.resize(frame1, [h, w_dim])
        frame2_s = tf.image.resize(frame2, [h, w_dim])
        warped = warp_image(frame2_s, flow)
        photo = photo_loss(frame1_s, warped)
        smooth = smooth_loss(flow)
        scale_loss = w * (photo + smooth_weight * smooth)
        scale_losses.append(scale_loss)
        total += scale_loss

    return total, scale_losses
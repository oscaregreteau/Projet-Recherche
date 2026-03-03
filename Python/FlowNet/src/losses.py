import tensorflow as tf

def epe_loss(flow_pred, flow_gt, valid=None):
    epe = tf.norm(flow_pred - flow_gt, axis=-1)
    if valid is not None:
        h = tf.shape(flow_pred)[1]
        w = tf.shape(flow_pred)[2]
        valid_resized = tf.image.resize(valid, (h, w), method='nearest')
        valid_mask = tf.squeeze(valid_resized, axis=-1) > 0.5
        epe = tf.boolean_mask(epe, valid_mask)
    return tf.reduce_mean(epe)

def multiscale_epe_loss(flow_preds, flow_gt, valid=None, weights=None):
    if weights is None:
        weights = [0.005, 0.01, 0.02, 0.08, 0.32]

    if len(weights) != len(flow_preds):
        raise ValueError(f"Number of weights ({len(weights)}) must match number of predictions ({len(flow_preds)})")

    total_loss = 0.0
    scale_losses = []

    for pred, weight in zip(flow_preds, weights):
        h = tf.shape(pred)[1]
        w = tf.shape(pred)[2]

        gt_scaled = tf.image.resize(flow_gt, (h, w), method='bilinear')

        scale_h = tf.cast(h, tf.float32) / tf.cast(tf.shape(flow_gt)[1], tf.float32)
        scale_w = tf.cast(w, tf.float32) / tf.cast(tf.shape(flow_gt)[2], tf.float32)
        gt_scaled = gt_scaled * tf.stack([scale_w, scale_h])[tf.newaxis, tf.newaxis, tf.newaxis, :]

        loss = weight * epe_loss(pred, gt_scaled, valid)
        scale_losses.append(loss)
        total_loss += loss

    return total_loss, scale_losses
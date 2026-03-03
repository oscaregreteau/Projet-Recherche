import tensorflow as tf

def random_color_jitter(frame1, frame2):
    brightness = tf.random.uniform((), -0.2, 0.2)
    frame1 = tf.clip_by_value(frame1 + brightness, 0.0, 1.0)
    frame2 = tf.clip_by_value(frame2 + brightness, 0.0, 1.0)

    contrast = tf.random.uniform((), 0.8, 1.2)
    frame1 = tf.clip_by_value((frame1 - 0.5) * contrast + 0.5, 0.0, 1.0)
    frame2 = tf.clip_by_value((frame2 - 0.5) * contrast + 0.5, 0.0, 1.0)

    return frame1, frame2

def augment(frame1, frame2, flow, valid, crop_size=(320, 448)):
    h, w = tf.shape(frame1)[0], tf.shape(frame1)[1]
    crop_h, crop_w = crop_size
    y = tf.random.uniform((), 0, h - crop_h, dtype=tf.int32)
    x = tf.random.uniform((), 0, w - crop_w, dtype=tf.int32)
    frame1 = frame1[y:y+crop_h, x:x+crop_w]
    frame2 = frame2[y:y+crop_h, x:x+crop_w]
    flow   = flow[y:y+crop_h, x:x+crop_w]
    valid  = valid[y:y+crop_h, x:x+crop_w]

    if tf.random.uniform(()) > 0.5:
        frame1 = tf.image.flip_left_right(frame1)
        frame2 = tf.image.flip_left_right(frame2)
        flow   = tf.image.flip_left_right(flow)
        flow   = flow * tf.constant([-1.0, 1.0]) 

    frame1, frame2 = random_color_jitter(frame1, frame2)
    return frame1, frame2, flow, valid
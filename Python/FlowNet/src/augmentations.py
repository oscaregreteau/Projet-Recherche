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
    frame1, frame2 = random_color_jitter(frame1, frame2)
    return frame1, frame2, flow, valid
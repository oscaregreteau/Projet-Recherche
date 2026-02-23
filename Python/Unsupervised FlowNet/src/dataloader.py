import tensorflow as tf
import os

def load_image_pair(frame1_path, frame2_path, image_size=(384, 512)):
    frame1 = tf.io.read_file(frame1_path)
    frame1 = tf.image.decode_png(frame1, channels=3)
    frame1 = tf.image.resize(frame1, image_size)
    frame1 = tf.cast(frame1, tf.float32) / 255.0

    frame2 = tf.io.read_file(frame2_path)
    frame2 = tf.image.decode_png(frame2, channels=3)
    frame2 = tf.image.resize(frame2, image_size)
    frame2 = tf.cast(frame2, tf.float32) / 255.0

    return frame1, frame2

def get_dataset(data_path, batch_size, image_size=(384, 512), shuffle=True):
    frame1_dir = os.path.join(data_path, 'image_2')
    frame2_dir = os.path.join(data_path, 'image_3')

    frame1_paths = sorted([
        os.path.join(frame1_dir, f) for f in os.listdir(frame1_dir)
        if f.endswith('.png') or f.endswith('.jpg')
    ])
    frame2_paths = sorted([
        os.path.join(frame2_dir, f) for f in os.listdir(frame2_dir)
        if f.endswith('.png') or f.endswith('.jpg')
    ])

    assert len(frame1_paths) == len(frame2_paths), "Mismatch between frame1 and frame2 counts"

    dataset = tf.data.Dataset.from_tensor_slices((frame1_paths, frame2_paths))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(frame1_paths))

    dataset = dataset.map(
        lambda f1, f2: load_image_pair(f1, f2, image_size),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
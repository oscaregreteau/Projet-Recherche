import tensorflow as tf
import numpy as np
from PIL import Image
import os

def read_ppm(path):
    path = path.numpy().decode('utf-8')
    img = Image.open(path).convert('RGB')
    return np.array(img, dtype=np.float32) / 255.0

def load_image_pair(frame1_path, frame2_path, image_size=(384, 512)):
    frame1 = tf.py_function(read_ppm, [frame1_path], tf.float32)
    frame1.set_shape([None, None, 3])
    frame1 = tf.image.resize(frame1, image_size)
    frame1.set_shape([image_size[0], image_size[1], 3])

    frame2 = tf.py_function(read_ppm, [frame2_path], tf.float32)
    frame2.set_shape([None, None, 3])
    frame2 = tf.image.resize(frame2, image_size)
    frame2.set_shape([image_size[0], image_size[1], 3])

    return frame1, frame2

def get_dataset(data_path, batch_size, image_size=(384, 512), shuffle=True, split='train', val_split=0.2):
    frame1_paths = sorted([
        os.path.join(data_path, f) for f in os.listdir(data_path)
        if f.endswith('_img1.ppm')
    ])
    frame2_paths = sorted([
        os.path.join(data_path, f) for f in os.listdir(data_path)
        if f.endswith('_img2.ppm')
    ])

    if len(frame1_paths) != len(frame2_paths):
        raise ValueError(
            f"File count mismatch: {len(frame1_paths)} frame1, "
            f"{len(frame2_paths)} frame2 files"
        )

    split_idx = int(len(frame1_paths) * (1 - val_split))

    if split == 'train':
        frame1_paths = frame1_paths[:split_idx]
        frame2_paths = frame2_paths[:split_idx]
    elif split == 'val':
        frame1_paths = frame1_paths[split_idx:]
        frame2_paths = frame2_paths[split_idx:]
    else:
        raise ValueError(f"split must be 'train' or 'val', got '{split}'")

    dataset = tf.data.Dataset.from_tensor_slices((frame1_paths, frame2_paths))

    if shuffle and split == 'train':
        dataset = dataset.shuffle(buffer_size=len(frame1_paths))

    dataset = dataset.map(
        lambda f1, f2: load_image_pair(f1, f2, image_size),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

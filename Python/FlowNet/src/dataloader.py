import tensorflow as tf
import numpy as np
from PIL import Image
import os
from augmentations import augment

def read_ppm(path):
    path = path.numpy().decode('utf-8')
    img = Image.open(path).convert('RGB')
    return np.array(img, dtype=np.float32) / 255.0

def load_image_pair(frame1_path, frame2_path, flo_path, image_size=(384, 512), training=True):
    frame1 = tf.py_function(read_ppm, [frame1_path], tf.float32)
    frame1.set_shape([None, None, 3])
    frame1 = tf.image.resize(frame1, image_size)
    frame1.set_shape([image_size[0], image_size[1], 3])

    frame2 = tf.py_function(read_ppm, [frame2_path], tf.float32)
    frame2.set_shape([None, None, 3])
    frame2 = tf.image.resize(frame2, image_size)
    frame2.set_shape([image_size[0], image_size[1], 3])

    flow, valid = load_flo(flo_path, image_size)

    if training:
        frame1, frame2, flow, valid = augment(frame1, frame2, flow, valid)

    return frame1, frame2, flow, valid

def load_flo(flo_path, image_size=(384, 512)):
    raw = tf.io.read_file(flo_path)
    flow = tf.io.decode_raw(raw, tf.float32)[3:]
    flow = tf.reshape(flow, [384, 512, 2])
    flow = tf.image.resize(flow, image_size, method='bilinear')
    valid = tf.ones([image_size[0], image_size[1], 1], dtype=tf.float32)
    return flow, valid

#for kitti
# def load_flo(flo_path, image_size=(384, 512)):
#     raw = tf.io.read_file(flo_path)
#     img = tf.image.decode_png(raw, channels=3, dtype=tf.uint16)
#     img = tf.cast(img, tf.float32)

#     u = (img[..., 0] - 2**15) / 64.0
#     v = (img[..., 1] - 2**15) / 64.0
#     valid = img[..., 2]

#     flow = tf.stack([u, v], axis=-1)
#     flow = tf.image.resize(flow, image_size)
#     valid = tf.image.resize(valid[..., tf.newaxis], image_size)

#     return flow, valid

#for kitti
# def load_image_pair(frame1_path, frame2_path, flo_path, image_size=(384, 512)):
#     frame1 = tf.io.read_file(frame1_path)
#     frame1 = tf.image.decode_png(frame1, channels=3)
#     frame1 = tf.image.resize(frame1, image_size)
#     frame1 = tf.cast(frame1, tf.float32) / 255.0

#     frame2 = tf.io.read_file(frame2_path)
#     frame2 = tf.image.decode_png(frame2, channels=3)
#     frame2 = tf.image.resize(frame2, image_size)
#     frame2 = tf.cast(frame2, tf.float32) / 255.0
#     flow, valid = load_flo(flo_path, image_size)
#     return frame1, frame2, flow, valid

def get_dataset(data_path, batch_size, image_size=(384, 512), shuffle=True, split='train', val_split=0.2):
    frame1_dir = os.path.join(data_path)
    frame2_dir = os.path.join(data_path)
    flo_dir    = os.path.join(data_path)

    frame1_paths = sorted([
        os.path.join(frame1_dir, f) for f in os.listdir(frame1_dir)
        if f.endswith('_img1.ppm')
    ])
    frame2_paths = sorted([
        os.path.join(frame2_dir, f) for f in os.listdir(frame2_dir)
        if f.endswith('_img2.ppm')
    ])
    flo_paths = sorted([
        os.path.join(flo_dir, f) for f in os.listdir(flo_dir)
        if f.endswith('_flow.flo')
    ])
    
    #kitti loader
    # frame1_dir = os.path.join(data_path, 'image_2')
    # frame2_dir = os.path.join(data_path, 'image_3')
    # flo_dir    = os.path.join(data_path, 'flow_occ')
    # frame1_paths = sorted([
    #     os.path.join(frame1_dir, f) for f in os.listdir(frame1_dir)
    #     if f.endswith('_10.png')
    # ])
    # frame2_paths = sorted([
    #     os.path.join(frame2_dir, f) for f in os.listdir(frame2_dir)
    #     if f.endswith('_10.png')
    # ])
    # flo_paths = sorted([
    #     os.path.join(flo_dir, f) for f in os.listdir(flo_dir)
    #     if f.endswith('_10.png')
    # ])

    if len(frame1_paths) != len(frame2_paths) or len(frame2_paths) != len(flo_paths):
        raise ValueError(
            f"File count mismatch: {len(frame1_paths)} frame1, "
            f"{len(frame2_paths)} frame2, {len(flo_paths)} flow files"
        )

    split_idx = int(len(frame1_paths) * (1 - val_split))

    if split == 'train':
        frame1_paths = frame1_paths[:split_idx]
        frame2_paths = frame2_paths[:split_idx]
        flo_paths    = flo_paths[:split_idx]
    elif split == 'val':
        frame1_paths = frame1_paths[split_idx:]
        frame2_paths = frame2_paths[split_idx:]
        flo_paths    = flo_paths[split_idx:]
    else:
        raise ValueError(f"split must be 'train' or 'val', got '{split}'")

    dataset = tf.data.Dataset.from_tensor_slices((frame1_paths, frame2_paths, flo_paths))

    if shuffle and split == 'train':
        dataset = dataset.shuffle(buffer_size=len(frame1_paths))

    dataset = dataset.map(
        lambda f1, f2, flo: load_image_pair(f1, f2, flo, image_size),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
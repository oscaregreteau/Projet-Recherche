import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from pathlib import Path
import matplotlib.pyplot as plt
import struct

# ============================================================
# Middlebury .flo reader
# ============================================================

def read_flo(path):
    with open(path, 'rb') as f:
        magic = struct.unpack('f', f.read(4))[0]
        if magic != 202021.25:
            raise ValueError("Invalid .flo file")
        w = struct.unpack('i', f.read(4))[0]
        h = struct.unpack('i', f.read(4))[0]
        data = np.fromfile(f, np.float32, count=2 * w * h)
    return data.reshape(h, w, 2)


# ============================================================
# Middlebury color wheel visualization
# ============================================================

def make_colorwheel():
    RY, YG, GC, CB, BM, MR = 15, 6, 4, 11, 13, 6
    ncols = RY + YG + GC + CB + BM + MR
    wheel = np.zeros((ncols, 3))

    col = 0
    wheel[0:RY, 0] = 1
    wheel[0:RY, 1] = np.linspace(0, 1, RY)
    col += RY

    wheel[col:col+YG, 0] = np.linspace(1, 0, YG)
    wheel[col:col+YG, 1] = 1
    col += YG

    wheel[col:col+GC, 1] = 1
    wheel[col:col+GC, 2] = np.linspace(0, 1, GC)
    col += GC

    wheel[col:col+CB, 1] = np.linspace(1, 0, CB)
    wheel[col:col+CB, 2] = 1
    col += CB

    wheel[col:col+BM, 2] = 1
    wheel[col:col+BM, 0] = np.linspace(0, 1, BM)
    col += BM

    wheel[col:col+MR, 2] = np.linspace(1, 0, MR)
    wheel[col:col+MR, 0] = 1

    return wheel


def flow_to_middlebury(flow, max_flow=None):
    u, v = flow[..., 0], flow[..., 1]
    rad = np.sqrt(u**2 + v**2)
    if max_flow is None:
        max_flow = np.max(rad) + 1e-6

    u /= max_flow
    v /= max_flow

    angle = np.arctan2(-v, -u) / np.pi
    fk = (angle + 1) / 2 * 55

    k0 = np.floor(fk).astype(int)
    k1 = (k0 + 1) % 55
    f = fk - k0

    wheel = make_colorwheel()
    img = (1 - f)[..., None] * wheel[k0] + f[..., None] * wheel[k1]
    img *= (rad[..., None] / max_flow)

    return np.clip(img, 0, 1)


# ============================================================
# FlowNet Simple (TF2)
# ============================================================

def FlowNetSimple():
    inputs = layers.Input(shape=(384, 512, 6))

    x1 = layers.Conv2D(32, 7, strides=2, padding='same', activation='relu')(inputs)
    x2 = layers.Conv2D(64, 5, strides=2, padding='same', activation='relu')(x1)
    x3 = layers.Conv2D(128, 5, strides=2, padding='same', activation='relu')(x2)

    flow3 = layers.Conv2D(2, 3, padding='same')(x3)

    up2 = layers.Conv2DTranspose(64, 4, strides=2, padding='same', activation='relu')(x3)
    flow3_up = layers.Conv2DTranspose(2, 4, strides=2, padding='same')(flow3)
    concat2 = layers.Concatenate()([up2, x2, flow3_up])
    flow2 = layers.Conv2D(2, 3, padding='same')(concat2)

    up1 = layers.Conv2DTranspose(32, 4, strides=2, padding='same', activation='relu')(concat2)
    flow2_up = layers.Conv2DTranspose(2, 4, strides=2, padding='same')(flow2)
    concat1 = layers.Concatenate()([up1, x1, flow2_up])
    flow1 = layers.Conv2D(2, 3, padding='same')(concat1)

    return models.Model(inputs, flow1)


# ============================================================
# Dataset loader (FlyingChairs / Middlebury)
# ============================================================

def load_sample(root, name):
    img1 = plt.imread(root / f"{name}_img1.ppm") / 255.0
    img2 = plt.imread(root / f"{name}_img2.ppm") / 255.0
    flow = read_flo(root / f"{name}_flow.flo")

    x = np.concatenate([img1, img2], axis=-1)
    return x[None], flow


# ============================================================
# Demo / Visualization
# ============================================================

def main():
    data_root = Path("/Users/oscar/Downloads/FlyingChairs_release/data")

    model = FlowNetSimple()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="mse"
    )

    # Load one sample
    x, gt_flow = load_sample(data_root, "00001")

    # Predict
    pred_flow = model.predict(x)[0]

    # Resize GT for display
    gt_flow = tf.image.resize(gt_flow[None], pred_flow.shape[:2])[0].numpy()

    # Show Middlebury visualization
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(flow_to_middlebury(gt_flow))
    ax[0].set_title("Ground Truth Flow")
    ax[1].imshow(flow_to_middlebury(pred_flow))
    ax[1].set_title("Predicted Flow")
    plt.show()


if __name__ == "__main__":
    main()

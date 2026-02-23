import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def flow_to_color(flow):
    u = flow[..., 0].numpy()
    v = flow[..., 1].numpy()
    magnitude = np.sqrt(u**2 + v**2)
    angle = np.arctan2(v, u)
    # normalize
    magnitude = magnitude / (magnitude.max() + 1e-5)
    angle = (angle + np.pi) / (2 * np.pi)  # 0 to 1
    # HSV to RGB
    hsv = np.stack([angle, np.ones_like(angle), magnitude], axis=-1)
    from matplotlib.colors import hsv_to_rgb
    return hsv_to_rgb(hsv)

def visualize(frame1, frame2, flow):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(frame1[0].numpy())
    axes[0].set_title("Frame 1")
    axes[1].imshow(frame2[0].numpy())
    axes[1].set_title("Frame 2")
    axes[2].imshow(flow_to_color(flow[0]))
    axes[2].set_title("Predicted Flow")
    plt.savefig("flow_visualization.png")
    plt.show()
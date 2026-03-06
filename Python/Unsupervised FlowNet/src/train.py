import tensorflow as tf
import argparse
import os
from model import FlowNet
from losses import multiscale_loss
from dataloader import get_dataset
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--epochs',         type=int,   default=50)
parser.add_argument('--batch_size',     type=int,   default=4)
parser.add_argument('--data_root',      type=str,   default='./FlyingChairs_release')
parser.add_argument('--checkpoint_dir', type=str,   default='./checkpoints')
parser.add_argument('--lr',             type=float, default=1e-4)
args = parser.parse_args()

train_dataset = get_dataset(
    args.data_root, args.batch_size,
    split='train', shuffle=True
)
val_dataset = get_dataset(
    args.data_root, args.batch_size,
    split='val', shuffle=False
)

model = FlowNet()

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=args.lr,
    decay_steps=100000,
    decay_rate=0.5,
    staircase=True
)
optimizer = tf.keras.optimizers.Adam(
    learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999
)

os.makedirs(args.checkpoint_dir, exist_ok=True)
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
manager    = tf.train.CheckpointManager(checkpoint, args.checkpoint_dir, max_to_keep=3)

if manager.latest_checkpoint:
    checkpoint.restore(manager.latest_checkpoint)
    print(f"Restored from {manager.latest_checkpoint}")
else:
    print("Starting from scratch")

loss_params = {
    "photo":      {"alpha": 0.25, "beta": 1.0},
    "smooth":     {"alpha": 0.37, "beta": 1.0, "weight": 1.0},
    "smooth_occ": {"alpha": 0.37, "beta": 1.0},
    "smooth2nd":  {"alpha": 0.37, "beta": 1.0, "weight": 0.0},
    "grad":       {"alpha": 0.37, "beta": 1.0, "weight": 0.0},
    "boundary_alpha":        0.0,
    "use_asymmetric_smooth": True,
    "use_smooth2nd":         True,
    "use_grad_loss":         True,
    "use_boundaries":        True,
}

SCALE_WEIGHTS = [1.0, 0.5, 0.25, 0.125, 0.0625]


@tf.function
def train_step(frame1, frame2):
    with tf.GradientTape() as tape:
        inputs = tf.concat([frame1, frame2], axis=-1)          
        outputs = model(inputs, training=True)
        flows_coarse_to_fine = list(reversed(outputs))       
        loss, scale_losses = multiscale_loss(
            flows_coarse_to_fine, frame1, frame2,
            params=loss_params, scale_weights=SCALE_WEIGHTS
        )
    gradients = tape.gradient(loss, model.trainable_variables)
    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, scale_losses


@tf.function
def val_step(frame1, frame2):
    inputs = tf.concat([frame1, frame2], axis=-1)
    outputs = model(inputs, training=False)
    loss, scale_losses = multiscale_loss(
        [outputs], frame1, frame2,
        params=loss_params, scale_weights=[1.0]
    )
    return loss

with open("training_log.txt", "w") as log_file:
    log_file.write("epoch,train_loss,val_loss\n")

    for epoch in range(args.epochs):
        train_loss_sum = 0.0
        num_train      = 0

        for frame1, frame2, _ in tqdm(train_dataset, desc=f"Epoch {epoch+1}/{args.epochs} [train]"):
            loss, _ = train_step(frame1, frame2)
            train_loss_sum += loss.numpy()
            num_train      += 1

        avg_train = train_loss_sum / max(num_train, 1)

        val_loss_sum = 0.0
        num_val      = 0

        for frame1, frame2, _ in tqdm(val_dataset, desc=f"Epoch {epoch+1}/{args.epochs} [val]  ", leave=False):
            val_loss_sum += val_step(frame1, frame2).numpy()
            num_val      += 1

        avg_val = val_loss_sum / max(num_val, 1)

        log_line = f"{epoch+1},{avg_train:.4f},{avg_val:.4f}"
        print(f"Epoch {epoch+1}/{args.epochs} — train: {avg_train:.4f} | val: {avg_val:.4f}")
        log_file.write(log_line + "\n")
        log_file.flush()

        if (epoch + 1) % 5 == 0:
            manager.save()
            print(f"Checkpoint saved at epoch {epoch+1}")

    model.save_weights("flownet.weights.h5")
    print("Training complete.")
import tensorflow as tf
import argparse
from model import FlowNet
from losses import multiscale_epe_loss
from dataloader import get_dataset
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--data_path', type=str, default='./data')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
args = parser.parse_args()

train_dataset = get_dataset(args.data_path, args.batch_size, split='train')
val_dataset   = get_dataset(args.data_path, args.batch_size, split='val')

model = FlowNet()
lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=[300000, 400000, 500000],
    values=[1e-4, 5e-5, 2.5e-5, 1.25e-5]
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
manager = tf.train.CheckpointManager(checkpoint, args.checkpoint_dir, max_to_keep=3)

if manager.latest_checkpoint:
    checkpoint.restore(manager.latest_checkpoint)
    print(f"Restored from {manager.latest_checkpoint}")
else:
    print("Starting from scratch")

@tf.function
def train_step(frame1, frame2, flow_gt, valid):
    with tf.GradientTape() as tape:
        inputs = tf.concat([frame1, frame2], axis=-1)
        flows = model(inputs, training=True)
        loss, scale_losses = multiscale_epe_loss(flows, flow_gt, valid)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, scale_losses

@tf.function
def val_step(frame1, frame2, flow_gt, valid):
    inputs = tf.concat([frame1, frame2], axis=-1)
    flows = model(inputs, training=False)
    loss, scale_losses = multiscale_epe_loss(flows, flow_gt, valid)
    return loss, scale_losses

log_file = open("training_log.txt", "w")
log_file.write("Type, Epoch, Total, Scale1, Scale2, Scale3, Scale4\n")

for epoch in range(args.epochs):

    # --- Training ---
    epoch_loss = 0.0
    epoch_scale_losses = None
    num_batches = 0

    for frame1, frame2, flow_gt, valid in tqdm(train_dataset, desc=f"Epoch {epoch+1}/{args.epochs}"):
        loss, scale_losses = train_step(frame1, frame2, flow_gt, valid)
        epoch_loss += loss.numpy()
        if epoch_scale_losses is None:
            epoch_scale_losses = [0.0] * len(scale_losses)
        for i, sl in enumerate(scale_losses):
            epoch_scale_losses[i] += sl.numpy()
        num_batches += 1

    scale_str = " | ".join([f"Scale {i+1}: {l/num_batches:.4f}" for i, l in enumerate(epoch_scale_losses)])
    train_log_line = f"Train {epoch+1}/{args.epochs} — Total: {epoch_loss/num_batches:.4f} | {scale_str}"
    print(train_log_line)
    log_file.write(train_log_line + "\n")
    log_file.flush()

    # --- Validation ---
    val_loss = 0.0
    val_scale_losses = None
    val_batches = 0

    for frame1, frame2, flow_gt, valid in tqdm(val_dataset, desc=f"Val   {epoch+1}/{args.epochs}"):
        loss, scale_losses = val_step(frame1, frame2, flow_gt, valid)
        val_loss += loss.numpy()
        if val_scale_losses is None:
            val_scale_losses = [0.0] * len(scale_losses)
        for i, sl in enumerate(scale_losses):
            val_scale_losses[i] += sl.numpy()
        val_batches += 1

    val_scale_str = " | ".join([f"Scale {i+1}: {l/val_batches:.4f}" for i, l in enumerate(val_scale_losses)])
    val_log_line = f"Val   {epoch+1}/{args.epochs} — Total: {val_loss/val_batches:.4f} | {val_scale_str}"
    print(val_log_line)
    log_file.write(val_log_line + "\n")
    log_file.flush()

    # --- Checkpoint ---
    if (epoch + 1) % 5 == 0:
        manager.save()
        print(f"Checkpoint saved at epoch {epoch + 1}")

model.save_weights("flownet.weights.h5")
log_file.close()
print("Training complete.")
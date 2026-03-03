import tensorflow as tf
import argparse
from model import FlowNet
from losses import multiscale_loss
from dataloader import get_dataset
from tqdm import tqdm
from utils import visualize

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--data_path', type=str, default='./data')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
args = parser.parse_args()

train_dataset = get_dataset(args.data_path, args.batch_size)

model = FlowNet()

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1.6e-5,
    decay_steps=100000,
    decay_rate=0.5,
    staircase=True  # hard steps, not smooth decay
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999)

checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
manager = tf.train.CheckpointManager(checkpoint, args.checkpoint_dir, max_to_keep=3)

if manager.latest_checkpoint:
    checkpoint.restore(manager.latest_checkpoint)
    print(f"Restored from {manager.latest_checkpoint}")
else:
    print("Starting from scratch")

@tf.function
def train_step(frame1, frame2):
    with tf.GradientTape() as tape:
        inputs = tf.concat([frame1, frame2], axis=-1)
        flows = model(inputs, training=True)
        loss, scale_losses = multiscale_loss(flows, frame1, frame2)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, scale_losses

log_file = open("training_log.txt", "w")
log_file.write("Epoch, Total, Scale1, Scale2, Scale3, Scale4\n")

for epoch in range(args.epochs):
    epoch_loss = 0.0
    epoch_scale_losses = None
    num_batches = 0

    for frame1, frame2 in tqdm(train_dataset, desc=f"Epoch {epoch+1}/{args.epochs}"):
        loss, scale_losses = train_step(frame1, frame2)
        epoch_loss += loss.numpy()

        if epoch_scale_losses is None:
            epoch_scale_losses = [0.0] * len(scale_losses)
        for i, sl in enumerate(scale_losses):
            epoch_scale_losses[i] += sl.numpy()
        num_batches += 1

    scale_str = " | ".join([f"Scale {i+1}: {l/num_batches:.4f}" for i, l in enumerate(epoch_scale_losses)])
    log_line = f"Epoch {epoch+1}/{args.epochs} — Total: {epoch_loss/num_batches:.4f} | {scale_str}"
    print(log_line)
    log_file.write(log_line + "\n")
    log_file.flush()

    if (epoch + 1) % 5 == 0:
        manager.save()
        print(f"Checkpoint saved at epoch {epoch + 1}")
model.save_weights("flownet.weights.h5")
log_file.close()
print("Training complete.")
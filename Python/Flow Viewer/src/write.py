import tensorflow as tf
from tensorflow import keras
from PIL import Image
import argparse
import numpy as np
import sys
from model import FlowNet


parser = argparse.ArgumentParser()
parser.add_argument('--frame1', type=str)
parser.add_argument('--frame2', type=str)
parser.add_argument('--weights_path', type=str)
parser.add_argument('--model_path', type=str)
args = parser.parse_args()

#sys.path.append(args.model_path)
frame1_path=args.frame1
frame2_path=args.frame2

def load_frame(path):
    img = Image.open(path).convert('RGB')
    img = np.array(img, dtype=np.float32) / 255.0
    img = tf.convert_to_tensor(img)
    img = tf.image.resize(img, [384, 512])
    img = tf.expand_dims(img, axis=0)
    return img

frame1=load_frame(frame1_path)
frame2=load_frame(frame2_path)
inputs = tf.concat([frame1, frame2], axis=-1)

model = FlowNet()
dummy = tf.zeros([1, 384, 512, 6])
model(dummy)
model.load_weights(args.weights_path)

flows=model(inputs,training=False)
flow = flows[-1][0]

def write(flow, filename):
	height, width, nBands = flow.shape
	u = flow[: , : , 0]
	v = flow[: , : , 1]	
	height, width = u.shape
	f = open(filename,'wb')
	f.write(b'PIEH')
	np.array(width).astype(np.int32).tofile(f)
	np.array(height).astype(np.int32).tofile(f)
	tmp = np.zeros((height, width*nBands))
	tmp[:,np.arange(width)*2] = u
	tmp[:,np.arange(width)*2 + 1] = v
	tmp.astype(np.float32).tofile(f)
	f.close()

write(flow, 'sortie.flo')

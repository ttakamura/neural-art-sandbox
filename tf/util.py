import numpy as np
import scipy.misc
import os
from datetime import datetime as dt
import argparse
import models
import tensorflow as tf

JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'

mean = np.array([103.939, 116.779, 123.68], dtype=np.float32)
def add_mean(img):
  for i in range(3):
    img[0,:,:,i] += mean[i]
  return img

def sub_mean(img):
  for i in range(3):
    img[0,:,:,i] -= mean[i]
  return img

def read_image(path, w=None):
  img = scipy.misc.imread(path)
  # Resize if ratio is specified
  if w:
    r = w / np.float32(img.shape[1])
    img = scipy.misc.imresize(img, (int(img.shape[0]*r), int(img.shape[1]*r)))
  img = img.astype(np.float32)
  img = img[None, ...]
  # Subtract the image mean
  img = sub_mean(img)
  return img

def save_image(im, iteration, out_dir):
  img = im.copy()
  # Add the image mean
  img = add_mean(img)
  img = np.clip(img[0, ...],0,255).astype(np.uint8)
  nowtime = dt.now().strftime('%Y_%m_%d_%H_%M_%S')
  if not os.path.exists(out_dir):
    os.mkdir(out_dir)
  scipy.misc.imsave("{}/neural_art_{}_iteration{}.png".format(out_dir, nowtime, iteration), img)

def getModel(sess, graph_path, model):
  if model == 'inception':
    return models.InceptionV3(sess, graph_path)
  else:
    print('Invalid model name: use `inception`')
    return None

def load_network(sess, graph_path):
  with tf.gfile.FastGFile(graph_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    jpeg_data_tensor = tf.import_graph_def(graph_def, name='', return_elements=[JPEG_DATA_TENSOR_NAME])
  return jpeg_data_tensor

import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 0.01
MARGIN = 1.5
CAPTION_INPUT_SIZE = 300
FRAME_INPUT_SIZE = 500
CAPTION_LATENT_SIZE = 256
FRAME_LATENT_SIZE = 256

X1_placeholder = tf.placeholder(tf.float32, shape = [None, None, CAPTION_INPUT_SIZE])
X2_placeholder = tf.placeholder(tf.float32, shape = [None, None, FRAME_INPUT_SIZE])

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.8)

def train_caption_embeddings(x_placeholder, latent_dim):
    cell = tf.nn.rnn_cell.GRUCell(latent_dim, kernel_initializer = tf.contrib.layers.variance_scaling_initializer())
    cells = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob = 0.5)
    _, s = tf.nn.dynamic_rnn(cells, x_placeholder, dtype = tf.float32, swap_memory = True)
    return s

def train_frame_embeddings(x_placeholder, latent_dim):
    cell = tf.nn.rnn_cell.GRUCell(latent_dim, kernel_initializer = tf.contrib.layers.variance_scaling_initializer())
    cells = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob = 0.5)
    _, s = tf.nn.dynamic_rnn(cells, x_placeholder, dtype = tf.float32, swap_memory = True)
    return s

caption_out = train_caption_embeddings(X1_placeholder, CAPTION_LATENT_SIZE) #Anchor
frame_out_1 = train_frame_embeddings(X2_placeholder, FRAME_LATENT_SIZE) #Positive
frame_out_2 = train_frame_embeddings(X2_placeholder, FRAME_LATENT_SIZE) #Negative

positive_distance = tf.reduce_sum(tf.square(caption_out - frame_out_1), 1)
negative_distance = tf.reduce_sum(tf.square(caption_out - frame_out_2), 1)

loss = tf.reduce_mean(tf.maximum(0., positive_distance - negative_distance + MARGIN))

init = tf.global_variables_initializer()
sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))

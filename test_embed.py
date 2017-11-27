import numpy as np
import tensorflow as tf
import training_utils

if __name__ == "__main__":

  subdir = 'patient'
  X = np.load('data/%s/gbm/sparse.npy' % subdir)
  E = np.load('data/%s/gbm/embedding_gene_coexpression.npy' % subdir)
  # E = np.load('data/%s/gbm/dummy_embedding.npy' % subdir)
  Y = np.load('data/%s/gbm/labels.npy' % subdir)

  embed_shape = E.shape
  dataset = training_utils.train_test_split({'x': X, 'y': Y}, split=0.8, sparse_keys=['x'])
  print("Dataset contains the following:")
  for key in dataset:
    print(key, dataset[key].shape)
  print("Embedding is size: ", E.shape)
  train_feed = {
    key.split('_')[0] : dataset[key]
    for key in dataset if 'train' in key
  }
  train_feed['embed'] = E

  # train_feed = {
  #   'x':[[0],[0],[1]],
  #   'x_values': [0,1,3], 
  #   'x_shape': [3],
  #   'embed': [[10,0], [11,1], [12,2], [13,3]]
  # }
  # embed_shape = [4,2]
  
  x = tf.placeholder(tf.int64, [None, 1])
  x_values = tf.placeholder(tf.int64, [None,])
  x_shape = tf.placeholder(tf.int64, [1,])
  embed = tf.placeholder(tf.float32, embed_shape)
  sparse_input = tf.SparseTensor(indices=x, values=x_values, dense_shape=x_shape)
  embed_input = tf.nn.embedding_lookup_sparse(embed, sparse_input, None, combiner='sum') 

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sp_input, em_input = sess.run([sparse_input, embed_input], feed_dict = {x: train_feed['x-indices'], x_values: train_feed['x-values'], x_shape: train_feed['x-shape'], embed: train_feed['embed']})
    print(sp_input)
    print(em_input)

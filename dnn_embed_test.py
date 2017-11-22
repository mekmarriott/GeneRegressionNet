import tensorflow as tf
import numpy as np
from models.dnn_embed import TFDNNEmbeddingModel
import training_utils

CANCERS = ['gbm']
DATA_DIR = 'data/dummy_patient' # switch to data/dummy_patient if you don't have access to patient data
SEED = 0
NUM_ITERATIONS = 2000
# EMBEDDING = 'embedding_gene_gene_interaction'
EMBEDDING = 'dummy_embedding'
if __name__ == "__main__":

  np.random.seed(SEED)
  for cancer in CANCERS:
    print("#### CANCER - %s ####" % cancer)

    print("Setting Up .... Loading data")
    X = np.load("%s/%s/sparse.npy" % (DATA_DIR, cancer))
    Y = np.load("%s/%s/labels.npy" % (DATA_DIR, cancer))
    D = np.load("%s/%s/survival.npy" % (DATA_DIR, cancer))
    E = np.load("%s/%s/%s.npy" % (DATA_DIR, cancer, EMBEDDING))
    embed_shape = E.shape
    dataset = training_utils.train_test_split({'x': X, 'y': Y, 'd': D}, split=0.8)
    print("Dataset contains the following:")
    for key in dataset:
      print key, dataset[key].shape
    print("Embedding is size: ", E.shape)
    train_feed = {
      key.split('_')[0] : dataset[key]
      for key in dataset if 'train' in key
    }
    test_feed = {
      key.split('_')[0] : dataset[key]
      for key in dataset if 'test' in key
    }

    train_feed['embed'] = E
    train_feed['x'] = training_utils.flatten_nested_list(train_feed['x'])
    train_feed['x_shape'] = [train_feed['y'].shape[0], embed_shape[0]]
    test_feed['embed'] = E
    test_feed['x'] = training_utils.flatten_nested_list(test_feed['x'])
    test_feed['x_shape'] = [test_feed['y'].shape[0], embed_shape[0]]
    print("*"*40)

    # Check that simple TFDNNEmbeddingModel has a better test error than the linear and simple DNN models without embeddings
    print("Testing custom loss on DNN Embedding Tensorflow Model using %s" % EMBEDDING)
    m = TFDNNEmbeddingModel(embed_shape, alpha=0.001)
    m.initialize()
    m.train(train_feed, num_iterations=NUM_ITERATIONS, debug=True)
    embed_train_loss = m.test(train_feed)
    embed_test_loss = m.test(test_feed)
    print("DNN Embedding Model train loss is %.6f and test loss is %.6f" % (embed_train_loss, embed_test_loss))


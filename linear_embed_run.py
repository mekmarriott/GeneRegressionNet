import tensorflow as tf
import numpy as np
from models.dnn_embed import TFDNNEmbeddingModel
import training_utils

ALPHAS = {'gbm': 0.05, 'luad': 0.08, 'lusc': 0.1}
CANCERS = ['lusc']
DATA_DIR = 'data/patient' # switch to data/dummy_patient if you don't have access to patient data
SEED = 0
NUM_ITERATIONS = 30000
EMBEDDING = 'embedding_gene_gene_interaction'
# EMBEDDING = 'embedding_gene_coexpression'
# EMBEDDING = 'dummy_embedding'
LAYERS = []
ACTIVATION = None
OPTIMIZATION = None
COMBINER = 'mean'

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
    dataset = training_utils.train_test_split({'x': X, 'y': Y, 'd': D}, split=0.8, sparse_keys=['x'])
    print("Dataset contains the following:")
    for key in dataset:
      print(key, dataset[key].shape)
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
    test_feed['embed'] = E
    print("*"*40)

    print("Testing custom loss on DNN Embedding Tensorflow Model using %s" % EMBEDDING)
    m = TFDNNEmbeddingModel(embed_shape, alpha=ALPHAS[cancer], combiner=COMBINER, layer_dims=LAYERS, activation=ACTIVATION, opt=OPTIMIZATION)
    m.initialize()
    for i in range(int(NUM_ITERATIONS/100)):
      m.train(train_feed, num_iterations=100, debug=False)
      embed_train_loss = m.test(train_feed)
      embed_test_loss = m.test(test_feed)
      print("DNN Embedding Model train loss is %.6f and test loss is %.6f" % (embed_train_loss, embed_test_loss))

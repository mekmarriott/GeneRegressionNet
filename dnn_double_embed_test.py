import tensorflow as tf
import numpy as np
from models.dnn_embed_double import TFDNNDoubleEmbeddingModel
import training_utils

ALPHAS = {'gbm': 0.000075, 'luad': 0.001, 'lusc': 0.0011}
CANCERS = ['gbm']
DATA_DIR = 'data/patient' # switch to data/dummy_patient if you don't have access to patient data
RUN_DIR = 'output/loss_history'
SEED = 0
NUM_ITERATIONS = 10000
EMBEDDINGS = ['embedding_gene_gene_interaction','embedding_gene_coexpression']
LAYERS = [2,2]
ACTIVATION = 'relu'
OPTIMIZATION = 'adam'
COMBINER = 'mean'
SURVIVAL = 'cox'

if __name__ == "__main__":

  np.random.seed(SEED)
  for cancer in CANCERS:
    print("#### CANCER - %s ####" % cancer)

    print("Setting Up .... Loading data")
    X = np.load("%s/%s/sparse.npy" % (DATA_DIR, cancer))
    Y = np.load("%s/%s/labels.npy" % (DATA_DIR, cancer))
    D = np.load("%s/%s/survival.npy" % (DATA_DIR, cancer))
    E1 = np.load("%s/%s/%s.npy" % (DATA_DIR, cancer, EMBEDDINGS[0]))
    E2 = np.load("%s/%s/%s.npy" % (DATA_DIR, cancer, EMBEDDINGS[1]))
    embed1_shape, embed2_shape = E1.shape, E2.shape
    if SURVIVAL == 'cox':
      Y = training_utils.discretize_label(Y, D)
    time_buckets = Y.shape[1]
    dataset = training_utils.train_test_split({'x': X, 'y': Y, 'd': D}, split=0.8, sparse_keys=['x'])
    print("Dataset contains the following:")
    for key in dataset:
      print(key, dataset[key].shape)
    print("Embedding is size: ", E1.shape, E2.shape)
    train_feed = {
      key.split('_')[0] : dataset[key]
      for key in dataset if 'train' in key
    }
    test_feed = {
      key.split('_')[0] : dataset[key]
      for key in dataset if 'test' in key
    }

    train_feed['embed1'] = E1
    train_feed['embed2'] = E2
    test_feed['embed1'] = E1
    test_feed['embed2'] = E2
    print("*"*40)

    print("Testing custom loss on DNN Embedding Tensorflow Model using %s" % str(EMBEDDINGS))
    loss_history = []
    m = TFDNNDoubleEmbeddingModel(embed1_shape, embed2_shape, alpha=ALPHAS[cancer], combiner=COMBINER, layer_dims=LAYERS, loss=SURVIVAL, activation=ACTIVATION, opt=OPTIMIZATION, output_sz=time_buckets)
    m.initialize()
    for i in range(int(NUM_ITERATIONS/100)):
      m.train(train_feed, num_iterations=100, debug=False)
      embed_train_loss = m.test(train_feed)
      embed_test_loss = m.test(test_feed)
      loss_history.append(np.array([i,embed_train_loss, embed_test_loss]))
      print("DNN Embedding Model train loss is %.6f and test loss is %.6f" % (embed_train_loss, embed_test_loss))

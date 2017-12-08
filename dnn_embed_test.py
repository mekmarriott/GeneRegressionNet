import tensorflow as tf
import numpy as np
from models.dnn_embed import TFDNNEmbeddingModel
import training_utils

ALPHAS = {'gbm': 0.001, 'luad': 0.0003, 'lusc': 0.001}
CANCERS = ['lusc']
DATA_DIR = 'data/patient' # switch to data/dummy_patient if you don't have access to patient data
RUN_DIR = 'output/loss_history'
SEED = 0
NUM_ITERATIONS = 5000
EMBEDDING = 'embedding_gene_gene_interaction'
# EMBEDDING = 'embedding_gene_coexpression'
# EMBEDDING = 'dummy_embedding'
LAYERS = [16, 4]
ACTIVATION = 'relu'
OPTIMIZATION = 'adam'
COMBINER = 'mean'
SURVIVAL = 'cox'
CV_FOLD = 4

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
    if SURVIVAL == 'cox':
      Y = training_utils.discretize_label(Y, D)
    time_buckets = Y.shape[1]
    dataset = training_utils.train_test_split({'x': X, 'y': Y, 'd': D}, split=0.8, sparse_keys=['x'])
    print("Dataset contains the following:")
    for key in dataset:
      print(key, dataset[key].shape)
    print("Embedding is size: ", E.shape)
    data_feed = {
      key.split('_')[0] : dataset[key]
      for key in dataset if 'train' in key
    }
    validation_feed = {
      key.split('_')[0] : dataset[key]
      for key in dataset if 'test' in key
    }

    datasets = training_utils.cv_split(data_feed, partitions=CV_FOLD, sparse_keys=['x'])
    print("CV Dataset contains the following:")
    for key in dataset:
      print(key, dataset[key].shape)

    for d in datasets:
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
      loss_history = []
      m = TFDNNEmbeddingModel(embed_shape, alpha=ALPHAS[cancer], combiner=COMBINER, layer_dims=LAYERS, loss=SURVIVAL, activation=ACTIVATION, opt=OPTIMIZATION, output_sz=time_buckets)
      m.initialize()
      for i in range(int(NUM_ITERATIONS/100)):
        m.train(train_feed, num_iterations=100, debug=False)
        embed_train_loss = m.test(train_feed)
        embed_test_loss = m.test(test_feed)
        loss_history.append(np.array([i,embed_train_loss, embed_test_loss]))
        print("DNN Embedding Model train loss is %.6f and test loss is %.6f" % (embed_train_loss, embed_test_loss))

      # Save loss history data under descriptive name
      np.save('%s/%s/%s_layers-%s_%s_%s_%s_it-%d_alpha-%s' % (RUN_DIR, cancer, EMBEDDING, '-'.join([str(l) for l in LAYERS]), ACTIVATION, OPTIMIZATION, COMBINER, NUM_ITERATIONS, str(ALPHAS[cancer])), loss_history)

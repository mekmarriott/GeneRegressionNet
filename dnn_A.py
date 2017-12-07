import tensorflow as tf
import numpy as np
from models.linear import TFLinearModel
from models.dnn import TFDNNModel
import training_utils

CANCERS = ['gbm', 'luad', 'lusc']
ALPHAS = {'gbm': 0.001, 'luad': 0.001, 'lusc': 0.001}
DATA_DIR = 'data/patient' # switch to data/dummy_patient if you don't have access to patient data
SEED = 0
NUM_ITERATIONS = 3000
LAYERS = [128, 64, 32]
OPTIMIZATION = 'ada'
ACTIVATION = 'relu'

if __name__ == "__main__":

  np.random.seed(SEED)
  for cancer in CANCERS:
    print("#### CANCER - %s ####" % cancer)

    print("Setting Up .... Loading data")
    X = np.load("%s/%s/one_hot.npy" % (DATA_DIR, cancer))
    Y = np.load("%s/%s/labels.npy" % (DATA_DIR, cancer))
    D = np.load("%s/%s/survival.npy" % (DATA_DIR, cancer))
    num_features = X.shape[1]
    dataset = training_utils.train_test_split({'x': X, 'y': Y, 'd': D}, split=0.8)
    print("Dataset contains the following:")
    for key in dataset:
      print(key, dataset[key].shape)
    train_feed = {
      key.split('_')[0] : dataset[key]
      for key in dataset if 'train' in key
    }
    test_feed = {
      key.split('_')[0] : dataset[key]
      for key in dataset if 'test' in key
    }
    print("*"*40)

    m = TFDNNModel(num_features, activation=ACTIVATION, alpha=ALPHAS[cancer], layer_dims=LAYERS, opt=OPTIMIZATION)
    m.initialize()
    for _ in range(int(NUM_ITERATIONS/100)):
      m.train(train_feed, num_iterations=100)
      dnn_train_loss = m.test(train_feed)
      dnn_test_loss = m.test(test_feed)
      print("DNN TFModel train loss is %.6f and test loss is %.6f" % (dnn_train_loss, dnn_test_loss))

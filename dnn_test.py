import tensorflow as tf
import numpy as np
from models.linear import TFLinearModel
from models.dnn import TFDNNModel
import training_utils

CANCERS = ['gbm', 'luad', 'lusc']
ALPHAS = {'gbm': 0.5, 'luad': 0.085, 'lusc': 0.15}
DATA_DIR = 'data/patient' # switch to data/dummy_patient if you don't have access to patient data
SEED = 0
NUM_ITERATIONS = 2000
LAYERS = []
OPTIMIZATION = 'sgd'
ACTIVATION = None

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
    
    # Check that TFLinearModel is the same as TFDNNModel with no layer_input and overfits (training loss -> 0, test loss > training loss)
    print("Testing custom loss on Linear Tensorflow Model")
    m = TFLinearModel(num_features, alpha=ALPHAS[cancer])
    m.initialize()
    m.train(train_feed, num_iterations=NUM_ITERATIONS, debug=True)
    linear_train_loss = m.test(train_feed)
    linear_test_loss = m.test(test_feed)
    print("Linear TFModel train loss is %.6f and test loss is %.6f" % (linear_train_loss, linear_test_loss))
    
    # Check that TFDNNModel overfits (training loss -> 0, test loss > training loss)
    print("*"*40)
    print("Testing custom loss on DNN no layer input Tensorflow Model")
    m = TFDNNModel(num_features, activation=ACTIVATION, alpha=ALPHAS[cancer], layer_dims=LAYERS, opt=OPTIMIZATION)
    m.initialize()
    m.train(train_feed, num_iterations=NUM_ITERATIONS, debug=True)
    dnn_train_loss = m.test(train_feed)
    dnn_test_loss = m.test(test_feed)
    print("DNN TFModel train loss is %.6f and test loss is %.6f" % (dnn_train_loss, dnn_test_loss))

    print("*"*40)
    if abs(linear_train_loss - dnn_train_loss) < 1e-6 and linear_test_loss > linear_train_loss and dnn_test_loss > dnn_train_loss:
      print("PASS: Linear and DNN-linear models achieved same (zero) training error and overfit")
    else:
      print("FAIL: Linear and DNN-linear models did not behave as expected")

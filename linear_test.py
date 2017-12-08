import tensorflow as tf
import numpy as np
from sklearn import linear_model
from sklearn import metrics
from models import model
import training_utils

CANCERS = ['gbm', 'luad', 'lusc']
ALPHAS = {'gbm': 0.5, 'luad': 0.085, 'lusc': 0.05}
DATA_DIR = 'data/patient' # switch to data/dummy_patient if you don't have access to patient data
SEED = 0
NUM_ITERATIONS = 2000

if __name__ == "__main__":

  np.random.seed(SEED)
  for cancer in CANCERS:
    print("#### CANCER - %s ####" % cancer)
    
    print("Setting Up .... Loading data")
    X = np.load("%s/%s/one_hot.npy" % (DATA_DIR, cancer))
    Y = np.load("%s/%s/labels.npy" % (DATA_DIR, cancer))
    num_features = X.shape[1]
    dataset = training_utils.train_test_split({'x': X, 'y': Y}, split=0.8)
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
    print("Testing regular loss on Tensorflow Model")
    m = model.TFModel(num_features, alpha=ALPHAS[cancer])
    m.initialize()
    m.train(train_feed, num_iterations=NUM_ITERATIONS, debug=True)
    tf_train_loss = m.test(train_feed)
    print("Training error after %d is %.6f" % (NUM_ITERATIONS, tf_train_loss))
    print("Test error is %.6f" % (m.test(test_feed)))

    print("*"*40)
    print("Testing loss on sklearn linear model")
    regr = linear_model.LinearRegression()
    regr.fit(dataset['x_train'], dataset['y_train'])
    pred_y = regr.predict(dataset['x_train'])
    sklearn_train_loss = metrics.mean_squared_error(dataset['y_train'], pred_y)
    print("Loss of linear regression is ", sklearn_train_loss)

    print("*"*40)
    if abs(tf_train_loss - sklearn_train_loss):
      print("PASS: Training losses are approximately equal/zero")
    else:
      print("FAIL: Expected equivalent training losses but got %.6f for TF model training loss and %.6f for sklearn model training loss")

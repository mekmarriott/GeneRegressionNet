import tensorflow as tf
import numpy as np
import argparse
from models.linear import TFLinearModel
from models.dnn import TFDNNModel
import training_utils
import visualization_utils

CANCERS = ['gbm', 'luad', 'lusc']
ALPHAS = {'gbm': 0.3, 'luad': 0.085, 'lusc': 0.15}
DATA_DIR = 'data/patient' # switch to data/dummy_patient if you don't have access to patient data

parser = argparse.ArgumentParser(description='Define learning parameters')
parser.add_argument('-alpha', metavar='a', nargs='?', type=float)
parser.add_argument('-cancer', metavar='c', nargs='?', type=str)
parser.add_argument('-iterations', metavar='it', nargs='?', type=int, default=1000)
parser.add_argument('-layers', metavar='l', nargs='?', type=str, default=None)
parser.add_argument('-optimization', metavar='op', nargs='?', type=str, default='sgd')
parser.add_argument('-mode', metavar='m', nargs='?', type=str, default='cv')
parser.add_argument('-activation', metavar='ac', nargs='?', type=str, default='relu')
parser.add_argument('-loss', metavar='s', nargs='?', type=str, default='normal')
parser.add_argument('-cv_fold', metavar='f', nargs='?', type=int, default=4)
parser.add_argument('-seed', metavar='s', nargs='?', type=int, default=0)
parser.add_argument('-debug_steps', metavar='ds', nargs='?', type=int, default=100)
parser.add_argument('-cv_split', metavar='cs', nargs='?', type=float, default=0.8)
parser.add_argument('-dropout', metavar='d', nargs='?', type=float)
parser.add_argument('-regularization', metavar='r', nargs='?', type=float, default=0.0)
parser.add_argument('-visualize_performance', metavar='vp', nargs='?', type=bool, default=False)
if __name__ == "__main__":

  args = parser.parse_args()
  print(args)
  np.random.seed(args.seed)
  tf.set_random_seed(args.seed)
  if args.cancer is not None: CANCERS = [args.cancer]
  if args.layers is None:
     args.layers = []
  else:
     args.layers = [int(layer) for layer in args.layers.split('-')]

  for cancer in CANCERS:
    print("#### CANCER - %s ####" % cancer)

    print("Setting Up .... Loading data")
    X = np.load("%s/%s/one_hot.npy" % (DATA_DIR, cancer))
    Y = np.load("%s/%s/labels.npy" % (DATA_DIR, cancer))
    D = np.load("%s/%s/survival.npy" % (DATA_DIR, cancer))
    num_features = X.shape[1]
    output_sz = 1
    if args.loss == 'softmax':
      Y = training_utils.discretize_label(Y, D)
    output_sz = Y.shape[1]

    dataset = training_utils.train_test_split({'x': X, 'y': Y, 'd': D}, split=args.cv_split)
    print("Dataset contains the following:")
    for key in dataset:
      print(key, dataset[key].shape)

    performance = []
    if args.mode == 'cv':
      print("CROSS VALIDATION TEST")
      data_feed = {
        key.split('_')[0] : dataset[key]
        for key in dataset if 'train' in key
      }
      datasets = training_utils.cv_split(data_feed, partitions=args.cv_fold)
    elif args.mode == 'test':
      print("TESTING")
      datasets = [dataset]
 
    for d in datasets:
      train_feed = {
        key.split('_')[0] : dataset[key]
        for key in dataset if 'train' in key
      }
      test_feed = {
        key.split('_')[0] : dataset[key]
        for key in dataset if 'test' in key
      }
      if args.dropout is not None:
        train_feed['keep_prob'] = [args.dropout]
        test_feed['keep_prob'] = [1.0]
      print("*"*40)

      print("Testing custom loss on DNN Tensorflow Model")
      if args.alpha is None:
        args.alpha = ALPHAS[cancer]
      m = TFDNNModel(num_features, activation=args.activation, alpha=args.alpha, regularization=args.regularization, layer_dims=args.layers, loss=args.loss, opt=args.optimization, output_sz=output_sz)
      m.initialize()
      for i in range(int(args.iterations/args.debug_steps)):
        m.train(train_feed, num_iterations=args.debug_steps, debug=False)
        dnn_train_loss = m.test(train_feed)
        dnn_test_loss = m.test(test_feed)
        print("Epoch %d train loss is %.6f and test loss is %.6f" % (i*args.debug_steps, dnn_train_loss, dnn_test_loss))
      performance.append((dnn_train_loss, dnn_test_loss))

      # Look at weights for any particular gene of interest
      # W, b = m.get_params(train_feed)
      # visualization_utils.plot_performance(test_feed['y'], test_feed['d'], m.predict(test_feed))
    if args.visualize_performance:
      # visualization_utils.plot_performance(train_feed['y'], train_feed['d'], m.predict(train_feed), 'Linear No-Embed (Train): Predictions vs. Ground Truth')
      # visualization_utils.plot_performance(test_feed['y'], test_feed['d'], m.predict(test_feed), 'Linear No-Embed (Test): Predictions vs. Ground Truth')
        visualization_utils.plot_train_test_performance(train_feed, test_feed, m.predict(train_feed), m.predict(test_feed), title='Linear No-Embed Predictions vs. Ground Truth for GBM')
      # break
    # Give summary of loss, if CV
    if args.mode == 'cv':
      print("CROSS VALIDATION PERFORMANCE: train is %.6f and tests is %.6f" % (np.mean([x[0] for x in performance]), np.mean([x[1] for x in performance])))
    

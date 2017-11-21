import tensorflow as tf

def optimization_fn(opt, alpha=0.001):
  if opt == 'sgd' or opt is None:
    return tf.train.GradientDescentOptimizer(alpha)
  elif opt == 'adam':
    return tf.train.GradientDescentOptimizer(alpha)
  elif top == 'ada':
    return tf.train.GradientDescentOptimizer(alpha)
  else:
    raise ValueError("Optimization %s not recognized!" % opt)

def activation_fn(activation):
  # Does nothing, identity function
  if activation is None:
    return tf.identity
  elif activation == 'relu':
    return tf.nn.relu
  elif activation == 'relu6':
    return tf.nn.relu6
  elif activation == 'sigmoid':
    return tf.sigmoid
  elif activation == 'tanh':
    return tf.tanh
  else:
    raise ValueError("Activation %s not recognized!" % activation)

def aggregation_fn(agg):
  if agg == 'sum':
    return tf.reduce_sum
  elif agg == 'avg':
    return tf.reduce_mean
  elif agg == 'max':
    return tf.reduce_max
  elif agg == 'min':
    return tf.reduce_min
  else:
    raise ValueError("Aggregation %s not recognized!" % agg)

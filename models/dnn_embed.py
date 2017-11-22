import sys
import model_utils
import tensorflow as tf
from model import TFModel

class TFDNNEmbeddingModel(TFModel):
  def __init__(self, embed_shape, alpha=0.001, layer_dims=[], combiner='mean', opt='sgd', activation=None):
    # Inherit all self attributes
    TFModel.__init__(self, embed_shape[1], alpha=alpha)
    self.inputs['x'] = tf.placeholder(tf.int64, [None, 2]) # two is the number of dimensions in tuple (example_idx, gene_idx)
    self.inputs['x_shape'] = tf.placeholder(tf.int64, [2,]) # the shape of the dense representation of x
    self.inputs['d'] = tf.placeholder(tf.float32, [None, 1])
    self.inputs['embed'] = tf.placeholder(tf.float32, embed_shape) 
    
    layers = [embed_shape[1]] + layer_dims + [1]
    self.W = {}
    self.b = {}
    
    # Create sparse input
    weights = tf.cast(tf.ones([embed_shape[0]]), tf.int64)
    sparse_input = tf.SparseTensor(self.inputs['x'], weights, self.inputs['x_shape'])
    print sparse_input
    
    # Look up embeddings from one hot encodings
    self.embed_input = tf.nn.embedding_lookup_sparse(self.inputs['embed'], sparse_input, None, combiner=combiner)
    print self.embed_input
    self.layer_inputs = {
        0: self.embed_input
    }
    # Fill in layers
    for i in range(len(layers) - 1):
      self.W[i] = tf.Variable(tf.random_normal([layers[i], layers[i+1]]))
      self.b[i] = tf.Variable(tf.random_normal([layers[i+1]]))
      self.layer_inputs[i+1] = tf.add(tf.matmul(self.layer_inputs[i], self.W[i]), self.b[i])
      self.layer_inputs[i+1] = model_utils.activation_fn(activation)(self.layer_inputs[i+1])

    self.pred = self.layer_inputs[len(self.layer_inputs)-1]
    # Survival loss - if the person is alive (1-d = 1) AND the prediction is greater than the label, apply no loss. Otherwise apply L2 loss.
    survival_loss = tf.square(tf.maximum(self.inputs['y']-self.pred, 0))
    # Non-survival loss - if the person is dead (d = 1) apply L2 loss
    non_survival_loss = tf.square(self.inputs['y']-self.pred)
    # Restructure loss as a function of 'd', survival
    if sys.version_info.major > 2:
      loss = tf.multiply(1-self.inputs['d'], survival_loss) + tf.multiply(self.inputs['d'], non_survival_loss)
    else:
      loss = tf.mul(1-self.inputs['d'], survival_loss) + tf.mul(self.inputs['d'], non_survival_loss)
    self.loss = tf.reduce_mean(loss)
    self.op = model_utils.optimization_fn(opt, alpha)

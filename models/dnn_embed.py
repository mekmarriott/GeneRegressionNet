import sys
from models import model_utils
import tensorflow as tf
from models.model import TFModel

class TFDNNEmbeddingModel(TFModel):
  def __init__(self, embed_shape, alpha=0.001, layer_dims=[], loss='normal', combiner=None, opt='sgd', output_sz=1, activation=None, dropout=False):
    # Inherit all self attributes
    TFModel.__init__(self, embed_shape[1], alpha=alpha)
    del self.inputs['x']
    self.inputs['x-indices'] = tf.placeholder(tf.int64, [None, 1]) # the number of dimensions for the patient index
    self.inputs['x-shape'] = tf.placeholder(tf.int64, [1]) # the number of indices
    self.inputs['x-values'] = tf.placeholder(tf.int64, [None]) # the number of values correpsonding to x-indices
    self.inputs['d'] = tf.placeholder(tf.float32, [None, 1])
    self.inputs['embed'] = tf.placeholder(tf.float32, embed_shape)
    self.inputs['y'] = tf.placeholder(tf.float32, [None, output_sz]) 
    if dropout:
      self.inputs['keep_prob'] = tf.placeholder(tf.float32, [1])

    layers = [embed_shape[1]] + layer_dims + [output_sz]
    self.W = {}
    self.b = {}
    
    # Create sparse input
    sparse_input = tf.SparseTensor(indices=self.inputs['x-indices'], values=self.inputs['x-values'], dense_shape=self.inputs['x-shape'])
    
    # Look up embeddings from one hot encodings
    self.embed_input = tf.nn.embedding_lookup_sparse(self.inputs['embed'], sparse_input, None, combiner=combiner)
    self.layer_inputs = {
        0: self.embed_input
    }
    # Fill in layers
    for i in range(len(layers) - 1):
      self.W[i] = tf.Variable(tf.random_normal([layers[i], layers[i+1]]))
      self.b[i] = tf.Variable(tf.random_normal([layers[i+1]]))
      self.layer_inputs[i+1] = tf.add(tf.matmul(self.layer_inputs[i], self.W[i]), self.b[i])
      self.layer_inputs[i+1] = model_utils.activation_fn(activation)(self.layer_inputs[i+1])
      if dropout and i < len(layers) - 2:
        self.layer_inputs[i+1] = tf.layers.dropout(self.layer_inputs[i+1], rate=self.inputs['keep_prob'])

    self.pred = self.layer_inputs[len(self.layer_inputs)-1]
    if loss == 'normal':
      # Survival loss - if the person is alive (1-d = 1) AND the prediction is greater than the label, apply no loss. Otherwise apply L2 loss.
      survival_loss = tf.square(tf.maximum(self.inputs['y']-self.pred, 0))
      # Non-survival loss - if the person is dead (d = 1) apply L2 loss
      non_survival_loss = tf.square(self.inputs['y']-self.pred)
      if sys.version_info.major > 2:
        loss = tf.multiply(1-self.inputs['d'], survival_loss) + tf.multiply(self.inputs['d'], non_survival_loss)
      else:
        loss = tf.mul(1-self.inputs['d'], survival_loss) + tf.mul(self.inputs['d'], non_survival_loss)
      self.loss = tf.reduce_mean(loss)
    elif loss == 'softmax':
      self.loss = tf.losses.softmax_cross_entropy(self.inputs['y'], self.pred)
    self.op = model_utils.optimization_fn(opt, alpha)

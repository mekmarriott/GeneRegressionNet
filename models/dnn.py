import model
import model_utils
import tensorflow as tf

class TFDNNModel(TFModel):
  def __init__(self, num_features, alpha=0.001, layer_dims=[], opt='sgd', activation=None):
    # Inherit all self attributes
    TFModel.__init__(self, num_features, alpha=alpha)
    self.inputs['d'] = tf.placeholder(tf.float32, [None, 1])
    
    layers = [num_features] + layer_dims + [1]
    self.W = {}
    self.b = {}
    self.layer_inputs = {
        0: self.inputs['x']
    }
    
    # Fill in layers
    for i in range(len(layers) - 1):
      self.W[i] = tf.Variable(tf.random_normal([layers[i], layers[i+1]]))
      self.b[i] = tf.Variable(tf.random_normal([layers[i+1]]))
      self.layer_inputs[i+1] = tf.add(tf.matmul(self.layer_inputs[i], self.W[i]), self.b[i])
      self.layer_inputs[i+1] = model_utils.activation_fn(self.layer_inputs[i+1])
    
    self.pred = list(self.layer_inputs.values())[-1]
    # Survival loss - if the person is alive (1-d = 1) AND the prediction is greater than the label, apply no loss. Otherwise apply L2 loss.
    survival_loss = tf.square(tf.maximum(self.inputs['y']-self.pred, 0))
    # Non-survival loss - if the person is dead (d = 1) apply L2 loss
    non_survival_loss = tf.square(self.inputs['y']-self.pred)

    self.loss = tf.reduce_mean(tf.multiply(1-self.inputs['d'], survival_loss) + tf.multiply(self.inputs['d'], non_survival_loss))
    self.op = model_utils.optimization_fn(opt, alpha)

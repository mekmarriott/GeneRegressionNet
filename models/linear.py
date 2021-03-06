import tensorflow as tf
import sys
from models.model import TFModel

class TFLinearModel(TFModel):
  
  def __init__(self, num_features, alpha=0.001, regularization=0.0):
    # Inherit all self attributes
    TFModel.__init__(self, num_features, alpha=alpha)
    self.inputs['d'] = tf.placeholder(tf.float32, [None, 1])
    # Survival loss - if the person is alive (1-d = 1) AND the prediction is greater than the label, apply no loss. Otherwise apply L2 loss.
    survival_loss = tf.square(tf.maximum(self.inputs['y']-self.pred, 0))
    # Non-survival loss - if the person is dead (d = 1) apply L2 loss
    non_survival_loss = tf.square(self.inputs['y']-self.pred)
    # Restructure loss as a function of 'd', survival
    # if sys.version_info.major > 2:
    loss = tf.multiply(1-self.inputs['d'], survival_loss) + tf.multiply(self.inputs['d'], non_survival_loss)
    # else:
    #   loss = tf.mul(1-self.inputs['d'], survival_loss) + tf.mul(self.inputs['d'], non_survival_loss)
    if regularization > 0:      # Loss function using L2 Regularization
      loss += regularization * tf.nn.l2_loss(self.W)
    self.loss = tf.reduce_mean(loss)
    if regularization > 0:
      for w in self.W:
        self.loss += tf.reduce_mean(tf.square(w))
      for b in self.b:
        self.loss += tf.reduce_mean(tf.square(b))
      

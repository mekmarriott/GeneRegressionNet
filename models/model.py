import tensorflow as tf
import sys

class TFModel():
  def __init__(self, num_features, alpha=0.1, regularization=0.0):
    self.inputs = {
        'x': tf.placeholder(tf.float32, [None, num_features]),
        'y': tf.placeholder(tf.float32, [None, 1])
    }
    self.W = tf.Variable(tf.zeros([num_features, 1]))
    self.b = tf.Variable(tf.zeros([1]))
    self.pred = tf.add(tf.matmul(self.inputs['x'], self.W), self.b)
    loss = tf.square(self.inputs['y']-self.pred)
    if regularization > 0:      # Loss function using L2 Regularization
      loss += regularization * tf.nn.l2_loss(self.W)
    self.loss = tf.reduce_mean(loss)
    self.op = tf.train.GradientDescentOptimizer(alpha)
    
  def initialize(self):
    self.train_op = self.op.minimize(self.loss)
    self.sess = tf.Session()
    # if sys.version_info.major > 2:
    self.sess.run(tf.global_variables_initializer())
    # else:
    #   self.sess.run(tf.initialize_all_variables())
    
  def translate_feed(self, input_map):
    feed = {}
    for key in self.inputs:
      assert key in input_map, "Input key %s not found in given feed" % key
      feed[self.inputs[key]] = input_map[key]
    return feed
    
  def train(self, feed, num_iterations=1000, debug=False):
    _feed = self.translate_feed(feed)
    for step in range(num_iterations):
      _, intermediate_loss = self.sess.run([self.train_op, self.loss], feed_dict=_feed)
      if debug and step%100==0:
        print("Step: %d: %f" % (step, intermediate_loss))
  
  def test(self, feed):
    _feed = self.translate_feed(feed)
    return self.sess.run(self.loss, feed_dict=_feed)
  
  def get_params(self, feed):
    _feed = self.translate_feed(feed)
    return self.sess.run([self.W, self.b], feed_dict=_feed)

  def predict(self, feed):
    _feed = self.translate_feed(feed)
    return self.sess.run(self.pred, feed_dict=_feed)
  

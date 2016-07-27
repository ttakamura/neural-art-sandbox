import util

class Network(object):
  def __init__(self, sess, graph_path):
    self.sess = sess
    self.input_tensor = util.load_network(sess, graph_path)
    self.setup()

class InceptionV3(Network):
  alpha = [0, 0, 0, 1, 1]
  beta  = [1, 1, 1, 1, 1]

  def setup(self):
    graph = tf.get_default_graph()
    self.feature_tensors = [
      graph.get_tensor_by_name('pool_3/_reshape:0')
    ]

  def y(self):
    return self.feature_tensors

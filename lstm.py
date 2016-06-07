import tensorflow as tf

def _variable_on_cpu(name, shape, initializer)
  """Helper to create a Variable stored on CPU memory."""
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var



class lstm_cell():
    def __init__(self, batch_size, input_dim, cell_dim):
      # Parameters:
      # Input gate: input, previous output, and bias.

      self.ix = tf.Variable(tf.truncated_normal([input_dim, cell_dim], -0.1, 0.1))
      self.ih = tf.Variable(tf.truncated_normal([cell_dim, cell_dim], -0.1, 0.1))
      self.ic = tf.Variable(tf.truncated_normal([cell_dim, cell_dim], -0.1, 0.1))
      self.ib = tf.Variable(tf.zeros([1, cell_dim]))
      # Forget gate: input, previous output, and bias.
      self.fx = tf.Variable(tf.truncated_normal([input_dim, cell_dim], -0.1, 0.1))
      self.fh = tf.Variable(tf.truncated_normal([cell_dim, cell_dim], -0.1, 0.1))
      self.fb = tf.Variable(tf.zeros([1, cell_dim]))
      # Memory cell: input, state and bias.                             
      self.cx = tf.Variable(tf.truncated_normal([input_dim, cell_dim], -0.1, 0.1))
      self.ch = tf.Variable(tf.truncated_normal([cell_dim, cell_dim], -0.1, 0.1))
      self.cb = tf.Variable(tf.zeros([1, cell_dim]))
      # Output gate: input, previous output, and bias.
      self.ox = tf.Variable(tf.truncated_normal([input_dim, cell_dim], -0.1, 0.1))
      self.oh = tf.Variable(tf.truncated_normal([cell_dim, cell_dim], -0.1, 0.1))
      self.ob = tf.Variable(tf.zeros([1, cell_dim]))
      # Variables saving state across unrollings.
      self.saved_output = tf.Variable(tf.zeros([batch_size, cell_dim]), trainable=False)
      self.cell = tf.Variable(tf.zeros([batch_size, cell_dim]), trainable=False)
      self.saved_state = tf.Variable(tf.zeros([batch_size, cell_dim]), trainable=False)
      # Classifier weights and biases.
      self.w = tf.Variable(tf.truncated_normal([cell_dim, input_dim], -0.1, 0.1))
      self.b = tf.Variable(tf.zeros([input_dim]))
      
# Definition of the cell computation.
def lstm_encode(cell, i, state):
  """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
  Note that in this formulation, we omit the various connections between the
  previous state and the gates."""
  # input_gate = tf.sigmoid(tf.matmul(i, self.ix) + tf.matmul(o, self.im) + self.ib)
  # forget_gate = tf.sigmoid(tf.matmul(i, self.fx) + tf.matmul(o, self.fm) + self.fb)
  # update = tf.matmul(i, self.cx) + tf.matmul(o, self.cm) + self.cb
  # state = forget_gate * state + input_gate * tf.tanh(update)
  # output_gate = tf.sigmoid(tf.matmul(i, self.ox) + tf.matmul(o, self.om) + self.ob)
  # return output_gate * tf.tanh(state), state
  input_gate = tf.sigmoid(tf.matmul(cell.ix, i) + tf.matmul(cell.ih, state) + \
				tf.matmul(cell.ic, cell.cell + cell.ib))
  forget_gate = tf.sigmoid(tf.matmul(cell.fx, i) + tf.matmul(cell.fh, state) + \
  				tf.matmul(cell.fc,self.cell) + cell.fb)
  cell.cell = tf.matmul(forget_gate, cell.cell) + tf.tanh(tf.matmul(cell.cx,i) + \
  				tf.matmul(cell.ch,state) + cell.cb)
  output_gate = tf.sigmoid(tf.matmul(cell.ox, i) + tf.matmal(cell.oh,state) + \
  				tf.matmal(cell.oc,cell.cell) + cell.ob)
  output = tf.matmul(output_gate, tf.tanh(cell.cell))
  return output, cell.cell

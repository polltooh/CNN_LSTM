import tensorflow as tf

def _variable_on_cpu(name, shape, initializer = tf.contrib.layers.xavier_initializer_conv2d()):
  """Helper to create a Variable stored on CPU memory."""
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var

def _conv2d(x, w, strides = [1,1,1,1]):
    return tf.nn.conv2d(x, w,strides=strides, padding = 'SAME')

class con_lstm_cell():
	def __init__(self, batch_size, input_h, input_w, input_c, ksize, cell_c):
		""" 
		Convolutional lstm

		Args:
			batch_size: the batch size
			input_h: the input tensor height
			input_w: the input tensor width
			input_c: the input tensor channel
			ksize: kernal size of the filter
			cell_c: cell channel
		"""
		# Parameters:
		# Input gate: input, previous output, and bias.
		self.ix = _variable_on_cpu("ix", [ksize,ksize,input_c,cell_c])
		self.ih = _variable_on_cpu("ih", [ksize,ksize,cell_c,cell_c])
		self.ic = _variable_on_cpu("ic", [ksize,ksize,cell_c,cell_c])
		self.ib = _variable_on_cpu("ib", [cell_c], tf.constant_initializer(0.0))

		# Forget gate: input, previous output, and bias.
		self.fx = _variable_on_cpu("fx", [ksize, ksize,input_c,cell_c])
		self.fh = _variable_on_cpu("fh", [ksize, ksize,cell_c,cell_c])
		self.fc = _variable_on_cpu("fc", [ksize, ksize, cell_c, cell_c])
		self.fb = _variable_on_cpu("fb", [cell_c], tf.constant_initializer(0.0))

		# Memory cell: input, state and bias.                             
		self.cx = _variable_on_cpu("cx", [ksize, ksize, input_c, cell_c])
		self.ch = _variable_on_cpu("ch", [ksize, ksize, cell_c, cell_c])
		self.cb = _variable_on_cpu("cb", [cell_c], tf.constant_initializer(0.0))

		# Output gate: input, previous output, and bias.
		self.ox = _variable_on_cpu("ox", [ksize, ksize, input_c, cell_c])
		self.oh = _variable_on_cpu("oh", [ksize, ksize, cell_c, cell_c])
		self.oc = _variable_on_cpu("oc", [ksize, ksize, cell_c, cell_c])
		self.ob = _variable_on_cpu("ob", [cell_c], tf.constant_initializer(0.0))

		# memory cell
		self.cell = tf.Variable(tf.zeros([batch_size, input_w, input_h, cell_c]), trainable=False)

		#initial statte
		self.zero_state = self.get_zero_state(batch_size, input_h, input_w, cell_c)

	def __call__(self, i, state):
		""" Convolutional LSTM

		Args:
			i: input
			state:
		"""
		input_gate = tf.sigmoid(_conv2d(i, self.ix) + _conv2d(state, self.ih) + \
					_conv2d(self.cell, self.ic) + self.ib)

		forget_gate = tf.sigmoid(_conv2d(i, self.fx) + _conv2d(state, self.fh) + \
					_conv2d(self.cell,self.fc) + self.fb)

		self.cell = tf.mul(forget_gate, self.cell) + tf.tanh(_conv2d(i, self.cx) + \
					_conv2d(state, self.ch) + self.cb)

		output_gate = tf.sigmoid(_conv2d(i, self.ox) + _conv2d(state, self.oh) + \
					_conv2d(self.cell, self.oc) + self.ob)

		output = tf.mul(output_gate, tf.tanh(self.cell))

		return output, state

	def get_zero_state(self, batch_size, input_h, input_w, cell_c):
		return tf.zeros((batch_size, input_h, input_w, cell_c), dtype = tf.float32)

def clstm_encode(cell, inputs, state = None, scope = None):
	""" Convolutional LSTM

	Args:
		cell: 
		inputs: inputs
		state:
	"""
	outputs = []
	if state == None:
		state = cell.zero_state

	for time, input_ in enumerate(inputs):
		if time > 0: tf.get_variable_scope().reuse_variables()
		call_cell = lambda:cell(input_, state)
		output, state = call_cell()
		outputs.append(output)

	return outputs, state

def clstm_decode(decoder_inputs, initial_state, cell, loop_function = None, scope = None):
	""" Convolutional LSTM decoding

	Args:
		decoder_inputs
		initial_state
		cell
		loop_function
		scope

	"""
	with variable_scope.variable_scope(scope or "clstm_decoder"):
		for i, inp in enumerate(decoder_inputs):
			if loop_function is not None and prev is not None:
				with variable_scope.variable_scope("loop_function", reuse=True):
					inp = loop_function(prev, i)
			if i > 0:
				variable_scope.get_variable_scope().reuse_variables()
			output, state = clstm_encode(cell, inp, state)
			outputs.append(output)
			if loop_function is not None:
				prev = output
	return outputs, state

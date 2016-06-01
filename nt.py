import tensorflow as tf

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var

def _variable_with_weight_decay(name, shape, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a xavier initialization.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  var = _variable_on_cpu(name, shape, tf.contrib.layers.xavier_initializer())
  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def add_leaky_relu(hl_tensor, leaky_param):
    return tf.maximum(hl_tensor, tf.mul(leaky_param, hl_tensor))

def inference(feature, feature_dim, label_dim, keep_prob = 1.0):
	feature_drop = tf.nn.dropout(feature,keep_prob)	
	with tf.variable_scope("fc") as scope:
		weights = _variable_with_weight_decay("weight", [feature_dim, label_dim], 0.001)
		biases = _variable_on_cpu("biases", [label_dim], tf.constant_initializer(0.0))
		fc = tf.nn.bias_add(tf.matmul(feature_drop, weights), biases)
	return fc

def inference2(feature, feature_dim, label_dim, keep_prob = 1.0):
	feature_drop = tf.nn.dropout(feature,keep_prob)	
	with tf.variable_scope("fc") as scope:
		weights = _variable_with_weight_decay("weight", [feature_dim, label_dim], 0.001)
		biases = _variable_on_cpu("biases", [label_dim], tf.constant_initializer(0.0))
		fc = tf.nn.bias_add(tf.matmul(feature_drop, weights), biases)

	with tf.variable_scope("fc2") as scope:
		weights = _variable_with_weight_decay("weight", [label_dim, label_dim], 0.001)
		biases = _variable_on_cpu("biases", [label_dim], tf.constant_initializer(0.0))
		fc2 = tf.nn.bias_add(tf.matmul(fc, weights), biases)
	with tf.variable_scope("fc3") as scope:
		weights = _variable_with_weight_decay("weight", [label_dim, label_dim], 0.001)
		biases = _variable_on_cpu("biases", [label_dim], tf.constant_initializer(0.0))
		fc3 = tf.nn.bias_add(tf.matmul(fc2, weights), biases)
	with tf.variable_scope("fc4") as scope:
		weights = _variable_with_weight_decay("weight", [label_dim, label_dim], 0.001)
		biases = _variable_on_cpu("biases", [label_dim], tf.constant_initializer(0.0))
		fc4 = tf.nn.bias_add(tf.matmul(fc3, weights), biases)
	return fc4

def train_op(learning_rate):
	with tf.name_scope("train_op"):
		# optimizer = tf.train.AdamOptimizer(learning_rate, epsilon = 1.0)
		optimizer = tf.train.RMSPropOptimizer(learning_rate,
			decay=0.9, momentum=0.0, 
			epsilon=1e-10, use_locking=False, 
			name='RMSProp')
	return optimizer

def train_op2(learning_rate):
	with tf.name_scope("train_op"):
		optimizer = tf.train.AdamOptimizer(learning_rate, epsilon = 10.0)
	return optimizer

def training(loss, learning_rate, global_step):
	with tf.name_scope("train"):
		# optimizer = tf.train.AdamOptimizer(learning_rate, epsilon = 1.0)
		optimizer = tf.train.AdamOptimizer(learning_rate, epsilon = 10.0)
		train_option = optimizer.minimize(loss, global_step = global_step)
	return train_option


def evaluation(infer):
	with tf.name_scope("eval"):
		infer_dense = tf.argmax(infer,1)
	return infer_dense

def loss(infer, labels):
	with tf.name_scope("train_cross_entropy"):
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(infer,labels,name='xentropy')
		loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
		tf.add_to_collection('losses', loss)

	return tf.add_n(tf.get_collection('losses'), name = 'total_loss')

def loss2(reg_res, labels):
	with tf.name_scope("l2_loss"):
		l2_norm_loss = tf.reduce_mean(tf.square(reg_res - labels))
		tf.add_to_collection('losses', l2_norm_loss)
		return tf.add_n(tf.get_collection('losses'), name = 'total_loss')

def test_loss(infer, labels):
	with tf.name_scope("test_cross_entropy"):
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(infer,labels,name='xentropy')
		loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

	return loss


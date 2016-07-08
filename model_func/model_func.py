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

def _variable_with_weight_decay(name, shape, wd = 0.0):
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
	# print("change var")
	# var = tf.Variable(tf.truncated_normal(shape, mean= 0.0, stddev = 1.0), name = name)
	if wd != 0.0:
		weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
		tf.add_to_collection('losses', weight_decay)
	return var

def _conv2d(x, w, strides = [1,1,1,1]):
    return tf.nn.conv2d(x, w,strides=strides, padding = 'SAME')

def _conv3d(x, w, strides = [1,1,1,1,1]):
    return tf.nn.conv3d(x, w,strides=strides, padding = 'SAME')
	
def add_leaky_relu(hl_tensor, leaky_param):
    return tf.maximum(hl_tensor, tf.mul(leaky_param, hl_tensor))

def _dconv2d(x, w, b, output_shape, strides = [1,1,1,1]):
    return tf.nn.bias_add(tf.nn.conv2d_transpose(x, w, output_shape, strides), b)

def _unpooling(x, output_size):
    # NEAREST_NEIGHBOR resize
    return tf.image.resize_images(x, output_size[1], output_size[2],1)

def _add_leaky_relu(hl_tensor, leaky_param):
    return tf.maximum(hl_tensor, tf.mul(leaky_param, hl_tensor))

def _max_pool(x, ksize, strides, name):
	pool = tf.nn.max_pool(x, ksize=ksize, strides= strides,
		padding='VALID', name = name)
	return pool

def _max_pool3(x, ksize, strides, name):
	pool = tf.nn.max_pool3d(x, ksize=ksize, strides= strides,
		padding='VALID', name = name)
	return pool

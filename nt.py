import tensorflow as tf

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory."""
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd):
  var = _variable_on_cpu(name, shape,
                         tf.truncated_normal_initializer(stddev=stddev))
  if wd:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def _conv2d(x, w, b, strides = [1,1,1,1]):
    return tf.nn.bias_add(tf.nn.conv2d(x,w,strides=strides, padding = 'SAME'), b)

def _dconv2d(x, w, b, output_shape, strides = [1,1,1,1]):
    return tf.nn.bias_add(tf.nn.conv2d_transpose(x, w, output_shape, strides), b)

def _unpooling(x, output_size):
    # NEAREST_NEIGHBOR resize
    return tf.image.resize_images(x, output_size[1], output_size[2],1)

def _add_leaky_relu(hl_tensor, leaky_param):
    return tf.maximum(hl_tensor, tf.mul(leaky_param, hl_tensor))


def inference1(data):
    batch_num = data.get_shape()[0]
    
    with tf.variable_scope('conv1') as scope:
      weights = _variable_with_weight_decay('weights', shape=[5, 5, 3, 16],
                                           stddev=1e-4, wd=0.0)
      biases = _variable_on_cpu('biases', [16], tf.constant_initializer(0.0))
      h_conv1 = _conv2d(data, weights, biases, [1,1,1,1])
      pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 6, 6, 1], strides=[1, 6, 6, 1],
                           padding='SAME', name='pool1')
      
    with tf.variable_scope('deconv1') as scope:

      unpool1 = _unpooling(pool1, data.get_shape())
      weights = _variable_with_weight_decay('weights', shape=[5, 5, 3, 16],
                                           stddev=1e-4, wd=0.0)
      biases = _variable_on_cpu('biases', [3], tf.constant_initializer(0.0))
      output_shape = tf.pack(data.get_shape().as_list())
      h_dconv1 = _dconv2d(unpool1, weights, biases, output_shape)
      
    return h_dconv1

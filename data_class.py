import tensorflow as tf

class DataClass():
	def __init__(self, data_format):
		self.data_format = data_format
		self.decode_class = None

class BINClass():
	def __init__(self):
		self.decode_fun = tf.decode_raw	
		self.dtype = None
		self.shape = [0]

	def decode(self, filename):
		bin_file = tf.read_file(filename)
		bin_tensor = tf.decode_raw(bin_file, self.dtype)
		bin_tensor = tf.to_float(bin_tensor)
		bin_tensor = tf.reshape(bin_tensor, shape)
		return bin_tensor	

class ImageClass():
	def __init__(self, shape, channels = None, ratio = None, name = None):
		self.channels = channels
		self.ratio = ratio
		self.name = name
		self.shape = shape
		self.decode_fun = None

	def decode(self, filename):
		image_tensor = tf.read_file(filename)
		image_tensor = self.decode_fun(image_tensor, channels = self.channels, ratio = self.ratio)
		image_tensor = tf.image.convert_image_dtype(image_tensor, tf.float32)
		image_tensor = tf.image.resize_images(image_tensor, self.shape[0], self.shape[1])
		return image_tensor
		
class JPGClass(ImageClass):
	def __init__(self, shape, channels = None, ratio = None, name = None):
		ImageClass.__init__(self, shape, channels, ratio, name)
		self.decode_fun = tf.image.decode_jpeg
		
class PNGClass(ImageClass):
	def __init__(self, shape, channels = None, ratio = None, name = None):
		ImageClass.__init__(self, shape, channels, ratio, name)
		self.decode_fun = tf.image.decode_png


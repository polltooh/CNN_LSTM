import tensorflow as tf
# import PIL
from PIL import Image
import numpy as np
import cv2
from PIL import Image
import os 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



def read_image(image_name, feature_row, feature_col):
    image_bytes = tf.read_file(image_name)
    image_tensor = tf.image.decode_jpeg(image_bytes, channels = 3)
    image_tensor = tf.image.convert_image_dtype(image_tensor, tf.float32)
    image_tensor = tf.image.resize_images(image_tensor, feature_row, feature_col)
    return image_tensor

def display_image(image_v):
    """accept 3D or 4D numpy array. if the input is 4D, it will use the first one"""
    display_image_v = image_v
    if image_v.ndim == 4:
        display_image_v = image_v[0]
    display_image_v[:,:,[2,0]] = display_image_v[:,:,[0,2]]
    cv2.imshow("image", display_image_v)
    cv2.waitKey(100)

def save_image(image_v, loss):
	"""accept 3D or 4D numpy array. if the input is 4D, it will use the first one"""
	save_image_v = image_v
	if save_image_v.ndim == 4:
		save_image_v = save_image_v[0]
	save_image_v[:,:,[2,0]] = save_image_v[:,:,[0,2]]
	save_image_v *= 255
	# filename = "loss_%f.jpg" % (loss)
	filename = "loss_%f.npy" % (loss)
	np.save(filename, save_image_v)
	return
	# cv2.imwrite(filename, save_image_v)
	# cv2.imwrite("aaa.jpg", I)

def define_graph_config(fraction):
	"""Define the GPU usage"""
	config_proto =  tf.ConfigProto()
	config_proto.gpu_options.per_process_gpu_memory_fraction = fraction
	config_proto.allow_soft_placement=False
	config_proto.log_device_placement=False
	return config_proto

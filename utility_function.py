import tensorflow as tf

def read_image(image_name, reverse_channel, feature_row, feature_col):
    image_bytes = tf.read_file(image_name)
    image_tensor = tf.image.decode_jpeg(image_bytes, channels = 3)
    image_tensor = tf.image.convert_image_dtype(image_tensor, tf.float32)
    image_tensor = tf.image.resize_images(image_tensor, feature_row, feature_col)
    if (reverse_channel):
        dim = tf.constant([False, False, True], dtype = tf.bool)
        image_tensor = tf.reverse(image_tensor, dim)
        image_tensor = image_tensor - [104.0/255, 117.0/255, 124.0/255]
    return (image_tensor) * 255

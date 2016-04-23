import tensorflow as tf

def read_image(image_name, feature_row, feature_col):
    image_bytes = tf.read_file(image_name)
    image_tensor = tf.image.decode_jpeg(image_bytes, channels = 3)
    image_tensor = tf.image.resize_images(image_tensor, feature_row, feature_col)
    return image_tensor

import sys
sys.path.insert(0, '../')

import tensor_data
import data_class
import tensorflow as tf

TRAIN_TXT = 'test_list_image.txt'

image_class = data_class.DataClass(tf.constant([], tf.string))
image_class.decode_class = data_class.JPGClass([299, 299], 3)

label_class = data_class.DataClass(tf.constant([], tf.int32))
data_list = list()
data_list.append(image_class)
data_list.append(label_class)

file_queue = tensor_data.file_queue(TRAIN_TXT, True)
batch_tensor_list = tensor_data.file_queue_to_batch_data(file_queue,data_list, True,2)

sess = tf.Session()

init_op = tf.initialize_all_variables()
sess.run(init_op)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord = coord, sess = sess)

batch_tensor= sess.run(batch_tensor_list)
print(batch_tensor)

import sys
import cv2
sys.path.insert(0, '../')

import tensor_data
import data_class
import tensorflow as tf

TRAIN_TXT = 'test_list_image.txt'
train_log_dir = "logs"
image_class = data_class.DataClass(tf.constant([], tf.string))
image_class.decode_class = data_class.JPGClass([227, 227], 3, 29)

label_class = data_class.DataClass(tf.constant([], tf.int32))
data_list = list()
data_list.append(image_class)
data_list.append(label_class)

batch_size = 2
is_train = True
file_queue = tensor_data.file_queue(TRAIN_TXT, is_train)
batch_tensor_list = tensor_data.file_queue_to_batch_data(file_queue,data_list, is_train, batch_size)

merged_train_sum = tf.merge_all_summaries()
sess = tf.Session()

writer_sum = tf.train.SummaryWriter(train_log_dir,sess.graph)
init_op = tf.initialize_all_variables()
sess.run(init_op)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord = coord, sess = sess)

run_v = sess.run(batch_tensor_list + [merged_train_sum])
cv2.imshow("img", img)
cv2.waitKey(0)
writer_sum.add_summary(run_v[-1])

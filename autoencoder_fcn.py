import tensorflow as tf
import utility_function as uf
import nt
import numpy as np
import os
import cv2

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_log_dir','logs','''directory wherer to write event logs''')
tf.app.flags.DEFINE_string('batch_size',1,'''batch size''')
tf.app.flags.DEFINE_integer('max_training_iter', 1000000,
        '''the max number of training iteration''')
tf.app.flags.DEFINE_float('init_learning_rate',0.1,
        '''initial learning rate''')
tf.app.flags.DEFINE_string('model_dir', 'model_logs','''directory where to save the model''')

IMG_ROW = 256
IMG_COL = 256
NUM_THREAD = 5

def batching(input_tensor, batch_size):
    return tf.train.batch([input_tensor], batch_size=batch_size, num_threads = NUM_THREAD,
            capacity = 100)

def train():
    image_name = tf.constant("lily.jpg", tf.string)
    image = uf.read_image(image_name, IMG_ROW, IMG_COL)
    batch_image = batching(image, FLAGS.batch_size)

    image_batch_ph = tf.placeholder(tf.float32, (FLAGS.batch_size, IMG_ROW, IMG_COL, 3), \
            name = "batch_images")

    label_batch_ph = tf.placeholder(tf.float32, (FLAGS.batch_size, IMG_ROW, IMG_COL, 3), \
            name = "batch_labels")

    global_step = tf.Variable(0, name = 'global_step', trainable = False)

    infer = nt.inference1(image_batch_ph)
    loss = nt.loss1(infer, label_batch_ph)
    train_op = nt.training1(loss,FLAGS.init_learning_rate, global_step)

    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)

    merged_sum =tf.merge_all_summaries()

    writer_sum = tf.train.SummaryWriter(FLAGS.train_log_dir)

    sess = tf.Session()
    
    init_op = tf.initialize_all_variables()
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord = coord, sess = sess)
    for i in xrange(FLAGS.max_training_iter):
        image_v = sess.run(batch_image)
        infer_v, loss_v, _= sess.run([infer, loss, train_op], 
                feed_dict = {image_batch_ph: image_v,label_batch_ph: image_v})
        if (i % 100 == 0):
            merged_sum_v = sess.run(merged_sum)
            writer_sum.add_summary(merged_sum_v, i)
            print("i: %d loss: %f" % (i,loss_v))
        if (i!= 0 and i % 1000 == 0):
            uf.display_image(np.hstack((image_v, infer_v)))
            # uf.save_image(np.hstack((image_v, infer_v)), loss_v)

    coord.request_stop()
    coord.join(threads)

    print("finished")

def main(argv = None):
    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)
    if not os.path.exists(FLAGS.train_log_dir):
        os.makedirs(FLAGS.train_log_dir)
    train()

if __name__ == '__main__':
    tf.app.run()

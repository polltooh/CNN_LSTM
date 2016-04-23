import tensorflow as tf
import utility_function as uf
import nt
import os
import cv2

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_log_dir','logs','''directory wherer to write event logs''')
tf.app.flags.DEFINE_integer('max_training_iter', 100000,
        '''the max number of training iteration''')
tf.app.flags.DEFINE_float('init_learning_rate',0.001,
        '''initial learning rate''')
tf.app.flags.DEFINE_string('model_dir', 'model_logs','''directory where to save the model''')

def batching(input_tensor, batch_size):
    return tf.train.batch([input_tensor], batch_size=batch_size)

def train():
    image_name = tf.constant("lily.jpg", tf.string)
    image = uf.read_image(image_name, 256, 256)
    batch_image = batching(image, 1)

    image_batch_ph = tf.placeholder(tf.float32, (1, 256, 256, 3), \
            name = "batch_image")
    infer = nt.inference1(image_batch_ph)

    sess = tf.Session()
    
    init_op = tf.initialize_all_variables()
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord = coord, sess = sess)

    image_v = sess.run(batch_image)
    infer_v = sess.run(infer, feed_dict = {image_batch_ph: image_v})
    print(infer_v[0])

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

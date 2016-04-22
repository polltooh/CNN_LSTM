import tensorflow as tf
import utility_function as uf



FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_log_dir','logs','''directory wherer to write event logs''')
tf.app.flags.DEFINE_integer('max_training_iter', 100000,
        '''the max number of training iteration''')
tf.app.flags.DEFINE_float('init_learning_rate',0.001,
        '''initial learning rate''')
tf.app.flags.DEFINE_string('model_dir', 'model_logs','''directory where to save the model''')

def train():
    image_name = tf.constant("lily.jpg", tf.string)
    image = uf.read_image(image_name)

    sess = tf.Session()
    image_v = sess.run(image)


def main(argv = None):
    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)
    if not os.path.exists(FLAGS.train_log_dir):
        os.makedirs(FLAGS.train_log_dir)
    train()

if __name__ == '__main__':
    tf.app.run()

import tensorflow as tf
import utility_function as uf
import con_lstm as crnn

IMG_ROW = 256
IMG_COL = 256
BATCH_SIZE = 2
CELL_C = 10

def train():
	image_name = tf.constant("lily.jpg", tf.string)
	image1 = uf.read_image(image_name, IMG_ROW, IMG_COL)
	image1 = tf.expand_dims(image1, 0)
	image2 = uf.read_image(image_name, IMG_ROW, IMG_COL)
	image2 = tf.expand_dims(image2, 0)
	image = tf.concat(0, (image1, image2))

	clstm = crnn.con_lstm_cell(BATCH_SIZE, IMG_ROW, IMG_COL, 3, 3, CELL_C)
	input_ = tf.placeholder(tf.float32, (BATCH_SIZE, IMG_ROW, IMG_COL, 3))
	inputs = []
	inputs.append(input_)
	inputs.append(input_)
	
	outputs, state = crnn.clstm_encode(clstm, inputs)

	sess = tf.Session()

	init_op = tf.initialize_all_variables()
	sess.run(init_op)

	for i in xrange(100):
		image_v = sess.run(image)
		feed_data = dict()
		feed_data[inputs[0]] = image_v
		feed_data[inputs[1]] = image_v
		outputs_v = sess.run(outputs, feed_dict = feed_data)
		print(outputs_v)

def main(argv = None):
    # if not os.path.exists(FLAGS.model_dir):
    #     os.makedirs(FLAGS.model_dir)
    # if not os.path.exists(FLAGS.train_log_dir):
    #    os.makedirs(FLAGS.train_log_dir)
	train()

if __name__ == '__main__':
	tf.app.run()

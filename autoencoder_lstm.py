from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  
import tensorflow as tf

import os
import data_queue
import nt

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_log_dir','logs','''directory wherer to write event logs''')
tf.app.flags.DEFINE_string('batch_size',1,'''batch size''')
tf.app.flags.DEFINE_integer('max_training_iter', 1000,
        '''the max number of training iteration''')
tf.app.flags.DEFINE_float('init_learning_rate',0.1,
        '''initial learning rate''')
tf.app.flags.DEFINE_string('model_dir', 'model_logs','''directory where to save the model''')

INPUT_DIM = 64 * 64
LABEL_DIM = INPUT_DIM
# CELL_DIM = 64 * 64

# INPUT_DIM = 10
CELL_DIM = 100
CELL_LAYER = 3

BATCH_SIZE = 10
UNROLLING_NUM = 10

def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
	return tf.nn.seq2seq.embedding_attention_seq2seq(
	  encoder_inputs, decoder_inputs, cell,
	  num_encoder_symbols=source_vocab_size,
	  num_decoder_symbols=target_vocab_size,
	  embedding_size=size,
	  output_projection=output_projection,
	  feed_previous=do_decode)

def train():
	input_data_queue = data_queue.DATA_QUEUE()
	train_inputs = input_data_queue.get_next_batch(BATCH_SIZE)

	single_cell = tf.nn.rnn_cell.BasicLSTMCell(CELL_DIM)
	multi_cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * CELL_LAYER)

	inputs_ph = list()
	labels_ph = list()
	decodes_ph = list()
   	for _ in range(UNROLLING_NUM):
		inputs_ph.append(tf.placeholder(tf.float32,[BATCH_SIZE, INPUT_DIM], name = "input_ph"))
	 	labels_ph.append(tf.placeholder(tf.float32,[BATCH_SIZE, INPUT_DIM], name = "label_ph"))
		decodes_ph.append(tf.placeholder(tf.float32,[BATCH_SIZE, INPUT_DIM], name = "decodes_ph"))	

	cell_initial_state = multi_cell.zero_state(BATCH_SIZE, tf.float32)

	outputs, state = tf.nn.seq2seq.basic_rnn_seq2seq(inputs_ph, decodes_ph, multi_cell)

	con_cat_out = tf.concat(0, outputs)
	con_cat_label = tf.concat(0, labels_ph)
	infer = nt.inference(con_cat_out, CELL_DIM, LABEL_DIM)	
	loss = nt.loss2(infer, con_cat_label)

	global_step = tf.Variable(0, name = 'global_step', trainable = False)
	train_op = nt.training(loss, FLAGS.init_learning_rate, global_step = global_step)

	sess = tf.Session()

	init_op = tf.initialize_all_variables()
	sess.run(init_op)

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord = coord, sess = sess)

	for i in xrange(FLAGS.max_training_iter):
		feed_data = dict()
		for i in range(UNROLLING_NUM):
			input_v = input_data_queue.get_next_batch(BATCH_SIZE)
			feed_data[inputs_ph[i]] = input_v[i,:,0:INPUT_DIM]
			feed_data[labels_ph[i]] = input_v[i,:,0:INPUT_DIM]
			feed_data[decodes_ph[i]] = input_v[UNROLLING_NUM - i + 1,:,0:INPUT_DIM]
		_, loss_v = sess.run([train_op, loss], feed_dict = feed_data)
		print(loss_v)

def main(argv = None):
    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)
    if not os.path.exists(FLAGS.train_log_dir):
        os.makedirs(FLAGS.train_log_dir)
    train()

if __name__ == '__main__':
    tf.app.run()

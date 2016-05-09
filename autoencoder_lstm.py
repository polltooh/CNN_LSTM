from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  
import tensorflow as tf
import os

import lstm

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_log_dir','logs','''directory wherer to write event logs''')
tf.app.flags.DEFINE_string('batch_size',1,'''batch size''')
tf.app.flags.DEFINE_integer('max_training_iter', 1000000,
        '''the max number of training iteration''')
tf.app.flags.DEFINE_float('init_learning_rate',0.1,
        '''initial learning rate''')
tf.app.flags.DEFINE_string('model_dir', 'model_logs','''directory where to save the model''')

INPUT_DIM = 2
CELL_DIM = 2
BATCH_SIZE = 1


def batching(input_tensor, batch_size):
    return tf.train.batch([input_tensor], batch_size=batch_size)

def train():
    lstm_cell = lstm.lstm_cell(BATCH_SIZE, INPUT_DIM, CELL_DIM)
    train_inputs = np.zeros((6,3), dtype=np.float32)
    train_inputs[0,0] = 1
    train_inputs[1,1] = 1
    train_inputs[2,2] = 1
    
    train_inputs = np.zeros((6,3), dtype=np.float32)
    train_inputs[3,2] = 1
    train_inputs[4,1] = 1
    train_inputs[5,0] = 1

    output = lstm_cell.saved_output
    state = lstm_cell.saved_state
    
    input_ph = tf.placeholder(tf.float32,[BATCH_SIZE, INPUT_DIM])
    label_ph = tf.placeholder(tf.float32,[Batch_SIZE, INPUT_DIM])
    outputs = list()

    for i in range(6):
        output, state = lstm_cell.inference(train_ph, output, state)
        outputs.append(output)

    sess = tf.Session()
    
    init_op = tf.initialize_all_variables()
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord = coord, sess = sess)

    # for i in xrange(FLAGS.max_training_iter):
    for i in xrange(10):
        for train_input in train_inputs:
            train_input = train_input.reshape((1,2))
            output, state = lstm_cell.one_pass(train_input, output, state)
            print(output)

    coord.request_stop()
    coord.join(threads)

def main(argv = None):
    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)
    if not os.path.exists(FLAGS.train_log_dir):
        os.makedirs(FLAGS.train_log_dir)
    train()

if __name__ == '__main__':
    tf.app.run()

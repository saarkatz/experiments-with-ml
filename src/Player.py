import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn

class Player:
    def __init__(self, name, cols, rows):
        self.name = name
        # Input layer
        board = tf.placeholder(dtype=tf.float32, shape=[-1, cols, rows])
        # Dense layer
        board_flat = tf.reshape(board, [-1, cols * rows])
        dense = tf.layers.dense(inputs=board, units=1024, activation=tf.sigmoid)
        # Dropout
        dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == learn.ModeKeys.TRAIN)
        logits = tf.layers.dense(inputs=dropout, units=cols)

        loss = None
        train_op = None

        # Calculate loss for both TRAIN and EVAL modes
        if mode != learn.ModeKeys.INFER:
            # onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
            loss = tf.losses.softmax_cross_entropy(
                onehot_labels=onehot_labels, logits=logits)

        # Configure the Training Op (for TRAIN mode)
        if mode == learn.ModeKeys.TRAIN:
            train_op = tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=tf.contrib.framework.get_global_step(),
                learning_rate=0.001,
                optimizer="SGD")

    def next_turn(self, state):
        print(np.flipud(np.transpose(state)))
        input_action = int(input('Player %s: Enter your move:' % self.name))
        action = np.zeros((self.cols,))
        action[input_action] = 1
        return action
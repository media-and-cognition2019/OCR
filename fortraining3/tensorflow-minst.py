
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500
IMAG_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10
CONV1_DEEP = 32
CONV1_SIZE = 5
CONV2_DEEP = 64
CONV2_SIZE = 5
FC_SIZE = 512

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZATION_RAZE = 0.001
TRAINING_SEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "data"
MODLE_NAME = "model.ckpt"


def inference(input_tensor, train, regularizer):
    with tf.variable_scope('layer1-conv1'):
        weights1 = tf.get_variable(
            "weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
            initializer=tf.truncated_normal_initializer(0.1)
        )
        biases1 = tf.get_variable(
            "biases", [CONV1_DEEP],
            initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(
            input_tensor, weights1, strides=[1, 1, 1, 1], padding='SAME'
        )
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, biases1))
        pool1 = tf.nn.max_pool(
            relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME'
        )
    with tf.variable_scope('layer2-conv2'):
        weights2 = tf.get_variable(
            "weight",[CONV2_SIZE,CONV2_SIZE,CONV1_DEEP, CONV2_DEEP],
            initializer=tf.truncated_normal_initializer(0.1)
        )
        biases2 = tf.get_variable(
            "biases",[CONV2_DEEP],
            initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(
            pool1, weights2, strides=[1,1,1,1], padding='SAME'
        )
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, biases2))
        pool2 = tf.nn.max_pool(
            relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        pool_shape = pool2.get_shape().as_list()
        nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
        reshaped = tf.reshape(pool2, [pool_shape[0], nodes])
    with tf.variable_scope('layer3'):
        weights3 = tf.get_variable(
            "weight", [nodes,FC_SIZE],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        biases3 = tf.get_variable(
            "biases", [FC_SIZE],
            initializer=tf.constant_initializer(0.0)
        )
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(weights3))
        fc1=tf.nn.relu(tf.matmul(reshaped, weights3) + biases3)
        if train: fc1 = tf.nn.dropout(fc1, 0.5)
    with tf.variable_scope('layer4'):
        weights4 = tf.get_variable(
            "weight", [FC_SIZE, NUM_LABELS],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        biases4 = tf.get_variable(
            "biases", [NUM_LABELS],
            initializer=tf.constant_initializer(0.0)
        )
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(weights4))
        fc2 = tf.nn.tanh(tf.matmul(fc1, weights4) + biases4)
    return fc2


def train(mnist):
    x = tf.placeholder(tf.float32, [
        BATCH_SIZE,
        IMAG_SIZE,
        IMAG_SIZE,
        NUM_CHANNELS
    ], name='x-input')
    y_=tf.placeholder(tf.float32, [BATCH_SIZE, NUM_LABELS])
    regularizer1 = tf.contrib.layers.l2_regularizer(REGULARAZATION_RAZE)
    y = inference(x, True, regularizer1)
    global_step = tf.Variable(0, trainable=False)
    var_ave = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step
    )
    var_ave_op = var_ave.apply(
        tf.trainable_variables()
    )
    cross_en = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(y_, 1)
    )
    cross_en_mean = tf.reduce_mean(cross_en)
    loss = cross_en_mean+ tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY
    )
    train_step = tf.train.GradientDescentOptimizer(learning_rate)\
        .minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, var_ave_op]):
        train_op = tf.no_op(name='train')
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        for i in range(TRAINING_SEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            r_xs = np.reshape(xs, (BATCH_SIZE, IMAG_SIZE, IMAG_SIZE, NUM_CHANNELS))
            _, loss_value, step = sess.run(
            [train_op, loss, global_step], feed_dict={x: r_xs,y_:ys})
            if i % 10== 0:
                print("after %d steps, loss is %g"%(step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH,MODLE_NAME), global_step=global_step)


def main(argv=None):
    mnist = input_data.read_data_sets("tmp/data/", one_hot=True)
    train(mnist)


if __name__=='__main__':
    tf.app.run()

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
import tensorflow as tf


learning_rate_base=0.8
learning_rate_decay=0.99
training_iters = 200000
batch_size = 64
display_step = 20
moving_average_decay = 0.99
#
n_input = 784  #
n_classes = 10  #
dropout = 0.8  # Dropout

#
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)


#
def conv2d(name, l_input, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME'), b), name=name)


#
def max_pool(name, l_input, k):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)


#
def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)


#
def alex_net(_X, _weights, _biases, _dropout):
    #
    _X = tf.reshape(_X, shape=[-1, 28, 28, 1])

    #
    conv1 = conv2d('conv1', _X, _weights['wc1'], _biases['bc1'])
    #
    pool1 = max_pool('pool1', conv1, k=2)
    #
    norm1 = norm('norm1', pool1, lsize=4)
    # Dropout
    norm1 = tf.nn.dropout(norm1, _dropout)

    #
    conv2 = conv2d('conv2', norm1, _weights['wc2'], _biases['bc2'])
    #
    pool2 = max_pool('pool2', conv2, k=2)
    #
    norm2 = norm('norm2', pool2, lsize=4)
    # Dropout
    norm2 = tf.nn.dropout(norm2, _dropout)

    #
    conv3 = conv2d('conv3', norm2, _weights['wc3'], _biases['bc3'])
    conv4 = conv2d('conv4', conv3, _weights['wc4'], _biases['bc4'])
    conv5 = conv2d('conv5', conv4, _weights['wc5'], _biases['bc5'])
    pool5 = max_pool('pool5', conv5, k=2)
    #
    norm5 = norm('norm5', pool5, lsize=4)
    # Dropout
    norm5 = tf.nn.dropout(norm5, _dropout)

    #
    dense1 = tf.reshape(norm5, [-1, _weights['wd1'].get_shape().as_list()[0]])
    dense1 = tf.nn.relu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'], name='fc1')
    #
    dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2')  # Relu activation

    #
    out = tf.matmul(dense2, _weights['out']) + _biases['out']
    return out


#
weights = {
    'wc1': tf.Variable(tf.random_normal([11, 11, 1, 64])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 64, 192])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 192, 384])),
    'wc4': tf.Variable(tf.random_normal([3, 3, 384, 256])),
    'wc5': tf.Variable(tf.random_normal([3, 3, 256, 256])),
    'wd1': tf.Variable(tf.random_normal([4 * 4 * 256, 1024])),
    'wd2': tf.Variable(tf.random_normal([1024, 1024])),
    'out': tf.Variable(tf.random_normal([1024, 10]))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([64])),
    'bc2': tf.Variable(tf.random_normal([192])),
    'bc3': tf.Variable(tf.random_normal([384])),
    'bc4': tf.Variable(tf.random_normal([256])),
    'bc5': tf.Variable(tf.random_normal([256])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'bd2': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

#
pred = alex_net(x, weights, biases, keep_prob)


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
global_step = tf.Variable(0, trainable=False)
variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
variables_averages_op = variable_averages.apply(tf.trainable_variables())
learning_rate = tf.train.exponential_decay(learning_rate_base, global_step, mnist.train.num_examples/batch_size, learning_rate_decay)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)


with tf.control_dependencies([train_step, variables_averages_op]):
    train_op = tf.no_op(name='train')
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    step = 1
    validate_feed = {x: mnist.validation.images, y: mnist.validation.labels, keep_prob: 1.}
    test_feed = {x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.}
    for i in range(training_iters):
        if i % display_step == 0:
            validate_acc = sess.run(accuracy, feed_dict=validate_feed)
            print("After %d trainingstep(s), validation accuracy using average model is %g" % (i, validate_acc))
        xs, ys = mnist.train.next_batch(batch_size)
        sess.run(train_op, feed_dict={x: xs, y: ys, keep_prob: dropout})
    test_acc=sess.run(accuracy, feed_dict=test_feed)
    print("After %d training step(s),test accuracy using average model is %g" % (training_iters, test_acc))

            #acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            #
            #loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            #print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + "{:.6f}".format(
                #loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
       # step += 1
  #  print("Optimization Finished!")
    #
  #  print("Testing Accuracy:",
     #     sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.}))
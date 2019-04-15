import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784
OUTPUT_NODE = 10

LAYER1_NODE = 500  # 隐藏层的节点数（本结构只有一个隐藏层）

BATCH_SIZE = 100

LRARNING_RATE_BASE = 0.8
LRARNING_RATE_DECAY = 0.99  # 学习率的衰减率

REGULARIZATION_RATE = 0.0001  # 正则化项在损失函数中的系数

TRAINNING_STEPS = 3000  # 训练轮数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率
DROPOUT = 0.8

DISPLAYSTEP = 1000;

x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
y = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
keep_prob = tf.placeholder(tf.float32)


def conv2d(name, l_input, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME'), b), name=name)


def max_pool(name, l_input, k):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)


def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

def inference(_X, _weights, _biases, _dropout):
    _X = tf.reshape(_X, shape=[-1, 28, 28, 1])
    conv1 = conv2d('conv1', _X, _weights['wc1'], _biases['bc1'])
    pool1 = max_pool('pool1', conv1, k=2)
    norm1 = norm('norm1', pool1, lsize=4)
    norm1 = tf.nn.dropout(norm1, _dropout)

    conv2 = conv2d('conv2', norm1, _weights['wc2'], _biases['bc2'])
    pool2 = max_pool('pool2', conv2, k=2)
    norm2 = norm('norm2', pool2, lsize=4)
    norm2 = tf.nn.dropout(norm2, _dropout)

    conv3 = conv2d('conv3', norm2, _weights['wc3'], _biases['bc3'])

    conv4 = conv2d('conv4', conv3, _weights['wc4'], _biases['bc4'])

    conv5 = conv2d('conv5', conv4, _weights['wc5'], _biases['bc5'])
    pool5 = max_pool('pool5', conv5, k=2)
    norm5 = norm('norm5', pool5, lsize=4)
    norm5 = tf.nn.dropout(norm5, _dropout)

    dense1 = tf.reshape(norm5, [-1, _weights['wd1'].get_shape().as_list()[0]])
    dense1 = tf.nn.relu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'], name='fc1')

    dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2')  # Relu activation

    out = tf.matmul(dense2, _weights['out']) + _biases['out']
    return out


# 训练模型过程

def train(mnist):


    # 生成隐藏层的参数
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
        'out': tf.Variable(tf.random_normal([OUTPUT_NODE]))
    }


    # 计算当前参数前向传播的结果。这里给出的用于计算平均滑动的类为 NONE,所以函数不会使用参数的滑动平均值
    y_ = inference(x, weights, biases, DROPOUT)
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    average_y = inference(x, weights, biases, keep_prob)

    # 计算交叉熵                (需要将原文的语句做以下修改)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_)
    # 计算当前batch中所有样例的交叉熵的平均值
    cost = tf.reduce_mean(cross_entropy)

    # 计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 计算模型的正则化损失
    regularization = regularizer(weights['wd1']) + regularizer(weights['wd2']) + regularizer(weights['out'])

    # 总损失等于交叉熵损失和正则化损失的和
    loss = cost + regularization
    # 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(LRARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE,
                                               LRARNING_RATE_DECAY)  # 学习率衰减速度
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    # 优化损失函数，这里的损失函数包括交叉熵损失函数和L2损失函数

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 一次进行反向传播参数更新网络中的 参数和参数的滑动平均值
    # train_op=tf.group(train_step,variable_averages_op)     #此句话下面的两句话是等价操作
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 初始化会话并开始训练过程
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  # 初始化所有的变量
        validate_feed = {x: mnist.validation.images,
                         y_: mnist.validation.labels}
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        # 迭代的训练神经网络
        for i in range(TRAINNING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d trainingstep(s), validation accuracy using average model is %g" % (i, validate_acc))

            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After %d training step(s),test accuracy using average model is %g" % (TRAINNING_STEPS, test_acc))



def main(argv=None):
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    train(mnist)


if __name__ == "__main__":
    tf.app.run()

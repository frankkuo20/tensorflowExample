import os

import tensorflow as tf
import math
import matplotlib.pyplot as plt
import numpy as np


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

IMG_SIZE = 48  # 图像大小
LABEL_CNT = 7  # 标签类别的数量


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')


class NnObj:
    def __init__(self):
        x = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, 1])
        y_ = tf.placeholder(tf.float32, [None, LABEL_CNT])  # right answer

        img = tf.cast(x, tf.float32)
        img -= tf.reduce_mean(img, axis=0)
        mean, var = tf.nn.moments(img, axes=[0])
        img /= var
        x = img

        # img = tf.cast(x, tf.float32)
        # img -= tf.reduce_mean(img, axis=0)
        # img = img*100/255
        # x = img

        # one
        W_conv = weight_variable([5, 5, 1, 32])
        b_conv = bias_variable([32])
        h_conv = tf.nn.relu(conv2d(x, W_conv) + b_conv)
        h_pool = max_pool_2x2(h_conv)
        h_pool = tf.nn.local_response_normalization(h_pool)

        # two
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)
        h_pool2 = tf.nn.local_response_normalization(h_pool2)

        # three
        W_conv3 = weight_variable([5, 5, 64, 128])
        b_conv3 = bias_variable([128])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
        h_pool3 = max_pool_2x2(h_conv3)
        h_pool3 = tf.nn.local_response_normalization(h_pool3)

        # final 128/2/2 = 32 48/2/2  /2
        # W_fc = weight_variable([32 * 32 * 64, 1024])
        # W_fc = weight_variable([12 * 12 * 64, 1024])
        # full connect
        W_fc = weight_variable([6 * 6 * 128, 128])
        b_fc = bias_variable([128])

        h_pool2_flat = tf.reshape(h_pool3, [-1, 6 * 6 * 128])
        h_fc = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc) + b_fc)

        keep_prob = tf.placeholder(tf.float32)
        h_fc_drop = tf.nn.dropout(h_fc, keep_prob)

        # full connect2
        W_fc_2 = weight_variable([128, 256])
        b_fc_2 = bias_variable([256])

        h_pool3_flat = tf.reshape(h_fc_drop, [-1, 128])
        h_fc = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc_2) + b_fc_2)

        h_fc_drop = tf.nn.dropout(h_fc, keep_prob)

        W_fc2 = weight_variable([256, LABEL_CNT])
        b_fc2 = bias_variable([LABEL_CNT])

        y = tf.nn.softmax(tf.matmul(h_fc_drop, W_fc2) + b_fc2)

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
        # 10/20 new
        beta = 0.01
        regularizers = tf.nn.l2_loss(W_conv) + tf.nn.l2_loss(W_conv2) + tf.nn.l2_loss(W_conv3)
        cross_entropy = cross_entropy + beta * regularizers
        cross_entropy = tf.reduce_mean(cross_entropy)

        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        init = tf.global_variables_initializer()

        self.x = x
        self.y = y
        self.y_ = y_
        self.init = init
        self.keep_prob = keep_prob
        self.train_step = train_step
        self.accuracy = accuracy

        self.h_conv = h_conv
        self.h_conv2 = h_conv2
        self.W_conv = W_conv

    def getPredList(self, image):
        im_raw = image.tobytes()
        img = tf.decode_raw(im_raw, tf.uint8)
        img = tf.reshape(img, [IMG_SIZE, IMG_SIZE, 1])
        image = img

        x = self.x
        y = self.y
        keep_prob = self.keep_prob

        sess = tf.Session()

        saver = tf.train.Saver()
        tf.reset_default_graph()

        # saver.restore(sess, './cnn_train/graph.ckpt-{}'.format(max_step))

        cnn_train = os.path.join('G:\\project\\faceEmotion\\emotion\\cnn', 'cnn_train')

        saver.restore(sess, tf.train.latest_checkpoint(cnn_train))

        image = sess.run([image])
        result = sess.run(y, feed_dict={x: image, keep_prob: 1.0})
        # plt.imshow(np.reshape(image, [48, 48]), interpolation="nearest", cmap="gray")
        # plt.show()
        # print(result.argmax())
        self.sess = sess
        return result

    def getPredNum(self, image):
        result = self.getPredList(image)
        resultNum = result.argmax()

        return resultNum


        # def plotNNFilter(self, units):
        #     filters = units.shape[3]
        #     plt.figure(1, figsize=(20, 20))
        #     n_columns = 6
        #     n_rows = math.ceil(filters / n_columns) + 1
        #     for i in range(filters):
        #         plt.subplot(n_rows, n_columns, i + 1)
        #         plt.title('Filter ' + str(i))
        #         plt.imshow(units[0, :, :, i], interpolation="nearest", cmap="gray")
        #     plt.show()
        #
        # def getActivations(self, layer, stimuli):
        #     x = self.x
        #     units = sess.run(layer, feed_dict={x: stimuli, keep_prob: 1.0})
        #     plotNNFilter(units)
        #
        #     imageToUse = test_imgs
        #
        # plt.imshow(np.reshape(imageToUse, [128, 128]), interpolation="nearest", cmap="gray")
        # plt.show()
        # getActivations(h_conv, imageToUse)
        # getActivations(h_conv2, imageToUse)
        # getActivations(W_conv, imageToUse)

if __name__ == '__main__':
    pass

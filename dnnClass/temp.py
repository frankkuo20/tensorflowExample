import csv
import tensorflow as tf
from dnnClass.dnn import NnObj

if __name__ == '__main__':
    # file = open('resultFind.csv', 'r')
    save_file = open('result.csv', 'a+')
    save_file.write('\n2556789,0.2758374810218811')

    save_file.close()

    # nnObj = NnObj()
    # init = nnObj.init
    # x = nnObj.x
    # y = nnObj.y
    # y_ = nnObj.y_
    # train_step = nnObj.train_step
    # accuracy = nnObj.accuracy
    # keep_prob = nnObj.keep_prob
    #
    # col = 'discover'
    # col2 = ''
    # col3 = 'online - playlist'
    #
    # col = tf.string_to_hash_bucket_fast(col, 9, name=None)
    # col2 = tf.string_to_hash_bucket_fast(col2, 21, name=None)
    # col3 = tf.string_to_hash_bucket_fast(col3, 13, name=None)
    #
    # features = tf.stack([col, col2, col3])
    # max_step = 6600
    # saver = tf.train.Saver()
    # with tf.Session() as sess:
    #
    #     saver.restore(sess, './models/graph.ckpt-{}'.format(max_step))
    #     features2 = sess.run([features])
    #     yy = sess.run(y, feed_dict={x: features2, keep_prob: 1.0})
    #
    #     resultNum = yy[0][1]
    #     print('{}, {}'.format(15, resultNum))


# 0.2758374810218811
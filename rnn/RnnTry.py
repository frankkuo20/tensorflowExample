import pandas as pd
import tensorflow as tf

tf.set_random_seed(1)  # set random seed

# hyperparameters
lr = 0.001  # learning rate
n_inputs = 3  # MNIST data input (img shape: 28*28)
n_steps = 1  # time steps
n_hidden_units = 10  # neurons in hidden layer
n_classes = 2  # MNIST classes (0-9 digits)
batch_size = 100

# 对 weights biases 初始值的定义
weights = {
    # shape (28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # shape (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    # shape (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # shape (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}


def RNN(X, weights, biases):
    # 原始的 X 是 3 维数据, 我们需要把它变成 2 维数据才能使用 weights 的矩阵乘法
    # X ==> (128 batches * 28 steps, 28 inputs)
    X = tf.reshape(X, [-1, n_inputs])

    # X_in = W*X + b
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (128 batches, 28 steps, 128 hidden) 换回3维
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # 使用 basic LSTM Cell.
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)  # 初始化全零 state
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)
    results = tf.matmul(final_state[1], weights['out']) + biases['out']
    return results


def dnn(X, weights, biases):

    X = tf.reshape(X, [-1, n_inputs])
    layer_1 = tf.matmul(X, weights['in']) + biases['in']
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.matmul(layer_1, weights['out']) + biases['out']
    layer_2 = tf.nn.relu(layer_2)
    return layer_2


def input_pipeline(features, labels, batch_size):
    # features = tf.reshape(features, [-1, 10])

    min_after_dequeue = 1000
    num_threads = 4
    capacity = min_after_dequeue + (num_threads + 3) * batch_size
    feature_batch, label_batch = tf.train.shuffle_batch(
        [features, labels], batch_size=batch_size, capacity=capacity, num_threads=num_threads,
        min_after_dequeue=min_after_dequeue)

    return feature_batch, label_batch


if __name__ == '__main__':
    filename_queue = tf.train.string_input_producer(['../csv/rnn/train.csv'])
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)

    record_defaults = [[0] for i in range(28 * 28 + 1)]
    record_defaults = [[''], [''], [''], [0]]
    # 解析 CSV 資料
    col1, col2, col3, col4 = tf.decode_csv(
        value, record_defaults=record_defaults)

    source_system_tab_dic = ['explore', 'my library',
                             'search', 'discover',
                             'None', 'radio', 'listen with',
                             'notification', 'settings']
    source_screen_name_dict = ['Explore', 'Local playlist more',
                               'None', 'My library', 'Online playlist more',
                               'Album more', 'Discover Feature', 'Unknown',
                               'Discover Chart', 'Radio',
                               'Artist more', 'Search', 'Others profile more', 'Search Trends',
                               'Discover Genre', 'My library_Search', 'Search Home',
                               'Discover New',
                               'Self profile more', 'Concert', 'Payment']

    source_type_dict = ['online-playlist', 'local-playlist', 'local-library',
                        'top-hits-for-artist', 'album', 'None',
                        'song-based-playlist', 'radio', 'song', 'listen-with',
                        'artist', 'topic-article-playlist', 'my-daily-playlist']

    # col1 = tf.one_hot(col1, 9, on_value=source_system_tab_dic)
    # col1 = col1[0]
    col1 = tf.string_to_hash_bucket_fast(col1, 9)
    col2 = tf.string_to_hash_bucket_fast(col2, 21)
    col3 = tf.string_to_hash_bucket_fast(col3, 13)
    # 把 CSV 資料的前四欄打包成一個 tensor
    features = tf.stack([col1, col2, col3])

    col4 = tf.one_hot(col4, 2)

    labels = col4

    feature_batch, label_batch = input_pipeline(features, labels, batch_size)

    # x y placeholder
    # x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    x = tf.placeholder(tf.float32, [None, n_inputs])
    y = tf.placeholder(tf.float32, [None, n_classes])

    # pred = RNN(x, weights, biases)
    pred = dnn(x, weights, biases)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    train_op = tf.train.AdamOptimizer(lr).minimize(cost)

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()  # 用来保存模型的

    step = 0
    max_step = 1000
    printStep = 5
    save_step = 50

    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            # 获取训练数据成功，并且没有到达最大训练次数
            while not coord.should_stop() and step < max_step:
                step += 1

                features, labels = sess.run([feature_batch, label_batch])

                features = features.reshape([batch_size, n_inputs])
                # features = features.reshape([batch_size, n_steps, n_inputs])
                sess.run([train_op], feed_dict={
                    x: features,
                    y: labels,
                })
                if step % printStep == 0:  # step
                    acc = sess.run(accuracy, feed_dict={
                        x: features,
                        y: labels
                    })
                    print('%s accuracy is %.2f' % (step, acc))
                if step % save_step == 0:
                    # 保存当前模型
                    save_path = saver.save(sess, './train_model/graph.ckpt', global_step=step)
                    print("save graph to %s" % save_path)
        except tf.errors.OutOfRangeError as e:
            print("reach epoch limit")
            print(e)
        except Exception as e:
            print('eee')
            print(e)
        finally:
            coord.request_stop()

        coord.join(threads)
        save_path = saver.save(sess, './train_model/graph.ckpt', global_step=step)

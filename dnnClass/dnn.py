import tensorflow as tf

INPUT_SIZE = 4
OUTPUT_SIZE = 1


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


class NnObj:
    def __init__(self):
        x = tf.placeholder(tf.float32, [None, INPUT_SIZE])
        y_ = tf.placeholder(tf.float32, [None, OUTPUT_SIZE])  # right answer

        networks = [INPUT_SIZE, 100, 100, 100]
        mat_1 = x

        keep_prob = tf.placeholder(tf.float32)
        regularizers = 0

        front = networks[0]
        for i in range(1, len(networks)):
            back = networks[i]
            weight = weight_variable([front, back])
            bias = bias_variable([back])
            mat_1 = tf.nn.relu(tf.matmul(mat_1, weight) + bias)
            mat_1 = tf.nn.dropout(mat_1, keep_prob)
            front = back
            regularizers += tf.nn.l2_loss(weight)

        back = networks[-1]
        w_4 = weight_variable([back, OUTPUT_SIZE])
        b_4 = bias_variable([OUTPUT_SIZE])

        y = tf.matmul(mat_1, w_4) + b_4
        # y = tf.nn.softmax(y)

        cross_entropy = tf.losses.mean_squared_error(labels=y_, predictions=y)
        # cross_entropy = tf.reduce_mean(tf.square(y_ - y))
        # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
        # 10/20 new
        beta = 0.01

        # cross_entropy = cross_entropy + beta * regularizers
        # cross_entropy = tf.reduce_mean(cross_entropy)

        # train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)

        # correct_prediction = tf.equal(y, y_)
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracy = ''
        init = tf.global_variables_initializer()

        self.x = x
        self.y = y
        self.y_ = y_
        self.init = init
        self.cross_entropy = cross_entropy
        self.keep_prob = keep_prob
        self.train_step = train_step
        self.accuracy = accuracy


def input_pipeline(features, labels, batch_size):
    # features = tf.reshape(features, [-1, 10])

    min_after_dequeue = 1000
    num_threads = 4
    capacity = min_after_dequeue + (num_threads + 3) * batch_size
    feature_batch, label_batch = tf.train.shuffle_batch(
        [features, labels], batch_size=batch_size, capacity=capacity, num_threads=num_threads,
        min_after_dequeue=min_after_dequeue)

    return feature_batch, label_batch


def getFeaturesLabels(csvPath):
    filename_queue = tf.train.string_input_producer([csvPath])

    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)

    record_defaults = [
        [''], [''],
        [''], [''], [''], [0],
        [0], [0], [''], [''], [''], [''],
        [''], [''], [''], [''], [''], [''],
    ]

    user, song, col, col2, col3, label, \
    city, bd, gender, registered_via, registration_init_time, expiration_date, \
    song_length, genre_ids, artist_name, composer, lyricist, language \
        = tf.decode_csv(value, record_defaults=record_defaults)

    # song_length, genre_ids, artist_name, composer, lyricist, language
    # city,bd,gender,registered_via,registration_init_time,expiration_date
    # labels = tf.one_hot(label, 2)
    labels = [label]

    user = tf.string_to_hash_bucket_fast(user, 100, name=None)
    song = tf.string_to_hash_bucket_fast(song, 100, name=None)

    col = tf.string_to_hash_bucket_fast(col, 9, name=None)
    col2 = tf.string_to_hash_bucket_fast(col2, 21, name=None)
    col3 = tf.string_to_hash_bucket_fast(col3, 13, name=None)

    city = tf.to_int64(city)
    bd = tf.to_int64(bd)
    
    features = tf.stack(
        [
            col, col2, col3,
            city, bd
        ]
    )
    return features, labels


# TRAIN_CSV = '../csv/dnn/train.csv'
# TRAIN_CSV = '../csv/train.csv'
TRAIN_CSV = '../csv/train_train.csv'
TEST_CSV = '../csv/train_test.csv'

TRAIN_CSV = '../csv/train_train2.csv'
TEST_CSV = '../csv/train_test2.csv'

if __name__ == '__main__':
    nnObj = NnObj()
    init = nnObj.init
    x = nnObj.x
    y = nnObj.y
    y_ = nnObj.y_
    train_step = nnObj.train_step
    accuracy = nnObj.accuracy
    keep_prob = nnObj.keep_prob
    cross_entropy = nnObj.cross_entropy

    features, labels = getFeaturesLabels(TRAIN_CSV)

    feature_batch, label_batch = input_pipeline(features, labels, 500)

    features_test, labels_test = getFeaturesLabels(TEST_CSV)
    feature_batch_test, label_batch_test = input_pipeline(features_test, labels_test, 500)

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    step = 0
    max_step = 10000
    printStep = 5
    save_step = 50

    with tf.Session() as sess:
        sess.run(init)

        # saver.restore(sess, tf.train.latest_checkpoint('models'))

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            # 获取训练数据成功，并且没有到达最大训练次数
            while not coord.should_stop() and step < max_step:
                step += 1

                features, labels = sess.run([feature_batch, label_batch])

                sess.run(train_step, feed_dict={x: features, y_: labels, keep_prob: 0.5})
                # if step % printStep == 0:  # step
                # 输出当前batch的精度。预测时keep的取值均为1
                # acc = sess.run(accuracy, feed_dict={x: features, y_: labels, keep_prob: 1.0})
                # print('%s accuracy is %.2f' % (step, acc))

                cross = sess.run(cross_entropy, feed_dict={x: features, y_: labels, keep_prob: 1.0})
                print('%s cross_entropy is %.2f' % (step, cross))

                if step % save_step == 0:
                    # 保存当前模型
                    save_path = saver.save(sess, './models/graph.ckpt', global_step=step)
                    print("save graph to %s" % save_path)

                    features_test, labels_test = sess.run([feature_batch_test, label_batch_test])
                    cross = sess.run(cross_entropy, feed_dict={x: features_test, y_: labels_test, keep_prob: 1.0})
                    print('......test%s cross_entropy is %.2f' % (step, cross))

                    # acc = sess.run(accuracy, feed_dict={x: features_test, y_: labels_test, keep_prob: 1.0})
                    # print('......test%s accuracy is %.2f' % (step, acc))
        except tf.errors.OutOfRangeError as e:
            print("reach epoch limit")
            print(e)
        except Exception as e:
            print('eee')
            print(e)
        finally:
            coord.request_stop()

        coord.join(threads)
        save_path = saver.save(sess, './models/graph.ckpt', global_step=step)

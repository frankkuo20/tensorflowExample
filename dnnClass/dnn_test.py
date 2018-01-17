import tensorflow as tf
from dnnClass.dnn import NnObj, input_pipeline

TEST_CSV = '../csv/test.csv'

if __name__ == '__main__':
    nnObj = NnObj()
    init = nnObj.init
    x = nnObj.x
    y = nnObj.y
    y_ = nnObj.y_
    train_step = nnObj.train_step
    accuracy = nnObj.accuracy
    keep_prob = nnObj.keep_prob

    filename_queue = tf.train.string_input_producer([TEST_CSV])

    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)

    record_defaults = [
        [''], [''], [''],
        [''], [''], ['']]

    _, user, song, col, col2, col3 = tf.decode_csv(value, record_defaults=record_defaults)

    user = tf.string_to_hash_bucket_fast(user, 100, name=None)
    song = tf.string_to_hash_bucket_fast(song, 100, name=None)

    col = tf.string_to_hash_bucket_fast(col, 9, name=None)
    col2 = tf.string_to_hash_bucket_fast(col2, 21, name=None)
    col3 = tf.string_to_hash_bucket_fast(col3, 13, name=None)

    features = tf.stack([col, col2, col3])

    saver = tf.train.Saver()

    num = 0
    max_step = 10000
    testNum = 2556790
    csvFile = open('result.csv', 'w')
    csvFile.write('id,target')

    with tf.Session() as sess:
        print('./models/graph.ckpt-{}'.format(max_step))
        saver.restore(sess, './models/graph.ckpt-{}'.format(max_step))

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            # 获取训练数据成功，并且没有到达最大训练次数
            while not coord.should_stop():
                features2 = sess.run([features])
                # 预测阶段，keep取值均为1
                yy = sess.run(y, feed_dict={x: features2, keep_prob: 1.0})

                # resultNum = yy[0][1]
                resultNum = yy[0][0]
                print('{}, {}'.format(num, resultNum))

                csvFile.write('\n{},{}'.format(num, str(resultNum)))
                num += 1
                if num == testNum:
                    break
        except tf.errors.OutOfRangeError as e:
            print("reach epoch limit")
            print(e)
        except Exception as e:
            print('eee')
            print(e)
        finally:
            coord.request_stop()
        coord.join(threads)

    csvFile.close()

import itertools
import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
_CSV_COLUMNS = [
    'msno',
    'song_id',
    'source_system_tab', 'source_screen_name',
    'source_type', 'target'
]

CATEGORICAL_COLUMNS = [
    "source_system_tab",
    "source_screen_name",
    "source_type"
]

CONTINUOUS_COLUMNS = ["msno", "song_id"]

SURVIVED_COLUMN = "target"


def build_estimator(model_dir):
    """Build an estimator."""
    # Categorical columns
    # sex = tf.contrib.layers.sparse_column_with_keys(column_name="Sex",
    #                                                 keys=["female", "male"])
    # embarked = tf.contrib.layers.sparse_column_with_keys(column_name="Embarked",
    #                                                      keys=["C",
    #                                                            "S",
    #                                                            "Q"])
    with tf.device('/device:GPU:0'):
        source_system_tab = tf.contrib.layers.sparse_column_with_keys(
            column_name="source_system_tab",
            keys=['explore', 'my library',
                  'search', 'discover',
                  'None', 'radio', 'listen with',
                  'notification', 'settings']
        )

        source_screen_name = tf.contrib.layers.sparse_column_with_keys(
            column_name="source_screen_name",
            keys=['Explore', 'Local playlist more',
                  'None', 'My library', 'Online playlist more',
                  'Album more', 'Discover Feature', 'Unknown',
                  'Discover Chart', 'Radio',
                  'Artist more', 'Search', 'Others profile more', 'Search Trends',
                  'Discover Genre', 'My library_Search', 'Search Home',
                  'Discover New',
                  'Self profile more', 'Concert', 'Payment']
        )

        source_type = tf.contrib.layers.sparse_column_with_keys(
            column_name="source_type",
            keys=['online-playlist', 'local-playlist', 'local-library',
                  'top-hits-for-artist', 'album', 'None',
                  'song-based-playlist', 'radio', 'song', 'listen-with',
                  'artist', 'topic-article-playlist', 'my-daily-playlist']
        )

        msno = tf.contrib.layers.sparse_column_with_hash_bucket(
            "msno", hash_bucket_size=30755)
        song_id = tf.contrib.layers.sparse_column_with_hash_bucket(
            "song_id", hash_bucket_size=1000)

        # Continuous columns
        # age = tf.contrib.layers.real_valued_column("Age")
        # passenger_id = tf.contrib.layers.real_valued_column("PassengerId")
        # sib_sp = tf.contrib.layers.real_valued_column("SibSp")
        # parch = tf.contrib.layers.real_valued_column("Parch")
        # fare = tf.contrib.layers.real_valued_column("Fare")
        # p_class = tf.contrib.layers.real_valued_column("Pclass")

        # Transformations.
        # age_buckets = tf.contrib.layers.bucketized_column(age,
        #                                                   boundaries=[
        #                                                       5, 18, 25, 30, 35, 40,
        #                                                       45, 50, 55, 65
        #                                                   ])

        # Wide columns and deep columns.
        # wide_columns = [sex, embarked, cabin, name, age_buckets,
        #                 tf.contrib.layers.crossed_column(
        #                     [age_buckets, sex],
        #                     hash_bucket_size=int(1e6)),
        #                 tf.contrib.layers.crossed_column([embarked, name],
        #                                                  hash_bucket_size=int(1e4))]

        wide_columns = [
            source_system_tab, source_screen_name, source_type,
            tf.contrib.layers.crossed_column(
                [source_system_tab, source_screen_name],
                hash_bucket_size=int(1e6)),
            tf.contrib.layers.crossed_column(
                [source_screen_name, source_type],
                hash_bucket_size=int(1e6)),
            tf.contrib.layers.crossed_column(
                [source_system_tab, source_type],
                hash_bucket_size=int(1e6)),
        ]

        deep_columns = [
            tf.contrib.layers.embedding_column(source_system_tab, dimension=8),
            tf.contrib.layers.embedding_column(source_screen_name, dimension=8),
            tf.contrib.layers.embedding_column(source_type, dimension=8),
            tf.contrib.layers.embedding_column(msno, dimension=8),
            tf.contrib.layers.embedding_column(song_id, dimension=8),
        ]

    return tf.contrib.learn.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[10, 5]
    )



def input_fn(df, train=False):
    """Input builder function."""
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.
    continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.
    # categorical_cols = {k: tf.constant(df[k].values) for k in CATEGORICAL_COLUMNS}

    categorical_cols = {
        k: tf.SparseTensor(
            indices=[[i, 0] for i in range(df[k].size)],
            values=df[k].values,
            dense_shape=[df[k].size, 1]
        ) for k in CATEGORICAL_COLUMNS}

    # Merges the two dictionaries into one.
    feature_cols = dict(continuous_cols)
    feature_cols.update(categorical_cols)

    # Converts the label column into a constant Tensor.
    if train:
        label = tf.constant(df[SURVIVED_COLUMN].values)
        # Returns the feature columns and the label.
        return feature_cols, label
    else:
        return feature_cols


def train_and_eval():
    """Train and evaluate the model."""
    df_train = pd.read_csv(
        tf.gfile.Open("./csv/train2.csv"),
        skipinitialspace=True)
    df_test = pd.read_csv(
        tf.gfile.Open("./csv/test2.csv"),
        skipinitialspace=True)

    # df_prediction = pd.read_csv(
    #     tf.gfile.Open("./csv/test22_5.csv"),
    #     skipinitialspace=True)

    model_dir = "./models"
    print("model directory = %s" % model_dir)

    m = build_estimator(model_dir)
    m.fit(input_fn=lambda: input_fn(df_train, True), steps=200)

    # print(m.predict(input_fn=lambda: input_fn(df_test)))
    results = m.evaluate(input_fn=lambda: input_fn(df_test, True), steps=1)
    for key in sorted(results):
        print("%s: %s" % (key, results[key]))



def test_result():
    testCsvs = [
        'test22_1',
        'test22_2',
        'test22_3',
        'test22_4',
        'test22_5',
    ]

    f = open('./csv/result.csv', 'w')
    f.write('id,target')
    for fileCount, testCsv in enumerate(testCsvs):
        fileName = "./csv/{}.csv".format(testCsv)
        df_prediction = pd.read_csv(
            tf.gfile.Open(fileName),
            skipinitialspace=True)

        model_dir = "./models"
        print("model directory = %s" % model_dir)

        m = build_estimator(model_dir)

        y = m.predict(input_fn=lambda: input_fn(df_prediction))

        predictions = list(y)

        for index, prediction in enumerate(predictions):
            index += 500000 * fileCount
            print(index)
            f.write('\n{},{}'.format(index, prediction))

            # print("Predictions: {}".format(str(predictions)))

    f.close()


def main(_):
    train_and_eval()
    # test_result()


if __name__ == "__main__":
    tf.app.run()

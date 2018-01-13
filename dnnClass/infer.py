import numpy as np
import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


class DnnObj:
    def __init__(self):
        pass


# Data sets
IRIS_TRAINING = "dnn/train.csv"

IRIS_TEST = "dnn/test.csv"

CATEGORICAL_COLUMNS = [
    "source_system_tab",
    "source_screen_name",
    "source_type"
]
SURVIVED_COLUMN = "target"


def input_fn(df, train=False):
    """Input builder function."""
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.

    # continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}

    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.
    # categorical_cols = {k: tf.constant(df[k].values) for k in CATEGORICAL_COLUMNS}

    categorical_cols = {
        k: tf.SparseTensor(
            indices=[[i, 0] for i in range(df[k].size)],
            values=df[k].values,
            dense_shape=[df[k].size, 1]
        ) for k in CATEGORICAL_COLUMNS}
    print(categorical_cols)
    continuous_cols = categorical_cols
    # Merges the two dictionaries into one.
    feature_cols = dict(continuous_cols)
    # feature_cols.update(categorical_cols)

    # Converts the label column into a constant Tensor.
    if train:
        label = tf.constant(df[SURVIVED_COLUMN].values)
        # Returns the feature columns and the label.
        return feature_cols, label
    else:
        return feature_cols


def main():
    # Load datasets.
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
    ]
    # Specify that all features have real-value data
    # feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]
    feature_columns = deep_columns

    # Build 3 layer DNN with 10, 20, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[5, 5],
                                            n_classes=2,
                                            model_dir="./models")

    df_train = pd.read_csv(
        tf.gfile.Open("../csv/dnn/train.csv"),
        skipinitialspace=True)

    # Define the training inputs
    # train_input_fn = tf.estimator.inputs.numpy_input_fn(
    #     x={"x": np.array(training_set.data)},
    #     y=np.array(training_set.target),
    #     num_epochs=None,
    #     shuffle=True)

    # Train model.
    classifier.train(input_fn=lambda: input_fn(df_train, True), steps=200)

    # Define the test inputs
    # test_input_fn = tf.estimator.inputs.numpy_input_fn(
    #     x={"x": np.array(test_set.data)},
    #     y=np.array(test_set.target),
    #     num_epochs=1,
    #     shuffle=False)

    # Evaluate accuracy.
    accuracy_score = classifier.evaluate(input_fn=lambda: input_fn(df_train, True))["accuracy"]

    print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

    # # Classify two new flower samples.
    # new_samples = np.array(
    #     [[6.4, 3.2, 4.5, 1.5],
    #      [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)
    # predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    #     x={"x": new_samples},
    #     num_epochs=1,
    #     shuffle=False)
    #
    # predictions = list(classifier.predict(input_fn=predict_input_fn))
    # predicted_classes = [p["classes"] for p in predictions]
    #
    # print(
    #     "New Samples, Class Predictions:    {}\n"
    #         .format(predicted_classes))


if __name__ == "__main__":
    main()

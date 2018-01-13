import shutil

import tensorflow as tf

_CSV_COLUMNS = [
    'msno', 'song_id', 'source_system_tab', 'source_screen_name',
    'source_type', 'target'
]

_CSV_COLUMN_DEFAULTS = [[0], [0], [0],
                        [0], [0], [0]]

_NUM_EXAMPLES = {
    'train': 7377418,
    'validation': 16281,
}

def input_fn(data_file, num_epochs, shuffle, batch_size):
    """Generate an input function for the Estimator."""
    assert tf.gfile.Exists(data_file), (
        '%s not found. Please make sure you have either run data_download.py or '
        'set both arguments --train_data and --test_data.' % data_file)

    def parse_csv(value):
        print('Parsing', data_file)
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('target')
        return features, labels

    # Extract lines from input files using the Dataset API.
    dataset = tf.contrib.data.TextLineDataset(data_file)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])

    dataset = dataset.map(parse_csv) # , num_parallel_calls=5

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels


def build_model_columns():
    """Builds a set of wide and deep feature columns."""
    # Continuous columns
    tf.feature_column.
    age = tf.feature_column.numeric_column('age')
    education_num = tf.feature_column.numeric_column('education_num')
    capital_gain = tf.feature_column.numeric_column('capital_gain')
    capital_loss = tf.feature_column.numeric_column('capital_loss')
    hours_per_week = tf.feature_column.numeric_column('hours_per_week')

    education = tf.feature_column.categorical_column_with_vocabulary_list(
        'education', [
            'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
            'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
            '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])

    marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
        'marital_status', [
            'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
            'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'])

    relationship = tf.feature_column.categorical_column_with_vocabulary_list(
        'relationship', [
            'Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried',
            'Other-relative'])

    workclass = tf.feature_column.categorical_column_with_vocabulary_list(
        'workclass', [
            'Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
            'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'])

    # To show an example of hashing:
    occupation = tf.feature_column.categorical_column_with_hash_bucket(
        'occupation', hash_bucket_size=1000)

    # Transformations.
    age_buckets = tf.feature_column.bucketized_column(
        age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

    # Wide columns and deep columns.
    base_columns = [
        education, marital_status, relationship, workclass, occupation,
        age_buckets,
    ]

    crossed_columns = [
        tf.feature_column.crossed_column(
            ['education', 'occupation'], hash_bucket_size=1000),
        tf.feature_column.crossed_column(
            [age_buckets, 'education', 'occupation'], hash_bucket_size=1000),
    ]

    wide_columns = base_columns + crossed_columns

    deep_columns = [
        age,
        education_num,
        capital_gain,
        capital_loss,
        hours_per_week,
        tf.feature_column.indicator_column(workclass),
        tf.feature_column.indicator_column(education),
        tf.feature_column.indicator_column(marital_status),
        tf.feature_column.indicator_column(relationship),
        # To show an example of embedding
        tf.feature_column.embedding_column(occupation, dimension=8),
    ]

    return wide_columns, deep_columns


def build_estimator(model_dir, model_type):
    """Build an estimator appropriate for the given model type."""
    wide_columns, deep_columns = build_model_columns()
    hidden_units = [100, 75, 50, 25]

    # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
    # trains faster than GPU for this model.
    run_config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(device_count={'GPU': 0}))

    if model_type == 'wide':
        return tf.estimator.LinearClassifier(
            model_dir=model_dir,
            feature_columns=wide_columns,
            config=run_config)
    elif model_type == 'deep':
        return tf.estimator.DNNClassifier(
            model_dir=model_dir,
            feature_columns=deep_columns,
            hidden_units=hidden_units,
            config=run_config)
    else:
        return tf.estimator.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=hidden_units,
            config=run_config)


batch_size = 40
epochs_per_eval = 2
model_dir = '/tmp/census_model'
model_type = 'wide_deep'
test_data = 'csv/test.csv'
train_data = 'csv/train.csv'
train_epochs = 40


def main(test):
    # Clean up the model directory if present
    shutil.rmtree(model_dir, ignore_errors=True)
    model = build_estimator(model_dir, model_type)

    # Train and evaluate the model every `FLAGS.epochs_per_eval` epochs.
    for n in range(train_epochs // epochs_per_eval):
        model.train(input_fn=lambda: input_fn(
            train_data, epochs_per_eval, True, batch_size))

        results = model.evaluate(input_fn=lambda: input_fn(
            test_data, 1, False, batch_size))

        # Display evaluation metrics
        print('Results at epoch', (n + 1) * epochs_per_eval)
        print('-' * 60)

        for key in sorted(results):
            print('%s: %s' % (key, results[key]))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    tf.app.run(main=main)

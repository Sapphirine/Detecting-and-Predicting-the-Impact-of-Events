
# coding: utf-8

# In[87]:


# !pip3 install --upgrade tensorflow


# In[4]:


import pandas as pd
import tensorflow as tf
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import numpy as np


# In[2]:



TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Sentosa', 'Versicolor', 'Virginica']

def maybe_download():
    train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)

    return train_path, test_path

def load_data(y_name='Species'):
    """Returns the iris dataset as (train_x, train_y), (test_x, test_y)."""
    train_path, test_path = maybe_download()

    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    train_x, train_y = train, train.pop(y_name)
    train_onehot = []
    for y in train_y:
      train_onehot.append(np.sum([np.eye(3, dtype=np.uint8)[y]], axis=0, dtype=np.uint8))
    
    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(y_name)
    test_onehot = []
    for y in test_y:
      test_onehot.append(np.sum([np.eye(3, dtype=np.uint8)[y]], axis=0, dtype=np.uint8))

    return (train_x, train_y), (test_x, test_y)


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()




def _parse_line(line):
    # Decode the line into its fields
    fields = tf.decode_csv(line, record_defaults=CSV_TYPES)

    # Pack the result into a dictionary
    features = dict(zip(CSV_COLUMN_NAMES, fields))

    # Separate the label from the features
    label = features.pop('Species')

    return features, label


def csv_input_fn(csv_path, batch_size):
    # Create a dataset containing the text lines.
    dataset = tf.data.TextLineDataset(csv_path).skip(1)

    # Parse each line.
    dataset = dataset.map(_parse_line)

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()


# In[5]:


(gtrain_x, gtrain_y), (gtest_x, gtest_y) = load_data()


# In[10]:


gtrain_x[~gtrain_x.applymap(np.isreal).all(1)]


# In[9]:


gtrain_x.to_dict('series')


# In[7]:


dict(gtrain_x)


# In[ ]:



batch_size = 100
train_steps = 2

def my_model(features, labels, mode, params):
    """DNN with three hidden layers, and dropout of 0.1 probability."""
    # Create three fully connected layers each layer having a dropout
    # probability of 0.1.
    net = tf.feature_column.input_layer(features, params['feature_columns'])
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    # Compute logits (1 per class).
    logits = tf.layers.dense(net, params['n_classes'], activation=tf.nn.sigmoid)

    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.sigmoid(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

      
    # Compute loss.
    #loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
#     onehot = []
#     for y in labels:
#       onehot.append()
#     fn_onehot = lambda y: np.sum([np.eye(params['n_classes'], dtype=np.uint8)[y]], axis=0, dtype=np.uint8) 
#     onehot = tf.map_fn(fn_onehot, labels, dtype=(tf.int64, tf.int64, tf.int64))
    loss = tf.losses.sigmoid_cross_entropy(tf.one_hot(labels, params['n_classes']), logits=logits)

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
#     metrics = {'accuracy': accuracy}
    
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

g_predictions = None

def main(argv):
#     args = parser.parse_args(argv[1:])

    # Fetch the data
    (train_x, train_y), (test_x, test_y) = load_data()

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.Estimator(
        model_fn=my_model,
        params={
            'feature_columns': my_feature_columns,
            # Two hidden layers of 10 nodes each.
            'hidden_units': [10, 10],
            # The model must choose between 3 classes.
            'n_classes': 3,
        })

    # Train the Model.
    classifier.train(
        input_fn=lambda:train_input_fn(train_x, train_y, batch_size),
        steps=train_steps)

    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda:eval_input_fn(test_x, test_y, batch_size))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Generate predictions from the model
    expected = ['Setosa', 'Versicolor', 'Virginica']
    predict_x = {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    }

    predictions = classifier.predict(
        input_fn=lambda:eval_input_fn(predict_x,
                                                labels=None,
                                                batch_size=batch_size))
    global g_predictions 
    g_predictions = predictions
    for pred_dict, expec in zip(predictions, expected):
        template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(SPECIES[class_id],
                              100 * probability, expec))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)


# In[59]:


if True:
  (train_x, train_y), (test_x, test_y) = load_data()


# In[60]:


train_y


# In[13]:


# [np.sum(x['probabilities']) for x in g_predictions]
iter_pred = g_predictions
[str(x)+str(np.sum(x['probabilities']))  for x in iter_pred]


# In[36]:


import numpy as np
np.array([1,15,6,19,5]).sum()


# In[49]:


g = [1,2]
l = np.sum([np.eye(3, dtype="uint8")[s] for s in g], axis=0)
l


# In[71]:


with tf.Session() as sess:
  print(sess.run(tf.losses.sigmoid_cross_entropy([[1.,0.,0.],[0.,1.,0.]], 
                                                 [[1.,1.,0.],[0.,0.,1.]])))


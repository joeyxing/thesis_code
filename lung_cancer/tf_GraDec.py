import numpy as np
import tensorflow as tf
import pickle
import os

PROJECT_ROOT = "/home/joey/Work/thesis_code"
IMAGE_SIZE = 100
IMAGE_CHANNEL = 3
pickle_file = os.path.join(PROJECT_ROOT, "lung_cancer_data.pickle")

with open(pickle_file, "rb") as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

# label -> one hot encoding
# data -> 2-D

def reformat(data, labels):
    data = data.reshape((-1, IMAGE_CHANNEL * IMAGE_SIZE * IMAGE_SIZE)).astype(np.float32)
    labels = (np.arange(2) == labels[:, None]).astype(np.float32)
    return data, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

# TensorFlow:
graph = tf.Graph()
with graph.as_default():
    tf_train_dataset = tf.constant(train_dataset)
    tf_train_labels = tf.constant(train_labels)
    tf_test_dataset = tf.constant(test_dataset)
    tf_valid_dataset = tf.constant(valid_dataset)
    weights = tf.Variable(
        tf.truncated_normal([IMAGE_CHANNEL * IMAGE_SIZE * IMAGE_SIZE * 2]))
    biases = tf.Variable(tf.zeros(2))
    logits = tf.matmul(tf_train_dataset, weights) + biases
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=tf_train_labels))
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    train_preduction = tf.nn.softmax(logits)
    valid_preduction = tf.nn.softmax(
        tf.matmul(tf_valid_dataset, weights) + biases)
    test_prediction = tf.nn.softmax(
        tf.matmul(tf_test_dataset, weights) + biases)

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))) / predictions.shape[0]


TRAIN_ROUNDS = 500 + 1
with tf.Session(graph=graph) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print("Initialized")
    for step in range(TRAIN_ROUNDS):
        _, l, predictions = sess.run([optimizer, loss, train_prediction])
        if (step % 100) ==0:
            print("Training round: %f" % step)
            print("Loss: %f" % l)
            print("Accuracy: %.1f%%" % accuracy(predictions, train_labels))

            print('Validation accuracy: %.1f%%' % accuracy(
                    valid_prediction.eval(), valid_labels))
    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))




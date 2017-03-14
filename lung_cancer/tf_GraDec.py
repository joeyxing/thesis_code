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
    

with tf.Session(graph=graph) as sess:
    init = tf.global_variables_initializer()
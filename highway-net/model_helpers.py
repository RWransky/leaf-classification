from sklearn.cross_validation import train_test_split
import numpy as np


# We will use 80% for training and 20% for validation
def split_data(x, y):
    data_train, data_valid, labels_train, labels_valid = train_test_split(x, y, test_size=0.20, random_state=42)
    print("Shape of training examples = ", data_train.shape)
    print("Shape of validation examples = ", data_valid.shape)
    return data_train, data_valid, labels_train, labels_valid


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/predictions.shape[0])


def reformat(dataset):
    return dataset.reshape((-1, 80, 80, 1)).astype(np.float32)

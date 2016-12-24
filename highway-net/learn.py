'''
Leaf Classifier via Deep Highway Networks

'''

import time
import os

from keras.optimizers import *
from keras.callbacks import *
from keras.models import model_from_json

import numpy as np
import matplotlib.pyplot as plt
import random
import os
import argparse

from model_helpers import *
from data_helpers import *
from network import *
from highway_unit import *

# Setting the training parameters

# Number of possible actions
actions = 99
# Continue training?
continueTraining = False
# The path to save our model to.
path = "./dhn"
base_path = os.getcwd()
model_file = "highway_model.json"
weight_file = "highway_model.h5"


def train():
    print('Loading data...')
    # load data
    images, labels = load_data()
    # convert to training and validation sets
    train_dataset, valid_dataset, train_labels, valid_labels = split_data(images, labels)

    train_mean = np.mean(train_dataset, axis=0)
    train_std = np.std(train_dataset, axis=0)
    np.save('train_mean', train_mean)
    np.save('train_std', train_std)

    train_dataset = (train_dataset - train_mean)/train_std
    valid_dataset = (valid_dataset - train_mean)/train_std

    train_dataset = reformat(train_dataset)
    valid_dataset = reformat(valid_dataset)

    train_labels = np.uint8(train_labels)
    valid_labels = np.uint8(valid_labels)
    train_labels = np.expand_dims(train_labels, -1)
    valid_labels = np.expand_dims(valid_labels, -1)

    # define optimizer
    opt = SGD(lr=0.01, momentum=0.9, nesterov=True)

    if continueTraining:
        with open(model_file, 'r') as jfile:
            model = model_from_json(jfile.read(), {'HighwayUnit': HighwayUnit()})

            model.compile(optimizer=opt, loss='sparse_categorical_crossentropy')
            # weights_file = model_file.replace('json', 'h5')
            # model.load_weights(weights_file)
            model.load_weights(weight_file)
            # define loss function and training metric
    else:
        loss = 'sparse_categorical_crossentropy'
        metric = 'acc'

        print('Building network ...')
        model = build_network()
        model.summary()

        model.compile(optimizer=opt,
                      loss=loss, metrics=[metric])

    print('Training...')

    start_time = time.time()

    saving = 'highway_model'
    fParams = '{0}/{1}.h5'.format(path, saving)
    saveParams = ModelCheckpoint(fParams, monitor='val_loss', save_best_only=True)

    callbacks = [saveParams]

    his = model.fit(train_dataset, train_labels,
                    validation_data=(valid_dataset, valid_labels),
                    nb_epoch=150, batch_size=8,
                    callbacks=callbacks)

    # save model to json file
    fModel = open(path + '/' + saving + '.json', 'w')
    json_str = model.to_json()
    fModel.write(json_str)

    end_time = time.time()

    print('training time: %.2f' % (end_time - start_time))


def test():
    with open(model_file, 'r') as jfile:
        model = model_from_json(jfile.read(), {'HighwayUnit': HighwayUnit()})

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    # weights_file = model_file.replace('json', 'h5')
    # model.load_weights(weights_file)
    model.load_weights(weight_file)

    # load data
    images, image_id, species = load_test_data()

    # load training mean and standard dev
    train_mean = np.load('train_mean.npy')
    train_std = np.load('train_std.npy')

    images = (images - train_mean)/train_std

    test_dataset = reformat(images)

    yP = model.predict(test_dataset)
    np.save('testProbs', yP)
    print('Completed processing {} test images'.format(str(image_id.shape[0])))
    write_results_to_file(species, image_id, yP)


def writeResults():
    # load data
    images, image_id, species = load_test_data()
    # load saved results
    probs = np.load('testProbs.npy')
    write_results_to_file(species, image_id, probs)


def main():
    parser = argparse.ArgumentParser(description="Train or run leaf classifier")
    parser.add_argument("-m", "--mode", help="Train / Run / Write", required=True)
    args = vars(parser.parse_args())
    if args['mode'] == 'Train':
        train()
    elif args['mode'] == 'Test':
        test()
    elif args['mode'] == 'Write':
        writeResults()
    else:
        print(':p Invalid Mode.')


if __name__ == "__main__":
    main()

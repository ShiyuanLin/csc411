import numpy as np


def LoadData(fname):
    """ Loads data """

    npzfile = np.load(fname)

    inputs_train = npzfile['inputs_train'].T / 255.0
    inputs_valid = npzfile['inputs_valid'].T / 255.0
    inputs_test = npzfile['inputs_test'].T / 255.0
    target_train = npzfile['target_train']
    target_valid = npzfile['target_valid']
    target_test = npzfile['target_test']

    return inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test


def LoadDataQ4(fname):
    """ Loads data """
    npzfile = np.load(fname)

    inputs_train = npzfile['inputs_train'].T / 255.0
    inputs_valid = npzfile['inputs_valid'].T / 255.0
    inputs_test = npzfile['inputs_test'].T / 255.0
    target_train = npzfile['target_train']
    target_valid = npzfile['target_valid']
    target_test = npzfile['target_test']

    data = {
        # image
        'x_train_anger': inputs_train[:, target_train == 0],
        'x_valid_anger': inputs_valid[:, target_valid == 0],
        'x_test_anger': inputs_test[:, target_test == 0],
        'x_train_happy': inputs_train[:, target_train == 3],
        'x_valid_happy': inputs_valid[:, target_valid == 3],
        'x_test_happy': inputs_test[:, target_test == 3],
        # label
        'y_train_anger': np.zeros_like(target_train[target_train == 0]),
        'y_valid_anger': np.zeros_like(target_valid[target_valid == 0]),
        'y_test_anger': np.zeros_like(target_test[target_test == 0]),
        'y_train_happy': np.ones_like(target_train[target_train == 3]),
        'y_valid_happy': np.ones_like(target_valid[target_valid == 3]),
        'y_test_happy': np.ones_like(target_test[target_test == 3])
    }

    return data

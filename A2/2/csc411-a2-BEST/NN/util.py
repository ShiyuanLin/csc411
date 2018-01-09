from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
plt.ion()


def LoadData(fname):
    """Loads data from an NPZ file.

    Args:
        fname: NPZ filename.

    Returns:
        data: Tuple {inputs, target}_{train, valid, test}.
              Row-major, outer axis to be the number of observations.
    """
    npzfile = np.load(fname)

    inputs_train = npzfile['inputs_train'].T / 255.0
    inputs_valid = npzfile['inputs_valid'].T / 255.0
    inputs_test = npzfile['inputs_test'].T / 255.0
    target_train = npzfile['target_train'].tolist()
    target_valid = npzfile['target_valid'].tolist()
    target_test = npzfile['target_test'].tolist()

    num_class = max(target_train + target_valid + target_test) + 1
    target_train_1hot = np.zeros([num_class, len(target_train)])
    target_valid_1hot = np.zeros([num_class, len(target_valid)])
    target_test_1hot = np.zeros([num_class, len(target_test)])

    for ii, xx in enumerate(target_train):
        target_train_1hot[xx, ii] = 1.0

    for ii, xx in enumerate(target_valid):
        target_valid_1hot[xx, ii] = 1.0

    for ii, xx in enumerate(target_test):
        target_test_1hot[xx, ii] = 1.0

    inputs_train = inputs_train.T
    inputs_valid = inputs_valid.T
    inputs_test = inputs_test.T
    target_train_1hot = target_train_1hot.T
    target_valid_1hot = target_valid_1hot.T
    target_test_1hot = target_test_1hot.T
    return inputs_train, inputs_valid, inputs_test, target_train_1hot, target_valid_1hot, target_test_1hot


def Save(fname, data):
    """Saves the model to a numpy file."""
    print('Writing to ' + fname)
    np.savez_compressed(fname, **data)


def Load(fname):
    """Loads model from numpy file."""
    print('Loading from ' + fname)
    return dict(np.load(fname))


#def DisplayPlot(train, valid, ylabel, network , c, eps, momentum, batch_size , number=0):
#    """Displays training curve.
#
#    Args:
#        train: Training statistics.
#        valid: Validation statistics.
#        ylabel: Y-axis label of the plot.
#    """
#    plt.figure(number)
#    plt.clf()
#    train = np.array(train)
#    valid = np.array(valid)
#    plt.plot(train[:, 0], train[:, 1], 'b', label='Train')
#    plt.plot(valid[:, 0], valid[:, 1], 'g', label='Validation')
#    plt.xlabel('Epoch',fontsize=12,fontweight='bold')
#    plt.ylabel(ylabel,fontsize =12,fontweight='bold') 
#    plt.title(r'$\epsilon = \ $%s, $\gamma = \ $ %s, Batch Size $=$ %s'%(eps,momentum, batch_size) )
#    plt.suptitle('Plot of %s for a %s with Hyperparameters :'%(ylabel, network,), fontsize=13, fontweight='bold')
#    plt.legend()
#    plt.draw()
#    plt.savefig('%s_%s e_%s m_%s b_%s.png' %( network,c,eps,momentum, batch_size) )


def DisplayPlot1(train, valid, ylabel, network, c , num_hiddens, number=0):
    """Displays training curve.

    Args:
        train: Training statistics.
        valid: Validation statistics.
        ylabel: Y-axis label of the plot.
    """
    
    plt.figure(number)
    plt.clf()
    train = np.array(train)
    valid = np.array(valid)
    plt.plot(train[:, 0], train[:, 1], 'b', label='Train')
    plt.plot(valid[:, 0], valid[:, 1], 'g', label='Validation')
    plt.xlabel('Epoch',fontsize=12,fontweight='bold')
    plt.ylabel(ylabel,fontsize =12,fontweight='bold') 
    plt.title(r'$\# $ of filters in layer 1,2 =  %s,%s '%( num_hiddens[0],num_hiddens[1]) )
    plt.suptitle('Plot of %s for a %s with Similar Parameters:'%(ylabel, network,), fontsize=13, fontweight='bold')
    plt.legend()
    plt.draw()
    plt.savefig('%s_%s_HU1_%s_HU2_%sSP.png' %( network, c , num_hiddens[0],num_hiddens[1] ) )

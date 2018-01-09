# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:39:09 2017

"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from scipy.misc import logsumexp
np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))

#helper function
def l2(A,B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist

#helper function
def run_on_fold(x_test, y_test, x_train, y_train, taus):
    '''
    Input: x_test is the N_test x d design matrix
           y_test is the N_test x 1 targets vector        
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           taus is a vector of tau values to evaluate
    output: losses a vector of average losses one for each tau value
    '''
    N_test = x_test.shape[0]
    losses = np.zeros(taus.shape)
    for j,tau in enumerate(taus):
        predictions =  np.array([LRLS(x_test[i,:].reshape(d,1),x_train,y_train, tau) \
                        for i in range(N_test)])
        losses[j] = ((predictions.flatten()-y_test.flatten())**2).mean()
    return losses
 
 
#to implement
def LRLS(test_datum,x_train,y_train, tau,lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    ## TODO
    #return None
    ## TODO
    a_i = np.divide((l2(test_datum.T, x_train)), ((-2)*(tau**2)))
    a = np.exp(np.subtract(a_i, logsumexp(a_i)))[0]
    # dist = np.exp(exp_a)
    # dist_total = logsumexp(exp_a)
    # a = np.divide(dist, dist_total)
    a = np.diag(a)
    a = a.T

    # a = logsumexp(np.divide((-l2(test_datum, y_train)), (2*tau)),
    #               np.divide((-l2(test_datum, x_train)), (2*tau)))
    # np.dot(x_train.T, a)
    # np.dot(np.dot(x_train.T, a), x_train)
    # np.add(np.dot(np.dot(x_train.T, a), x_train),
    #        np.dot(np.identity(x_train.shape), lam))

    w = np.linalg.solve(np.add(np.dot(np.dot(x_train.T, a), x_train),
                               np.dot(np.identity(x_train.shape[1]), lam)),
                        np.dot(np.dot(x_train.T, a), y_train))

    return np.dot(test_datum.T, w)



def run_k_fold(x,y,taus,k):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector    
           taus is a vector of tau values to evaluate
           K in the number of folds
    output is losses a vector of k-fold cross validation losses one for each tau value
    '''
    ## TODO
    # return None
    ## TODO
    random_i = np.random.choice(len(x), len(x), replace=False)
    # print(X.shape)
    # print(y.shape)
    for i in range(0, len(random_i), (len(random_i)//k)):
        x_test = x[random_i[i:(i + (len(random_i) // k))]]
        y_test = y[random_i[i:(i + (len(random_i) // k))]]
        print("Entering i:", i)
        if i == 0 :
            x_train = x[random_i[(i + (len(random_i) // k)):]]
            y_train = y[random_i[(i + (len(random_i) // k)):]]
            losses = run_on_fold(x_test, y_test, x_train, y_train, taus)
            #print(type(losses))
        else:
            x_train = np.concatenate((x[random_i[0:i]], x[random_i[(i + (len(random_i) // k)):]]), axis=0)
            y_train = np.concatenate((y[random_i[0:i]], y[random_i[(i + (len(random_i) // k)):]]), axis=0)
            print(x_train.shape, x_test.shape)
            print(y_train.shape, y_test.shape)
            np.add(run_on_fold(x_test, y_test, x_train, y_train, taus), losses)
        print('Finish i:', i)
    return losses


if __name__ == "__main__":
    # In this exercise we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0,3,200)
    losses = run_k_fold(x,y,taus,k=5)
    print(losses)
    plt.plot(taus,losses, "bo")
    plt.tight_layout()
    plt.show()
    print("min loss = {}".format(losses.min()))


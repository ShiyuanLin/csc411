""" Methods for doing logistic regression."""

import numpy as np
from utils import sigmoid


def logistic_predict(weights, data):
    """
    Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to the bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
    Outputs:
        y:          :N x 1 vector of probabilities of being second class. This is the output of the classifier.
    """
    # TODO: Finish this function
    a=np.array([],int)
    for row in data:
        b=weights[-1]
        for i in range(len(row)):
            b=b+row[i]*weights[i]
        a=np.append(a,b)
            
    y=np.reshape(sigmoid(a),(len(a),1) )    
    return y


def evaluate(targets, y):
    """
    Compute evaluation metrics.
    Inputs:
        targets : N x 1 vector of targets.
        y       : N x 1 vector of probabilities.
    Outputs:
        ce           : (scalar) Cross entropy. CE(p, q) = E_p[-log q]. Here we want to compute CE(targets, y)
        frac_correct : (scalar) Fraction of inputs classified correctly.
    """
    ce1=0
    for i in range(len(targets)):
        ce1=ce1+targets[i]*np.log(y[i])+(1-targets[i])*np.log(1-y[i])
     
    ce=-ce1
    
    errors= 0 
    for i in range(len(targets)):
        if y[i]>0.5 and targets[i]==0:
            errors=errors+1
        elif y[i]<0.5 and targets[i]==1:
            errors=errors+1
        
    frac_correct=float(1- errors/(len(targets)*1.0) )
        

    return float(ce), frac_correct


def logistic(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:       The sum of the loss over all data points. This is the objective that we want to minimize.
        df:      (M+1) x 1 vector of accumulative derivative of f w.r.t. weights, i.e. don't need to average over number of sample
        y:       N x 1 vector of probabilities.
    """

    y = logistic_predict(weights, data)
    if hyperparameters['weight_regularization'] is True:
        f, df = logistic_pen(weights, data, targets, hyperparameters)
    else:
        # TODO: compute f and df without regularization        
        df1=np.array([],int)
        for j in range(len(weights)-1):            
            a=0
            for i in range(len(data)):
                a=a+data[i][j]*( y[i] - targets[i] )
            df1=np.append(df1,a)    
        
        df_bias =0 
        for i in range(len(data)):
            df_bias =df_bias + y[i]-targets[i]            
        df1=np.append(df1,df_bias)
        df=np.reshape(df1, ( len(df1) , 1 ) )
    
        ce1=0
        for i in range(len(data)):
            ce1=ce1-targets[i]*np.log(y[i])-(1-targets[i])*np.log(1-y[i])
     
        f=float(ce1)
               
    return f, df, y


def logistic_pen(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:             The sum of the loss over all data points. This is the objective that we want to minimize.
        df:            (M+1) x 1 vector of accumulative derivative of f w.r.t. weights, i.e. don't need to average over number of sample
    """
    y = logistic_predict(weights, data)
    ce,fc = evaluate(targets, y)   
    ce=float(ce)
    
    alpha=hyperparameters['weight_decay'] 
    
    df1=np.array([],int)
    for j in range(len(weights)-1):
        a=alpha*weights[j]
        for i in range(len(data)):
            a=a+data[i][j]*( y[i] - targets[i] )
        df1=np.append(df1,a)    
        
    df_bias =alpha*weights[-1] 
    for i in range(len(data)):
        df_bias =df_bias + y[i]-targets[i]            
    df1=np.append(df1,df_bias)
    
    total_weights =0
    for i in range(len(weights)):
        total_weights = total_weights + weights[i]**2
    total_weights = total_weights[0]

    f = ce + 0.5*alpha*total_weights - 0.5*(len(weights)-1)*np.log(0.5*alpha/np.pi)   
        
    df=np.reshape(df1, ( len(df1) , 1 ) )           
    
    return f, df

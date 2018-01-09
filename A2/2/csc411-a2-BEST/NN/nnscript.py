from __future__ import division
from __future__ import print_function

from util import LoadData, Load, Save, DisplayPlot1
from conv2d import conv2d as Conv2D
from nndan import NNUpdate, InitNN, Affine, ReLU, AffineBackward, ReLUBackward, Softmax, CheckGrad, Train, Evaluate
from cnn import InitCNN, CNNUpdate,CNNForward, CNNBackward, Affine, ReLU, AffineBackward, ReLUBackward, Softmax, CheckGrad, Train, Evaluate


import sys 
import numpy as np

def main():
    model_fname = 'cnn_model.npz'
    stats_fname = 'cnn_stats.npz'

    # Hyper-parameters. 
    eps_NN   = 0.01
    eps_CNN  = 0.01    
    momentum = 0.0
    num_epochs_CNN = 30
    num_epochs_NN = 1000

    filter_size = 5
    num_filters_1 = 8
    num_filters_2 = 16

    batch_size = 100

    num_hiddens = [16, 32]
    num_inputs = 2304
    
    # Input-output dimensions.
    num_channels = 1
    num_outputs = 7


    # Uncomment to reload trained model here.
    # model = Load(model_fname)

    
    # PROBLEM 3.1        
    #model = InitNN(num_inputs, num_hiddens, num_outputs)
    
    #model, stats = Train(model, NNForward, NNBackward, NNUpdate, eps_NN,
    #              momentum, num_epochs_NN, batch_size,'NN') 

         
    #model = InitCNN(num_channels, filter_size, num_filters_1, num_filters_2,
    #                num_outputs)   
    #model, stats = Train(model, CNNForward, CNNBackward, CNNUpdate, eps_CNN,
    #              momentum, num_epochs_CNN, batch_size,'CNN')
    
    #PROBLEM 3.2
    
    
#    for eps in [0.05]:
#        model = InitNN(num_inputs, num_hiddens, num_outputs)
#        model, stats = Train(model, NNForward, NNBackward, NNUpdate, eps,
#                  momentum, num_epochs_NN, batch_size,'NN')
#        model = InitCNN(num_channels, filter_size, num_filters_1, num_filters_2,
#                    num_outputs)
#        model, stats = Train(model, CNNForward, CNNBackward, CNNUpdate, eps,
#                  momentum, num_epochs_CNN, batch_size,'CNN')
#        
#      
#    for momentum in [0.1,0.3,0.7]:
#        model = InitNN(num_inputs, num_hiddens, num_outputs)
#        model, stats = Train(model, NNForward, NNBackward, NNUpdate, eps_NN,
#                  momentum, num_epochs_NN, batch_size,'NN')
#        
#        model = InitCNN(num_channels, filter_size, num_filters_1, num_filters_2,
#                    num_outputs)
#        model, stats = Train(model, CNNForward, CNNBackward, CNNUpdate, eps_CNN,
#                  momentum, num_epochs_CNN, batch_size,'CNN')
#    
#    momentum =0.0
#    for batch_size in [ 10,50,100,500,1000]:
#        model = InitNN(num_inputs, num_hiddens, num_outputs)
#        model, stats = Train(model, NNForward, NNBackward, NNUpdate, eps_NN,
#                  momentum, num_epochs_NN, batch_size,'NN')
#        model = InitCNN(num_channels, filter_size, num_filters_1, num_filters_2,
#                    num_outputs)
#        model, stats = Train(model, CNNForward, CNNBackward, CNNUpdate, eps_CNN,
#                  momentum, num_epochs_CNN , batch_size,'CNN')

    #PROBLEM 3.3
    momentum = 0.9        
    
#    
#    for num_hiddens[0]  in [8,24,48]:
#        for num_hiddens[1] in [10,30,50]:
#            model = InitNN(num_inputs, num_hiddens, num_outputs)
#            model, stats = Train(model, NNForward, NNBackward, NNUpdate, eps_NN,
#                  momentum, num_epochs_NN, batch_size,'NN', num_hiddens)
        
        
    for num_filters_1 in [24]:
        for num_filters_2 in [18,48]:
            a=[num_filters_1,num_filters_2]
            model = InitCNN(num_channels, filter_size, num_filters_1, num_filters_2,
                    num_outputs)
            model, stats = Train(model, CNNForward, CNNBackward, CNNUpdate, eps_CNN,
                  momentum, num_epochs_CNN, batch_size,'CNN', a)
    
            
     #Uncomment if you wish to save the model.
    # Save(model_fname, model)

     #Uncomment if you wish to save the training statistics.
    # Save(stats_fname, stats)

if __name__ == '__main__': 
    main()
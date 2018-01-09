# -*- coding: utf-8 -*-

from utils import load_train, load_valid

from run_knn import run_knn

(train_inputs, train_targets)=load_train()
(valid_inputs, valid_targets)=load_valid()

for k in [1,3,5,7,9]:
    print run_knn(k, train_inputs, train_targets, valid_inputs)
     


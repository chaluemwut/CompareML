#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
 Deep Belief Nets (DBN)

 References :
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing
   Systems 19, 2007


   - DeepLearningTutorials
   https://github.com/lisa-lab/DeepLearningTutorials

'''

import sys
import numpy
from HiddenLayer import HiddenLayer
from LogisticRegression import LogisticRegression
from RBM import RBM
from utils import *
from load_data import DataLoader


class DBN(object):
    def __init__(self, input=None, label=None, \
                 n_ins=2, hidden_layer_sizes=[3, 3], n_outs=2, \
                 numpy_rng=None):
        
        self.x = input
        self.y = label

        self.sigmoid_layers = []
        self.rbm_layers = []
        self.n_layers = len(hidden_layer_sizes)  # = len(self.rbm_layers)

        if numpy_rng is None:
            numpy_rng = numpy.random.RandomState(1234)

        
        assert self.n_layers > 0


        # construct multi-layer
        for i in xrange(self.n_layers):
            # layer_size
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layer_sizes[i - 1]

            # layer_input
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].sample_h_given_v()
                
            # construct sigmoid_layer
            sigmoid_layer = HiddenLayer(input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layer_sizes[i],
                                        numpy_rng=numpy_rng,
                                        activation=sigmoid)
            self.sigmoid_layers.append(sigmoid_layer)


            # construct rbm_layer
            rbm_layer = RBM(input=layer_input,
                            n_visible=input_size,
                            n_hidden=hidden_layer_sizes[i],
                            W=sigmoid_layer.W,  # W, b are shared
                            hbias=sigmoid_layer.b)
            self.rbm_layers.append(rbm_layer)


        # layer for output using Logistic Regression
        self.log_layer = LogisticRegression(input=self.sigmoid_layers[-1].sample_h_given_v(),
                                            label=self.y,
                                            n_in=hidden_layer_sizes[-1],
                                            n_out=n_outs)

        # finetune cost: the negative log likelihood of the logistic regression layer
        self.finetune_cost = self.log_layer.negative_log_likelihood()



    def pretrain(self, lr=0.1, k=1, epochs=100):
        # pre-train layer-wise
        for i in xrange(self.n_layers):
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[i - 1].sample_h_given_v(layer_input)
            rbm = self.rbm_layers[i]
            
            for epoch in xrange(epochs):
                rbm.contrastive_divergence(lr=lr, k=k, input=layer_input)
                # cost = rbm.get_reconstruction_cross_entropy()
                # print >> sys.stderr, \
                #        'Pre-training layer %d, epoch %d, cost ' %(i, epoch), cost

    # def pretrain(self, lr=0.1, k=1, epochs=100):
    #     # pre-train layer-wise
    #     for i in xrange(self.n_layers):
    #         rbm = self.rbm_layers[i]
            
    #         for epoch in xrange(epochs):
    #             layer_input = self.x
    #             for j in xrange(i):
    #                 layer_input = self.sigmoid_layers[j].sample_h_given_v(layer_input)
            
    #             rbm.contrastive_divergence(lr=lr, k=k, input=layer_input)
    #             # cost = rbm.get_reconstruction_cross_entropy()
    #             # print >> sys.stderr, \
    #             #        'Pre-training layer %d, epoch %d, cost ' %(i, epoch), cost


    def finetune(self, lr=0.1, epochs=100):
        layer_input = self.sigmoid_layers[-1].sample_h_given_v()

        # train log_layer
        epoch = 0
        done_looping = False
        while (epoch < epochs) and (not done_looping):
            self.log_layer.train(lr=lr, input=layer_input)
            # self.finetune_cost = self.log_layer.negative_log_likelihood()
            # print >> sys.stderr, 'Training epoch %d, cost is ' % epoch, self.finetune_cost
            
            lr *= 0.95
            epoch += 1


    def predict(self, x):
        layer_input = x
        
        for i in xrange(self.n_layers):
            sigmoid_layer = self.sigmoid_layers[i]
            layer_input = sigmoid_layer.output(input=layer_input)

        out = self.log_layer.predict(layer_input)
        return out



def test_dbn(pretrain_lr=0.1, pretraining_epochs=1000, k=1, \
             finetune_lr=0.1, finetune_epochs=200):

    x = numpy.array([[10, 12, 1, 0, 0, 0],
                     [1, 0, 1, 0, 0, 0],
                     [1, 1, 1, 0, 0, 0],
                     [0, 0, 1, 1, 1, 0],
                     [0, 0, 1, 1, 0, 0],
                     [0, 0, 1, 1, 1, 0]])
    y = numpy.array([[10],
                     [20],
                     [30],
                     [40],
                     [50],
                     [60]])

    
    rng = numpy.random.RandomState(123)

    # construct DBN
    dbn = DBN(input=x, label=y, n_ins=6, hidden_layer_sizes=[3, 3], n_outs=1, numpy_rng=rng)

    # pre-training (TrainUnsupervisedDBN)
    dbn.pretrain(lr=pretrain_lr, k=1, epochs=pretraining_epochs)
    
    # fine-tuning (DBNSupervisedFineTuning)
    dbn.finetune(lr=finetune_lr, epochs=finetune_epochs)


    # test
    x = numpy.array([[1, 1, 0, 0, 0, 0],
                     [0, 0, 0, 1, 1, 0],
                     [1, 1, 1, 1, 1, 0]])
    
    print dbn.predict(x)

def test_dbn2(pretrain_lr=0.1, pretraining_epochs=1000, k=1, \
             finetune_lr=0.1, finetune_epochs=10):
    loader = DataLoader()
    x_train, y_train = loader.load_train()
    x_test, y_test = loader.load_test()
    print x_train
#     x = numpy.array()
#     y = numpy.array([[1,1],
#                      [0,0],
#                      [0,0],
#                      [0,1],
#                      [0,1],
#                      [0,0]])

    
    rng = numpy.random.RandomState(123)

    # construct DBN
    dbn = DBN(input=numpy.array(x_train[:5]), label=numpy.array(y_train[:5]), n_ins=8, hidden_layer_sizes=[8, 3], n_outs=1, numpy_rng=rng)

    # pre-training (TrainUnsupervisedDBN)
    dbn.pretrain(lr=pretrain_lr, k=1, epochs=pretraining_epochs)
    
    # fine-tuning (DBNSupervisedFineTuning)
    dbn.finetune(lr=finetune_lr, epochs=finetune_epochs)


    # test
#     x = numpy.array([[1, 1, 0, 0, 0, 0],
#                      [0, 0, 0, 1, 1, 0],
#                      [1, 1, 1, 1, 1, 0],
#                      [0, 0, 0, 0, 1, 0]])
    
    print dbn.predict(x_test)

if __name__ == "__main__":
    test_dbn2()

from load_data import DataLoader
__docformat__ = 'restructedtext en'


import os
import sys
import time

import numpy

import theano
import theano.tensor as T


from logistic_sgd import LogisticRegression, load_data, load_data2


# start-snippet-1
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        self.input = input

        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]


# start-snippet-2
class MLP(object):

    def __init__(self, rng, input, n_in, n_hidden, n_out):

        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )
        # end-snippet-2 start-snippet-3
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
        # end-snippet-3
    
#     def fit(self, x_train, y_train):
#         pass
#    
   
#         p_y = T.nnet.softmax(T.dot(x_train, self.params[0].get_value()) + self.params[1].get_value())
#         y_pred = T.argmax(p_y, axis=1)
#         f = theano.function([], y_pred)
#         return f()
        

def comput_out(w1, b1, w2, b2, x):
    import numpy as np
    input = T.dvector()
    n_in = 8
    n_out = 500
    n_out2 = 2
    rng = np.random.RandomState(1234)
    w = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
    b = np.zeros((n_out,), dtype=theano.config.floatX)
    line_out = T.dot(input, w1) + b1
    out = T.tanh(line_out)
    f = theano.function([input], out)
#     print 'w ', w1
#     print 'b ', b1
#     x = [1, 0, 1, 0, 1, 1, 1, 100]
#     print 'x ', x
#     print f(x)
    out_h = f(x)
#     print 'out h ', out_h
    
#     w2 = np.zeros((n_in, n_out2), dtype='float')
#     b2 = np.zeros((n_out2,), dtype='float')
#     print 'w 2 ',w2
#     print 'b 2 ',b2
    line_out2 = T.nnet.softmax(T.dot(out_h, w2) + b2)
    line_pred = T.argmax(line_out2, axis=1)
    f2 = theano.function([], line_pred)
#     print 'f2 ', f2()
    return f2()
    
def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=2, n_hidden=500):

    datasets = load_data2()

    train_set_x, train_set_y = datasets[0]
#     valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
#     n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
#     n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'
    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    classifier = MLP(
        rng=rng,
        input=x,
        n_in=8*1,
        n_hidden=n_hidden,
        n_out=2
    )

    # start-snippet-4
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    # start-snippet-5
    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-5

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

#     x_in = T.dvector()
#     predict = theano.function(inputs=[x_in], outputs=classifier.logRegressionLayer.y_pred)
    
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
        	# print 'test', n_train_batches

            minibatch_avg_cost = train_model(minibatch_index)
#             print 'predict : ',classifier.hiddenLayer.output.get_value()
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index
#             print classifier.hiddenLayer.output
 

            if patience <= iter:
                done_looping = True
                break
            
#     import numpy as np
#     aa = []
#     aa.append(test_set_x[0])
#     aa.append(test_set_x[1])
    
#     print predict(test_set_x[0])
#     y_predict = classifier.predict(test_set_x[0])
#     print y_predict
    w1 = classifier.params[0].get_value()
    b1 = classifier.params[1].get_value()
    w2 = classifier.params[2].get_value()
    b2 = classifier.params[3].get_value()
    
    loader = DataLoader()
    x_test, y_test = loader.load_test()
    for x_in, y_in in zip(x_test, y_test):
        y_pred = comput_out(w1, b1, w2, b2, x_in)
        print 'y pred ',y_pred,' y out ', y_in
    
#     print classifier.
#     print classifier.params[1].get_value()
    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == '__main__':
    test_mlp()

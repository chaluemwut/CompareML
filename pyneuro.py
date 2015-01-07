import numpy as np
from mlp import MLP
import theano
import theano.tensor as T
# import cPickle
import time

class MultiClassMLP(object):
    
    def __init__(self):
        pass

class MultiLayerPerceptron(object):
    learning_rate = 0.01
    L1_reg = 0.00
    L2_reg = 0.0001
    
    
    def __init__(self):
        self.f = None
        self.f2 = None
           
    def fit(self, x_train, y_train):
        x_train = theano.shared(np.asarray(x_train,
                                               dtype='float'),
                                 borrow=True)
        y_train = theano.shared(np.asarray(y_train,
                                               dtype='float'),
                                 borrow=True)
        y_train = T.cast(y_train, 'int32')       
        batch_size = 2  
        index = T.lscalar()  
        x = T.matrix('x')  
        y = T.ivector('y')         
        rng = np.random.RandomState(1234)
        self.classifier = MLP(rng=rng,
                        input=x,
                        n_in=8 * 1,
                        n_hidden=500, n_out=2)
        
        cost = (
                self.classifier.negative_log_likelihood(y) + 
                self.L1_reg * self.classifier.L1 + 
                self.L2_reg * self.classifier.L2_sqr)
        
        gparams = [T.grad(cost, param) for param in self.classifier.params]

        updates = [
            (param, param - self.learning_rate * gparam)
            for param, gparam in zip(self.classifier.params, gparams)
        ]
        
        train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: x_train[index * batch_size: (index + 1) * batch_size],
            y: y_train[index * batch_size: (index + 1) * batch_size]
        })        
        
        epoch = 0
        done_looping = False
        n_epochs = 1000
        patience = 10000
        n_train_batches = x_train.get_value(borrow=True).shape[0] / batch_size
        
        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in xrange(n_train_batches):
                minibatch_avg_cost = train_model(minibatch_index)
                iter = (epoch - 1) * n_train_batches + minibatch_index
                if patience <= iter:
                    done_looping = True
                    break
                
        self.w1 = self.classifier.params[0].get_value()
        self.b1 = self.classifier.params[1].get_value()
        self.w2 = self.classifier.params[2].get_value()
        self.b2 = self.classifier.params[3].get_value()
#         self.f.close()
    
    def comput_out(self, x):
        input = T.dvector()
        line_out = T.dot(input, self.w1) + self.b1
        out = T.tanh(line_out)
        if self.f is None:
            self.f = theano.function([input], out)
        out_h = self.f(x)
        line_out2 = T.nnet.softmax(T.dot(out_h, self.w2) + self.b2)
        line_pred = T.argmax(line_out2, axis=1)
        if self.f2 is None:
            self.f2 = theano.function([], line_pred)
        return self.f2()
    
    def predict(self, x_test):
        y_pred = []
        for x_in in x_test:
            y = self.comput_out(x_in)
            y_pred.append(y)
        return y_pred
                   

class DeepLearning(object):
     
    def __init__(self):
        pass
    
    def fit(self, x_train, x_test):
        pass
    
    def predict(self, x_test):
        pass

def test_pyneuro():
    start = time.time()
    from load_data import DataLoader
#     loader = DataLoader()
#     datasets = load_data2()
#     x_train, y_train = datasets[0]
    loader = DataLoader()
    x_train, y_train = loader.load_train()
    x_test, y_test = loader.load_test()
#     print 'x train ',x_train
#     print 'y train ', y_train
    mlp = MultiLayerPerceptron()
    mlp.fit(x_train, y_train)
    y_pred = mlp.predict(x_test)
    print y_pred    
    end = time.time()
    print end - start


def test_pyneuro2():
    mlp = MultiLayerPerceptron()
    mlp.fit([
            [0, 0, 1, 1, 1, 1, 1, 1],
            [1, 0, 1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 9, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 7],
            [1, 0, 0, 0, 0, 0, 0, 0]
            ],
            [0, 0, 0, 0, 0]
            )
    print mlp.predict([[1, 0, 0, 0, 0, 0, 0, 0]])
    
if __name__ == '__main__':
    test_pyneuro()

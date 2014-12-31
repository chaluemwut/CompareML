import theano
import theano.tensor as T
import numpy as np
import pylab as pl
from load_data import DataLoader

class BaseAI(object):
    pass

class Layer(object):
    
    def __init__(self, W_init, b_init, activation):
#         n_output, n_input = W_init.shape
        self.W = theano.shared(value=W_init.astype(theano.config.floatX),
                               name='W',
                               borrow=True)
        self.b = theano.shared(value=b_init.reshape(-1, 1).astype(theano.config.floatX),
                               name='b',
                               borrow=True,
                               broadcastable=(False, True))
        self.activation = activation
        self.params = [self.W, self.b]
        
    def output(self, x):
        lin_output = T.dot(self.W, x) + self.b
        return (lin_output if self.activation is None else self.activation(lin_output))


class MultilayerPerceptron(object):
    
    def __init__(self, W_init, b_init, activations):
        assert len(W_init) == len(b_init) == len(activations)
          
        # Initialize lists of layers
        self.layers = []
        # Construct the layers
        for W, b, activation in zip(W_init, b_init, activations):
            self.layers.append(Layer(W, b, activation))
  
        self.params = []
        for layer in self.layers:
            self.params += layer.params
        
    def output(self, x):
        for layer in self.layers:
            x = layer.output(x)
        return x

    def squared_error(self, x, y):
        return T.sum((self.output(x) - y)**2)
    

    
def gradient_updates_momentum(cost, params, learning_rate, momentum):
    assert momentum < 1 and momentum >= 0
    updates = []
    for param in params:
        param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
        updates.append((param, param - learning_rate*param_update))
        updates.append((param_update, momentum*param_update + (1. - momentum)*T.grad(cost, param)))
    return updates
    
def load_test_data():
    np.random.seed(0)
    N = 1000
    y = np.random.random_integers(0, 1, N)
    means = np.array([[-1, 1], [-1, 1]])
    covariances = np.random.random_sample((2, 2)) + 1
    X = np.vstack([np.random.randn(N)*covariances[0, y] + means[0, y],
                   np.random.randn(N)*covariances[1, y] + means[1, y]])
    return X, y

if __name__ == "__main2__":
    x,y = load_test_data()
    print 'x0', x[0]
    print 'x1', x[1]
    
if __name__ == "__main__":
    X, y = load_test_data()
    layer_sizes = [X.shape[0], X.shape[0]*2, 1]
#     print 'layer size : ', layer_sizes, ' end'
    # Set initial parameter values
    
    W_init = []
    b_init = []
    activations = []
    
#     for x_data, y_data in zip(x_train, y_train):
#         W_init.append(x_data)
#         b_init.append(np.ones(len(x_data)))
#         activations.append(T.nnet.sigmoid)
        
    for n_input, n_output in zip(layer_sizes[:-1], layer_sizes[1:]):
        print 'start : ',np.random.randn(n_output, n_input)
        print 'b init ', np.ones(n_output)
        W_init.append(np.random.randn(n_output, n_input))
        b_init.append(np.ones(n_output))
        activations.append(T.nnet.sigmoid)
    
    mlp = MultilayerPerceptron(W_init, b_init, activations)    
    mlp_input = T.matrix('mlp_input')
    mlp_target = T.vector('mlp_target')
    learning_rate = 0.01
    momentum = 0.9
    
    cost = mlp.squared_error(mlp_input, mlp_target)
    train = theano.function([mlp_input, mlp_target], cost,
                            updates=gradient_updates_momentum(cost, mlp.params, learning_rate, momentum))
    
    mlp_output = theano.function([mlp_input], mlp.output(mlp_input))
    
    iteration = 0

    while iteration < 20:
#         print len(X[0])
        current_cost = train(X, y)
        current_output = mlp_output(X)
#         print ' cost ',current_cost
        accuracy = np.mean((current_output > .5) == y)
        print accuracy
        iteration+=1
        
    

import theano
import theano.tensor as T
import numpy as np
from load_data import DataLoader

class Layer(object):
    
    def __init__(self, w, b, activation):
        self.w = theano.shared(w)
        self.b = theano.shared(b)
        self.activation = activation
        self.params = [self.w, self.b]

    def output(self, x):
        lin_output = T.dot(self.W, x) + self.b
        return (lin_output if self.activation is None else self.activation(lin_output))    
    
    def __str__(self, *args, **kwargs):
        return self.w
    
class MLP(object):
    
    def __init__(self, w_init, b_init, activations):
        self.layers = []
        
        for w, b, a in zip(w_init, b_init, activations):
            self.layers.append(Layer(w, b, a))
            
        self.params = []
        for layer in self.layers:
            self.params += layer.params
        print self.params
    
    def train(self, x_train, y_train):
        x_t = T.vector()
        w_t = T.vector()
        y_t = T.dot(w_t, x_t)
        f = theano.function([w_t, x_t], y_t)
        for x, y in zip(x_train, y_train):
            line_out = f(np.array(x), np.ones(len(x))) + np.ones(1)
            print line_out
            print T.nnet.sigmoid(line_out)

if __name__ == '__main__':
    load = DataLoader()
    x_train, y_train = load.load_train()
    w_init = []
    b_init = []
    activations = []
    for x in x_train[:10]:
        w_init.append(x)
        b_init.append(np.ones(len(x)))
        activations.append(T.nnet.sigmoid)
    mlp = MLP(w_init, b_init, activations)
    
#     mlp_input = 
    train = theano.function()
    
#     mlp.train(x_train, y_train)
    

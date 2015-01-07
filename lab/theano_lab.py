import theano
import theano.tensor as T
import numpy as np
from load_data import DataLoader

class Layer(object):
    
    def __init__(self, w, b, activation):
        self.w = theano.shared(value=w.astype(theano.config.floatX),
                               name='w',
                               borrow=True)
        self.b = theano.shared(value=b)
        self.activation = activation
        self.params = [self.w, self.b]

    def output(self, x):
        lin_output = T.dot(self.w, x) + self.b
        return (lin_output if self.activation is None else self.activation(lin_output))
            
class MLP(object):
    
    def __init__(self, w_init, b_init, activations):
#         self.input, self.output = w_init[0], w_init[1]
        self.layers = []
        for w, b, a in zip(w_init, b_init , activations):
            self.layers.append(Layer(w, b, a))
            
        self.params = []
        for layer in self.layers:
            self.params += layer.params 
        
    def squared_error(self, x, y):
        return T.sum((self.output(x) - y) ** 2)
    
    def output(self, x):
        for layer in self.layers:
            x = layer.output(x)
        return x
        
    def __str__(self, *args, **kwargs):
        return str(self.layer)

def gradient_updates_momentum(cost, params, learning_rate, momentum):
    assert momentum < 1 and momentum >= 0
    updates = []
    for param in params:
        param_update = theano.shared(param.get_value(), broadcastable=param.broadcastable)
        updates.append((param, param - learning_rate * param_update))
        updates.append((param_update, momentum * param_update + (1. - momentum) * T.grad(cost, param)))
    return updates

def test_update(cost, params, learning_rate, momentum):
    assert momentum < 1 and momentum >= 0
    updates = []
    for param in params:
        param_update = theano.shared(param.get_value())
        updates.append((param, param - learning_rate * param_update))
#         updates.append((param_update, momentum * param_update + (1. - momentum) * T.grad(cost, param)))
    return updates        
        
class LabTheano(object):
    
    def share1(self):
        x = theano.shared(np.array([1, 2, 2, 4], dtype='int'))
        y = T.lvector()
        f = theano.function(inputs=[y], updates=[(x, x[0:2] + y)])
        f([100, 10])
        print x.get_value()
        
    def share2(self):
        x = theano.shared(np.array([20, 50, 60, 80], dtype='int'))
        y = T.lvector()
#         z = theano.shared(np.array([2,2,2]))
        gparam = [a for a in x]
        print gparam
        f = theano.function(inputs=[y], updates=[(x, x[0:3] - 0.6 * gparam)])
        f([100, 200, 500])
        self.x_va = x.get_value()
    
    def share4(self):
        a = T.matrix()
        x, b = T.vectors('x', 'b')
        y = T.dot(a ** 2, x) + b
        z = T.sum(a ** 2)
        f = theano.function([a, x,
                             theano.Param(b, default=np.array([2])) 
                             ],
                            [y, z])
        print f(np.array([
                          [1, 2, 3]
                          ]),
                np.array([1, 1, 1]))

    def dot1(self):
    	w = T.matrix()
    	x = T.vector()
    	b = np.ones(1)
    	y = T.dot(w, x) + b
    	f = theano.function([w, x], y)
    	print f(np.array([[1, 1, 2], [2, 3, 4]]), np.array([1, 1, 1]))

    def dot2(self):
    	x = T.matrix()
    	w = T.vector()
    	b = np.ones(1)
    	y = T.nnet.sigmoid(T.dot(x, w) + b)
    	f = theano.function([w, x], y)
    	print f(np.array([1, 0, 1]), np.array([[1, 2, 3], [4, 4, 5]]))
   
if __name__ == '__main__':
    load = DataLoader()
    x_train, y_train = load.load_train()
#     print x_train
#     print np.array(x_train)

    w_init = []
    b_init = []
    activators = []
    
    mlp_input = T.dvector()
    mlp_out = T.dscalar()
    
#     w_init.append(np.array(x_train, dtype='float'))
    x_weigth = [[2, 2, 2, 2, 2, 2, 2, 2],
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 8]]
    w_init.append(np.array(x_weigth))
    b_init.append(1.)
    activators.append(T.nnet.sigmoid)
    
    learning_rate = 0.01
    momentum = 0.9
        
    x_data = [1., 1., 1., 1., 1., 1., 1., 1.]
    mlp = MLP(w_init, b_init, activators)
    cost = mlp.squared_error(mlp_input, mlp_out)
    f = theano.function([mlp_input, mlp_out], cost, updates=test_update(cost, mlp.params, learning_rate, momentum))
    
    
    for x_in, y_in in zip(x_train[:10], y_train):
        print 'w', mlp.params[0].get_value(), 'b', mlp.params[1].get_value()
        f(x_in, y_in)

#     print 'w', mlp.params[0].get_value(), 'b', mlp.params[1].get_value() 
#     f(x_weigth, 1)
#     print 'w', mlp.params[0].get_value(), 'b', mlp.params[1].get_value()    
#     f([1, 1, 1, 1, 1, 1, 1, 1], 1)
#     print 'w', mlp.params[0].get_value(), 'b', mlp.params[1].get_value()
#     f([2, 3, 1, 0, 1, 2, 4, 1], 0)
#     print 'w', mlp.params[0].get_value(), 'b', mlp.params[1].get_value()    
#     for x_in, y_in in zip(x_train[:10], y_train[:10]): 
#         print 'x in ',x_in
#         print 'y in ', y_in 
#         f(x_in, y_in)
#         print 'loop : ', mlp.params[0].get_value()
    
#     b = T.dscalar()
#     w = theano.shared(value=x_train.astype(theano.config.floatX))
#     x = theano.shared(value=x_data.astype(theano.config.floatX))
#     line_out = T.dot(w, x) + np.ones(1)
#     f = theano.function([w, x], line_out)
#     print f(x_train, np.ones(8))
#     f = theano.function([w,x, theano.Param(b, default=1)], line_out)
#     print f(x_train, np.ones(8))

#     w = T.matrix()
#     x = T.vector()
#     w_data = [[1., 2., 3., 4., 5., 6., 7., 8.],
#               [1., 0, 3., 0, 5., 0, 7., 0]]
#     x_data = [1., 1., 1., 1., 1., 1., 1., 1.]
#     layer = Layer(np.array(x_train, dtype='float'), 1., T.nnet.sigmoid)
#     cost = layer.output(x)
#     f = theano.function([w, x], cost, on_unused_input='warn')
#     current_cost = f(np.array(x_train, dtype='float'), x_data)
#     
#     f_pred, f_out = T.vectors('f1', 'f2')
#     error_out = T.sum((f_pred - f_out) ** 2)
#     fe = theano.function([f_pred, f_out], error_out)
#     print fe(current_cost, y_train)
    
#     print len(current_cost), len(y_train)
    
#     print np.ones(8) 
#     print 'x train',x_train[:10]
#     print 'y train',y_train[:10]
#     w_init = []
#     b_init = []
#     activations = []
#     for x_data, y_data in zip(x_train[:10], y_train[:10]):
#         print x_data
#         print np.ones(1)
#         w_init.append(x_data)
#         b_init.append(np.ones(8))
#         activations.append(T.nnet.sigmoid)
#     x_train1 = []
#     x_train1.append()
    
#     mlp = MLP(w_init, b_init, activations)
#     x = T.dmatrix()
#     y = T.dvector()
#     cost = mlp.squared_error(x, y)
#     train = theano.function([x,y], cost)
#     current_cost = train(x_train[:10], y_train[:10])
#     print current_cost


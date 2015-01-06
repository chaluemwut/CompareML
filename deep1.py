import theano
import theano.tensor as T
import numpy as np
from load_data import DataLoader

def floatX(x):
    return np.asarray(x, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def model(x, w):
    return T.nnet.softmax(T.dot(x, w))

loader = DataLoader()
x_train, y_train = floatX(loader.load_train()[0]), floatX(loader.load_train()[1])
x_test, y_test = floatX(loader.load_test()[0]), floatX(loader.load_test()[1])

# print x_train
x = T.fvector()
y = T.fvector()
 
w = init_weights((784, 10))
 
py_x = model(x,w)
y_pred = T.argmax(py_x, axis=1)
 
cost = T.mean(T.nnet.categorical_crossentropy(py_x, y))
gradient = T.grad(cost=cost, wrt=w)
updates=[[w, w-gradient*0.05]]
 
train = theano.function(inputs=[x,y], outputs=cost, allow_input_downcast=True)
predict = theano.function(inputs=[x], outputs=y_pred, allow_input_downcast=True)


cost = train(x_train[0], y_train[0])
# for i in range(1000):
#     for start, end in zip(range(0, len(x_train), 2), range(2, len(x_train), 2)):
#         cost = train(x_train[start:end], y_train[start:end])
#     print np.mean(np.argmax(y_test, axis=1)) == predict(x_test)
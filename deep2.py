import theano
import theano.tensor as T
import numpy as np

def test_mlp():
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
    line_out = T.dot(input, w) + b 
    out = T.tanh(line_out)
    f = theano.function([input], out)
    print 'w ', w
    print 'b ', b
    x = [1, 0, 1, 0, 1, 1, 1, 100]
    print 'x ', x
#     print f(x)
    out_h = f(x)
    print 'out h ', out_h
    
    w2 = np.zeros((n_in, n_out2), dtype='float')
    b2 = np.zeros((n_out2,), dtype='float')
    print 'w 2 ',w2
    print 'b 2 ',b2
    line_out2 = T.nnet.softmax(T.dot(out_h, w2) + b2)
    line_pred = T.argmax(line_out2, axis=1)
    f2 = theano.function([], line_pred)
    print 'f2 ', f2()
    
if __name__ == '__main__':
#     print np.ones((2,8), dtype='float')
    test_mlp()

import theano
import theano.tensor as T
import numpy as np

def update_data(params):
    updates = []
    for p in params:
        pass
    return updates

class ShareFunc(object):
    
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
        y = T.dot(a**2, x)+b
        z = T.sum(a**2)
        f = theano.function([a, x, 
                             theano.Param(b, default=np.array([2])) 
                             ],
                            [y, z])
        print f(np.array([
                          [1,2,3]
                          ]),
                np.array([1,1,1]))
    
if __name__ == '__main__':
    obj = ShareFunc()
    obj.share4()
        

import numpy as np
from mlp import MLP
import theano
import theano.tensor as T

class MultiLayerPerceptron(object):
    learning_rate = 0.01
    L1_reg = 0.00
    L2_reg = 0.0001
    batch_size = 20
    def __init__(self):
        pass
    
    def fit(self, x_train, y_train):
        
        index = T.lscalar()  
        x = T.matrix('x')  
        y = T.ivector('y')         
        rng = np.random.RandomState(1234)
        self.classifier = MLP(rng=rng,
                        input=x_train,
                        n_in=8 * 1,
                        n_hidden=500, n_out=2)
        cost = (
        self.classifier.negative_log_likelihood(y_train)
        + self.L1_reg * self.classifier.L1
        + self.L2_reg * self.classifier.L2_sqr)
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
            x: x_train[index * self.batch_size: (index + 1) * self.batch_size],
            y: y_train[index * self.batch_size: (index + 1) * self.batch_size]
        }
    )
    
    def predict(self, x_test):
        pass

class DeepLearning(object):
     
    def __init__(self):
        pass
    
    def fit(self, x_train, x_test):
        pass
    
    def predict(self, x_test):
        pass

def test_pyneuro():
    from load_data import DataLoader
    loader = DataLoader()
    x_train, y_train = loader.load_train()
    x_test, y_test = loader.load_test()
    mlp = MultiLayerPerceptron()
    mlp.fit(x_train, y_train)
    y_pred = mlp.predict(x_test)

if __name__ == '__main__':
    test_pyneuro()

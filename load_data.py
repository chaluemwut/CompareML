import numpy as np
from sklearn.cross_validation import train_test_split
from numpy import dtype
from sklearn import datasets

is_tranfer_data = True

class DataLoader(object):
    
    def __init__(self):
        self.x = np.loadtxt('data/fselect.txt', delimiter=',', dtype=int)
        self.y = np.loadtxt('data/fresult.txt', dtype=int)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.3, random_state=0)
    
    def tranform_y(self, y):
        lst = []
        for i in y:
            if i > 5:
                lst.append(1)
            else:
                lst.append(0)
        return lst
                
    def load_train(self):
        return self.x_train, [self.y_train, self.tranform_y(self.y_train)][is_tranfer_data]
    
    def load_test(self):
        return self.x_test, [self.y_test, self.tranform_y(self.y_test)][is_tranfer_data]

def tranform_y(y):
    lst = []
    for i in y:
        if i > 5:
            lst.append(1)
        else:
            lst.append(0)
    return lst

class MultiDataLoader(object):
    
    def __init__(self):
        self.x = np.loadtxt('data/fselect.txt', delimiter=',', dtype=int)
        self.y = np.loadtxt('data/fresult.txt', dtype=int)
        self.load()
    

    
    def load(self):
#         def template_load(x1, y1):
#             out = []
#             for xo1, yo1 in zip(x1, y1):
#                 out_data = (xo1, yo1)
#                 out.append(out_data)
#             return out
        self.x1, xi, self.y1, yi = train_test_split(self.x, self.y, test_size=0.5, random_state=0)
        self.x2, self.x3, self.y2, self.y3 = train_test_split(xi, yi, test_size=0.5, random_state=0)
#         return template_load(x1, y1), template_load(x2, y2), template_load(x3, y3)
    def train(self):
        return np.array(self.x1, dtype='float'), tranform_y(self.y1)
    
    def test(self):
        return np.array(self.x2, dtype='float'), tranform_y(self.y2)
    
    def validation(self):
        return np.array(self.x3, dtype='float'), tranform_y(self.y3)
 
class IrisLoader(object):
    
    def __init__(self):
        iris = datasets.load_iris()
        self.x = iris.data
        self.y = iris.target
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=0)
    
    def load_train(self):
        return self.x_train, self.y_train
    
    def load_test(self):
        return self.x_test, self.y_test
    
if __name__ == '__main__':
    from logistic_sgd import *
    train, test, validate = load_data('mnist.pkl.gz')
#     print train[0].get_value()[0]
    print train[0].get_value()[1]
#     obj = MultiDataLoader()
#     o1, o2, o3 = obj.load()
#     print o2

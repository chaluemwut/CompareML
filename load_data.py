import numpy as np
from sklearn.cross_validation import train_test_split

class DataLoader(object):
    
    def __init__(self):
        print 'data loader'
        self.x = np.loadtxt('data/fselect.txt', delimiter=',', dtype=int)
        self.y = np.loadtxt('data/fresult.txt', dtype=int)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.3, random_state=0)
        
    def load_train(self):
        return self.x_train, self.y_train
    
    def load_test(self):
        return self.x_test, self.y_test

# obj = DataLoader()
# x_train, y_train = obj.load_train()
# x_test, y_test = obj.load_test()
# print ','.join(list(x_train[2])), y_train[2]
# print x_test[1], y_test[1]
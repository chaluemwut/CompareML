import numpy as np
from sklearn.cross_validation import train_test_split

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

# obj = DataLoader()
# print obj.load_test()
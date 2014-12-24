from machine_learning import *
from load_data import DataLoader
from sklearn.metrics import mean_squared_error

class CmpML(object):
    
    def process(self):
        print 'process'
        data_loader = DataLoader()
        x_train, y_train = data_loader.load_train()
        x_test, y_test = data_loader.load_test()
        ml = [MLSVM(x_train, y_train),
              MLDecisionTree(x_train, y_train),
              MLKNN(x_train, y_train),
#               MLCRF(x_train, y_train),
              MLGaussianNaiveBayes(x_train, y_train)
              ]
        
#         print mean_squared_error(ml[3].predict(x_test), y_test)
        
        for a in ml:
            y_pred = a.predict(x_test)
            print a, " : ", mean_squared_error(y_pred, y_test)
        
        
cmpMl = CmpML()
cmpMl.process()

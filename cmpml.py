from machine_learning import *
from load_data import DataLoader
from sklearn.metrics import *

class CmpML(object):
    
    def process(self):
        print 'process'
        data_loader = DataLoader()
        x_train, y_train = data_loader.load_train()
        x_test, y_test = data_loader.load_test()
        ml = [
              MLNeuralNetwork(x_train, y_train),
              MLSVM(x_train, y_train),
              MLDecisionTree(x_train, y_train),
              MLKNN(x_train, y_train),
#               MLCRF(x_train, y_train),
              MLGaussianNaiveBayes(x_train, y_train)
              ]
                
        for a in ml:
            y_pred = a.predict(x_test)
#             classification_report(y_test, y_pred)
            print "acc", a, " : ", accuracy_score(y_test, y_pred)*100,'%'
#             print "mea", a, " : ", mean_squared_error(y_test, y_pred)
        
        
cmpMl = CmpML()
cmpMl.process()

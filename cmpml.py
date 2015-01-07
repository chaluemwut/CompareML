from machine_learning import *
from load_data import DataLoader
from sklearn.metrics import *

from mlp5 import *

from nolearn.dbn import DBN
# 
# class MLDBN(object):
#     def __init__(self, x_train, y_train):
#         self.dbn = DBN(
#                 [x_train.shape[1], 300, 10],
#                 learn_rates = 0.3,
#                 learn_rate_decays = 0.9,
#                 epochs = 10,
#                 verbose = 1)
#         self.dbn.fit(x_train, y_train)
#         
#     def predict(self, x_test):
#         y_pred = self.db.predict(x_test)
#         print y_pred
               
class CmpML(object):
    
    def process(self):
#         print 'process'
        data_loader = DataLoader()
        x_train, y_train = data_loader.load_train()
        x_test, y_test = data_loader.load_test()
        
        mlp = MLPClassifier(batch_size=1)
        x_mlp = np.asarray([
                 [0, 0, 1, 1, 1, 1, 1, 1],
                 [1, 0, 1, 0, 0, 0, 0, 0],
                 [1, 0, 0, 0, 9, 0, 0, 0],
                 [1, 0, 0, 0, 0, 0, 0, 7],
                 [10, 0, 0, 0, 0, 0, 0, 0]
                 ])
        y_mlp = np.asarray([1, 5, 6, 7, 8])
        mlp.fit(x_mlp,
                y_mlp)
        print mlp.predict(np.asarray([[10, 0, 0, 0, 0, 0, 0, 0]]))
        
#         dbn = DBN(
#                 [x_train.shape[1], 300, 10],
#                 learn_rates = 0.3,
#                 learn_rate_decays = 0.9,
#                 epochs = 10,
#                 verbose = 1)
#         dbn.fit(x_train, y_train)
        
#         ml = [
# #               mlp,
#               LinearNeuralNetwork(x_train, y_train),
#               MLSVM(x_train, y_train),
#               MLSVMKernel(x_train, y_train, 'rbf'),
#               MLDecisionTree(x_train, y_train),
#               MLKNN(x_train, y_train),
# #               MLCRF(x_train, y_train),
#               MLGaussianNaiveBayes(x_train, y_train)
#               ]
#         print "--------------------------------------------"
#         print "{:<17} | {}".format('Method',' Percent prediction')
#         print "--------------------------------------------"
#         for a in ml:
#             y_pred = a.predict(x_test)
#             print "{:<17}  {}%".format(a,accuracy_score(y_test, y_pred)*100)

        
# cmpMl = CmpML()
# cmpMl.process()
if __name__ == '__main__':
    cmpMl = CmpML()
    cmpMl.process()  
#     data_loader = DataLoader()
#     x_train, y_train = data_loader.load_train()
#     x_test, y_test = data_loader.load_test()    
#     obj = MLDBN(x_train , y_train)
#     obj.predict(x_test[0])

from machine_learning import *
from load_data import DataLoader, IrisLoader, SDDataSets
from sklearn.metrics import *

from multilayer_perceptron_classifier import *
from pyneuro import MultiLayerPerceptron

def str_rf(self):
    return 'Random Forest'

def str_svm(self):
    return 'svm'
               
class CmpML(object):
    
    def _corss_validation(self, x_train, y_train):
        pass
    
    def report_result(self, data_map, header):
        print "------------------------------------------------------------------"
        str = "{:<14} | ".format("cls name")
        for h in header:
            str += "{:<14} | ".format(h)
        str += "mean"
        print str
        print "------------------------------------------------------------------"
        for key, value in data_map.iteritems():
            str = "{:<14} | ".format(key)
            for data in value:
                str += "{} | ".format(data)
            str += "{}".format(np.mean(value))
            print str
            print "------------------------------------------------------------------"
        
    def process_cmp_new(self):
        from sklearn import cross_validation
        from sklearn import svm
        datasets = ['adult','cov_type', 'letter']
        RandomForestClassifier.__str__ = str_rf
        svm.SVC.__str__ = str_svm
        
        ml = [RandomForestClassifier(), svm.SVC()]
        result = {}   
        sd = SDDataSets()
        for m in ml:
            ml_result = []
            for d_name in datasets:
                x, y = sd.load(d_name)
                scores = cross_validation.cross_val_score(m, x, y, cv=5, 
                                                          scoring='f1')
                ml_result.append(scores.mean())
            result[m] = ml_result
        self.report_result(result, datasets)
    
    def process_cmp(self):
#         print 'process'
        data_loader = IrisLoader()
        x_train, y_train = data_loader.load_train()
        x_test, y_test = data_loader.load_test()

        ml = [
              MLRandomForest(x_train, y_train),
              LinearNeuralNetwork(x_train, y_train),
              MLSVM(x_train, y_train),
              MLSVMKernel(x_train, y_train, 'rbf'),
              MLDecisionTree(x_train, y_train),
              MLKNN(x_train, y_train),
#               MLCRF(x_train, y_train),
              MLGaussianNaiveBayes(x_train, y_train)
              ]
        print "----------------------------------------"
        print "{:<17} | {} |".format('Method',' Percent prediction')
        print "----------------------------------------"
        for a in ml:
            y_pred = a.predict(x_test)           
            print "{:<17} | {:<19} |%".format(a,accuracy_score(y_test, y_pred)*100)
#             print "y {}".format(y_pred)
            print "----------------------------------------"

                  
if __name__ == '__main__':
    cmpMl = CmpML()
    cmpMl.process_cmp_new()  


# from machine_learning import *
from load_data import DataLoader, IrisLoader, SDDataSets
from sklearn.metrics import *

# from multilayer_perceptron_classifier import *
# from pyneuro import MultiLayerPerceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn import svm
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)

def str_rf(self):
    return 'Random Forest'

def str_svm(self):
    return 'svm'

def str_bagging(self):
    return 'Bagging'

def str_boosted(self):
    return 'Boosted'

def create_letter_p2():
    import string
    data = np.loadtxt('data/letter.data', dtype='S32', delimiter=',')
    lst_a_m = list(string.uppercase)[0:13]
    for d in data:
        if d[0] in lst_a_m:
            d[0] = 1
        else:
            d[0] = -1
    np.savetxt('data/letter.p2', data, fmt='%s', delimiter=',')
    
def create_letter_p1():
#     data = np.loadtxt('data/letter.data', dtype=('S32', int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int), delimiter=',')
    data = np.loadtxt('data/letter.p1', dtype='S32', delimiter=',')
#     data[data[:, 0] != 'O', 0] = 1
#     data[data[:, 0] == 'O', 0] = -1
#     print [i for i in data if i[0] == '-1']
#     np.savetxt('data/letter.p1', data, fmt='%s', delimiter=',')
#     print data
        
              
class CmpML(object):
    
    def select_best_model(self, m, x, y):
        from sklearn.cross_validation import KFold
        from sklearn.metrics import accuracy_score
        import copy
        kf = KFold(len(y), n_folds=5)
        lst_score = []
        lst_model = []
        for train, test in kf:
            m_i = copy.deepcopy(m)
            x_train, y_train = x[train], y[train]
            x_test, y_test = x[test], y[test]
            m_i = m_i.fit(x_train, y_train)
            y_pred = m_i.predict(x_test)
            score = accuracy_score(y_test, y_pred)            
            lst_score.append(score)
            lst_model.append(m_i)
        logging.debug(lst_score)
        max_index = lst_score.index(max(lst_score))
        return lst_model[max_index]
    
    def report_by_metrics(self, data_map, header):
        print "---------------  Report by metrics  -----------------------------"
        str = "{:<14} | ".format("model")
        for h in header:
            str+="{:<14} | ".format(h)
        str+="mean"
        print str
        print "------------------------------------------------------------------"
        for key, value in data_map.iteritems():
            str = "{:<14} | ".format(key)
            lst_value = []
            n = np.array(value)
            lst_value.append(np.average(n[:,0]))
            lst_value.append(np.average(n[:,1]))
            lst_value.append(np.average(n[:,2]))
            lst_value.append(np.average(n[:,3]))
            lst_value.append(np.average(n[:,4]))
            # lst_value.append(np.average(n[:,5]))
            str += "{:<14} | {:<14} | {:<14} | {:<14} | {:<14} | {:<14}".format(lst_value[0],
                                                           lst_value[1],
                                                           lst_value[2],
                                                           lst_value[3],
                                                           lst_value[4],
                                                           # lst_value[5],
                                                           np.average(lst_value))
            print str
    
    def report_by_datasets(self, data_map, header):
        print "------------------- Report by name ------------------------------"
        str = "{:<14} | ".format("model")
        for h in header:
            str+="{:<14} | ".format(h)
        str+="mean"
        print str
        print "------------------------------------------------------------------"
        for key, value in data_map.iteritems():
            str = "{:<14} | ".format(key)
            str += "{:<14} | {:<14} | {:<14} | {:<14} | {:<14} | {:<14}".format(np.mean(value[0]),
                                                           np.mean(value[1]),
                                                           np.mean(value[2]),
                                                           np.mean(value[3]),
                                                           np.mean(value[4]),
                                                           np.mean(value))
            print str


    
    def report_result(self, data_map, header):
        print "------------------------------------------------------------------"
        str = "{:<14} | ".format("model")
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
        from sklearn import svm
        datasets = ['adult','cov_type','letter.p1','letter.p2', 'fbcredibility']
        RandomForestClassifier.__str__ = str_rf
        svm.SVC.__str__ = str_svm
        BaggingClassifier.__str__ = str_bagging
        GradientBoostingClassifier.__str__ = str_boosted

        ml = [GradientBoostingClassifier(),
              RandomForestClassifier(n_estimators=1024),
              BaggingClassifier(DecisionTreeClassifier()),
              svm.SVC()]

        result = {}   
        sd = SDDataSets()
        for m in ml:
#             print '************** ',m
            m_lst = []
            for data_name in datasets:
#                 print '++++++ data set ',data_name
                lst = []
                x_train, y_train, x_test, y_true = sd.load(data_name)
                model = self.select_best_model(m, x_train, y_train)
                y_pred = model.predict(x_test)
                
                acc = accuracy_score(y_true, y_pred)
                fsc = f1_score(y_true, y_pred)
                roc_auc = roc_auc_score(y_true, y_pred)
                # fpr, tpr, thresholds = roc_curve(y_true, y_pred)
                # roc_auc = auc(fpr, tpr)
#                 print roc_auc
                apr = average_precision_score(y_true, y_pred)
                rms = mean_squared_error(y_true, y_pred)
                # mxe = log_loss(y_true, y_pred, normalize=True)
                lst.extend([acc, fsc, roc_auc, apr, rms])
                m_lst.append(lst)
            result[m] = m_lst
        self.report_by_metrics(result,['acc', 'fsc', 'roc', 'apr', 'rms'])
        print ''
        self.report_by_datasets(result, datasets)
    
    def process_cmp(self):
#         print 'process'
        data_loader = IrisLoader()
        x_train, y_train = data_loader.load_train()
        x_test, y_test = data_loader.load_test()

        ml = [
#               MLRandomForest(x_train, y_train),
#               LinearNeuralNetwork(x_train, y_train),
#               MLSVM(x_train, y_train),
#               MLSVMKernel(x_train, y_train, 'rbf'),
#               MLDecisionTree(x_train, y_train),
#               MLKNN(x_train, y_train),
#               MLGaussianNaiveBayes(x_train, y_train)
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
    logging.info('start...')
    cmpMl = CmpML()
    cmpMl.process_cmp_new()
    logging.info('end...')


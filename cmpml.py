# from machine_learning import *
from load_data import DataLoader, IrisLoader, SDDataSets
from sklearn.metrics import *

# from multilayer_perceptron_classifier import *
# from pyneuro import MultiLayerPerceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn import svm
import logging
import numpy as np
import pickle
import time

logging.basicConfig(level=logging.INFO)

def str_rf(self):
    return 'RandomForest'

def str_svm(self):
    return 'svm'

def str_bagging(self):
    return 'Bagging'

def str_boosted(self):
    return 'Boosted'

def str_nb(self):
    return 'NaiveBayes'

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

    def select_best_model_score(self, m, x, y):
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
        return lst_model[max_index], lst_score[max_index]

    def report_by_metrics(self, data_map, header, ml):
        print "----------------------------------------------------  Report by metrics  --------------------------------------------"
        str = "{:<14} | ".format("model")
        for h in header:
            str+="{:<14} | ".format(h)
        str+="mean"
        print str
        print "---------------------------------------------------------------------------------------------------------------------"
        for key in ml:
            value = data_map[key]
            str = "{:<14} | ".format(key)
            lst_value = []
            n = np.array(value)
            lst_value.append(np.average(n[:,0]))
            lst_value.append(np.average(n[:,1]))
            lst_value.append(np.average(n[:,2]))
            lst_value.append(np.average(n[:,3]))
            lst_value.append(np.average(n[:,4]))
            # lst_value.append(np.average(n[:,5]))
            str += "{:<14.3f} | {:<14.3f} | {:<14.3f} | {:<14.3f} | {:<14.3f} | {:<14.3f}".format(lst_value[0],
                                                           lst_value[1],
                                                           lst_value[2],
                                                           lst_value[3],
                                                           lst_value[4],
                                                           # lst_value[5],
                                                           np.mean(lst_value))
            print str
    
    def report_by_datasets(self, data_map, header, ml):
        print "--------------------------------------- Report by name ---------------------------------------------"
        str = "{:<14} | ".format("model")
        for h in header:
            str+="{:<14} | ".format(h)
        str+="mean"
        print str
        print "----------------------------------------------------------------------------------------------------"
        for key in ml:
            value = data_map[key]
            str = "{:<14} | ".format(key)
            str += "{:<14.3f} | {:<14.3f} | {:<14.3f} | {:<14.3f} | {:<14.3f}".format(np.mean(value[0]),
                                                           np.mean(value[1]),
                                                           np.mean(value[2]),
                                                           np.mean(value[3]),
                                                           # np.mean(value[4]),
                                                           np.mean([value[0], value[1], value[2], value[3]])
            )
            print str


    
    def report_result(self, data_map, header, ml):
        print "------------------------------------------------------------------"
        str = "{:<14} | ".format("model")
        for h in header:
            str += "{:<14} | ".format(h)
        str += "mean"
        print str
        print "------------------------------------------------------------------"
        for key in ml:
            value = data_map[key]
            str = "{:<14} | ".format(key)
            for data in value:
                str += "{} | ".format(data)
            str += "{}".format(np.mean(value))
            print str
            print "------------------------------------------------------------------"

    def save_model_nb(self):
        self.init_setup()
        sd = SDDataSets()
        # for m in self.ml:
        for data_name in self.datasets:
            m = GaussianNB()
            x_train, y_train, x_test, y_test = sd.load(data_name)
            model, predict_score = self.select_best_model_score(m, x_train, y_train)
            file_name = "model/{}_{}".format('NaiveBayes', data_name)
            print '{}'.format(predict_score)
            pickle.dump(model, open(file_name, 'wb'))

    def save_model(self):
        self.init_setup()
        sd = SDDataSets()
        # for m in self.ml:
        for data_name in self.datasets:
            lst_model = []
            lst_score = []
            # file_name = "model/{}_{}".format(m, data_name)
            # print file_name
            for i in range(1, 101):
                m = GradientBoostingClassifier(n_estimators=i)
                print 'i = ',i
                x_train, y_train, x_test, y_test = sd.load(data_name)
                model, predict_score = self.select_best_model_score(m, x_train, y_train)
                lst_model.append(model)
                lst_score.append(predict_score)
            # print '{} {}'.format(b_score, b_model.n_estimators)
            # print lst_score
            file_name = "model/{}_{}".format('NaiveBayes', data_name)
            max_index = lst_score.index(max(lst_score))
            b_score = lst_score[max_index]
            b_model = lst_model[max_index]
            print '{} {}'.format(b_score, b_model.n_estimators)
            pickle.dump(b_model, open(file_name, 'wb'))


    def init_setup(self):
        self.datasets = ['adult', 'cov_type', 'letter.p1', 'letter.p2']
        self.ml_name = ['Boosted', 'RandomForest', 'Bagging', 'NaiveBayes']
        RandomForestClassifier.__str__ = str_rf
        svm.SVC.__str__ = str_svm
        BaggingClassifier.__str__ = str_bagging
        GradientBoostingClassifier.__str__ = str_boosted
        GaussianNB.__str__ = str_nb
        self.ml = [GradientBoostingClassifier(),
              RandomForestClassifier(),
              BaggingClassifier(DecisionTreeClassifier()),
              GaussianNB()]
        # self.ml = [GradientBoostingClassifier(n_estimators=1024),
        #       RandomForestClassifier(n_estimators=1024),
        #       BaggingClassifier(DecisionTreeClassifier(), n_estimators=100),
        #       GaussianNB()]
        # return self.datasets, ml

    def process_n_estimation(self):
        self.init_setup()
        for m_name in self.ml_name:
            for d_name in self.datasets:
                file_name = "model/{}_{}".format(m_name, d_name)
                if m_name != 'NaiveBayes':
                    model = pickle.load(open(file_name, 'rb'))
                    print "{} {} {}".format(m_name, d_name, model.n_estimators)

                # print file_name

    def process_save_model(self):
        # ml_name = ['Boosted', 'RandomForest', 'Bagging', 'NaiveBayes']
        ml_name = ['Bagging', 'RandomForest', 'Boosted', 'NaiveBayes']
        datasets = ['adult', 'cov_type', 'letter.p1', 'letter.p2']
        sd = SDDataSets()
        result = {}
        for m_name in ml_name:
            m_lst = []
            for data_name in datasets:
                lst = []
                x_train, y_train, x_test, y_true = sd.load(data_name)
                file_name = "model/{}_{}".format(m_name,data_name)
                m = pickle.load(open(file_name, 'rb'))

                start_time = time.time()
                y_pred = m.predict(x_test)
                total_time = time.time() -start_time
                print "{} {} {} size {}".format(m_name, data_name, total_time, len(y_pred))

                acc = accuracy_score(y_true, y_pred)
                fsc = f1_score(y_true, y_pred)
                roc_auc = roc_auc_score(y_true, y_pred)
                apr = average_precision_score(y_true, y_pred)
                rms = 1-mean_squared_error(y_true, y_pred)
                # mxe = log_loss(y_true, y_pred, normalize=True)
                lst.extend([acc, fsc, roc_auc, apr, rms])
                m_lst.append(lst)
            result[m_name] = m_lst
        # self.report_by_metrics(result,['acc', 'fsc', 'roc', 'apr', 'rms'], ml_name)
        # print ''
        # self.report_by_datasets(result, datasets, ml_name)


    def process_cmp_new(self):
        self.init_setup()

        result = {}   
        sd = SDDataSets()
        for m in self.ml:
            m_lst = []
            for data_name in self.datasets:
#                 print '++++++ data set ',data_name
                lst = []
                x_train, y_train, x_test, y_true = sd.load(data_name)
                model = self.select_best_model(m, x_train, y_train)
                y_pred = model.predict(x_test)
                
                acc = accuracy_score(y_true, y_pred)
                fsc = f1_score(y_true, y_pred)
                roc_auc = roc_auc_score(y_true, y_pred)
                # fpr, tpr, thresholds = roc_curve(y_true, y_pred)
                # roc_auc2 = auc(fpr, tpr)
#                 print roc_auc
                apr = average_precision_score(y_true, y_pred)
                rms = 1-mean_squared_error(y_true, y_pred)
                # mxe = log_loss(y_true, y_pred, normalize=True)
                lst.extend([acc, fsc, roc_auc, apr, rms])
                m_lst.append(lst)
            result[m] = m_lst
        self.report_by_metrics(result,['acc', 'fsc', 'roc', 'apr', 'rms'], self.ml)
        print ''
        self.report_by_datasets(result, self.datasets, self.ml)
    
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
    # cmpMl.process_n_estimation()
    cmpMl.process_save_model()
    # cmpMl.save_model_nb()
    # cmpMl.process_cmp_new()
    logging.info('end...')


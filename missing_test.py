from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *
from svm import LibSVMWrapper
import time, pickle, math, copy
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.cross_validation import train_test_split, KFold
from load_data import SDDataSets
import numpy as np

ml = ['bagging', 'boosted', 'randomforest', 'nb', 'knn', 'decsiontree', 'svm']
metrics = ['acc', 'fsc', 'roc', 'apr', 'rms']
datasets = ['adult', 'cov_type', 'letter.p1', 'letter.p2']
test_size = [0.5]
line_header = '-'*99
line_header_metric = '-'*108

class MissingTest(object):
    sd = SDDataSets()

    def create_data_set(self, data_set_name):
        x, y = self.sd.loadAll(data_set_name)
        if data_set_name == 'cov_type':
            y = y[x[:,4]>0]
            x = x[x[:,4]>0]

        feature_len = len(x[0])
        one_percent_feature = int(math.ceil(feature_len*0.01))
        x_transform = np.transpose(x)
        f_obj = SelectKBest(chi2, k=3)
        f_select = f_obj.fit_transform(x, y)
        x_select_tran = np.transpose(f_select)
        selection_index = []
        for feature_index in range(0, feature_len):
            x_origin = x_transform[feature_index]
            for f_select_index in x_select_tran:
                if np.array_equal(x_origin, f_select_index):
                    selection_index.append(feature_index)

        x_remove = []
        for i in range(0, feature_len):
            x_origin = x_transform[i]
            if i not in selection_index:
                x_remove.append(x_origin)

        # print selection_index
        x_new = np.transpose(x_remove)
        return x_new, y

    def generate_model(self, rate, data_set_name):
        base_model = 101
        base_model_lst = [2,4,8,16,32,64,128,256,1024]
        bagging_lst = []
        for i in base_model_lst:
            bagging_lst.append(BaggingClassifier(DecisionTreeClassifier(), n_estimators=i))

        boosted_lst = []
        for i in base_model_lst:
            boosted_lst.append(GradientBoostingClassifier(n_estimators=i))

        random_lst = []
        for i in base_model_lst:
            random_lst.append(RandomForestClassifier(n_estimators=i))
        knn_lst = []
        for i in range(2, base_model):
            knn_lst.append(KNeighborsClassifier(n_neighbors=i))

        svm_lst = [LibSVMWrapper(kernel=0, degree=0, rate=rate, data_set_name=data_set_name),
                   LibSVMWrapper(kernel=1, degree=0, rate=rate, data_set_name=data_set_name),
                   LibSVMWrapper(kernel=1, degree=3, rate=rate, data_set_name=data_set_name),
                   LibSVMWrapper(kernel=2, degree=0, rate=rate, data_set_name=data_set_name),
                   LibSVMWrapper(kernel=3, degree=0, rate=rate, data_set_name=data_set_name)]

        return {ml[0]:bagging_lst,
                ml[1]:boosted_lst,
                ml[2]:random_lst,
                ml[3]:[GaussianNB()],
                ml[4]:knn_lst,
                ml[5]:[DecisionTreeClassifier()],
                ml[6]:svm_lst
        }

    def array_k_fold(self, rate, data_set_name, x_train, y_train):
        result = {}
        for model_name, model_lst in self.generate_model(rate, data_set_name).iteritems():
            k_model = []
            k_metric = []
            for m in model_lst:
                print 'rate ',rate, ' data set ',data_set_name, ' model ',m
                ki_metric = []
                ki_model = []
                kf = KFold(len(y_train), n_folds=5)
                k_counter = 1
                for train, test in kf:
                    kx_train, ky_train = x_train[train], y_train[train]
                    kx_test, ky_test = x_train[test], y_train[test]
                    mc = copy.deepcopy(m)
                    # mc.fit(kx_train, ky_train)
                    if model_name == ml[6]:
                        mc.fit(kx_train, ky_train, k_counter)
                    else:
                        mc.fit(kx_train, ky_train)
                    y_pred = mc.predict(kx_test)
                    avg_metric = self.find_avg_metric(ky_test, y_pred)
                    ki_metric.append(avg_metric)
                    ki_model.append(mc)
                    k_counter = k_counter+1
                ki_max_metric, ki_max_model = self.find_max_index(ki_metric, ki_model)
                k_metric.append(ki_max_metric)
                k_model.append(ki_max_model)
            max_metric, max_model = self.find_max_index(k_metric, k_model)
            print 'max model ', max_model
            result[model_name] = max_model
        return result

    def find_max_index(self, metric_lst, model_lst):
        max_metric = max(metric_lst)
        max_index = metric_lst.index(max_metric)
        return max_metric, model_lst[max_index]

    def find_avg_metric(self, y_true, y_pred):
        return np.average(self.find_metric(y_true, y_pred))

    def find_metric(self, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        fsc = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred)
        apr = average_precision_score(y_true, y_pred)
        rms = 1-mean_squared_error(y_true, y_pred)
        return [acc, fsc, roc_auc, apr, rms]

    def report_by_datasets(self, result):
        # pickle.dump(result, open('result/result_report', 'wb'))
        print '***************************** Report by datasets ****************\n'
        header = '{:<12}'.format('datasets')
        for data_name in datasets:
            header=header+' | {:<14}'.format(data_name)
        header=header+' | average '

        for rate in test_size:
            print '******************* training rate {} %'.format((1-rate)*100)
            print line_header
            print header
            print line_header
            for m in ml:
                str = '{:<12} | '.format(m)
                lst_datasets = []
                for data_name in datasets:
                    key = (rate, data_name, m,)
                    value = result[key]
                    mean_datasets = np.average(value)
                    str = str+'{:<14} | '.format(mean_datasets)
                    lst_datasets.append(mean_datasets)
                print str+' {:<14}'.format(np.average(lst_datasets))
            print '\n'

    def report_by_metric(self, result):
        print '***************************** Report by metrics ****************\n'
        header = '{:<12} |'.format('datasets')
        for data_name in metrics:
            header=header+'{:<14} |'.format(data_name)
        header=header+'average '

        for rate in test_size:
            print '******************* training rate {} %'.format((1-rate)*100)
            print line_header_metric
            print header
            print line_header_metric
            for m in ml:
                str = '{:<12} |'.format(m)
                acc_lst = []
                fsc_lst = []
                roc_lst = []
                apr_lst = []
                rms_lst = []
                for data_name in datasets:
                    key = (rate, data_name, m,)
                    value = result[key]
                    acc_lst.append(value[0])
                    fsc_lst.append(value[1])
                    roc_lst.append(value[2])
                    apr_lst.append(value[3])
                    rms_lst.append(value[4])
                acc_avg = np.average(acc_lst)
                fsc_avg = np.average(fsc_lst)
                roc_avg = np.average(roc_lst)
                apr_avg = np.average(apr_lst)
                rms_avg = np.average(rms_lst)
                str = str+('{:<14} |'*5+'{:<14}').format(
                    acc_avg,
                    fsc_avg,
                    roc_avg,
                    apr_avg,
                    rms_avg,
                    np.average([acc_avg, fsc_avg, roc_avg, apr_avg, rms_avg])
                )
                print str
            print '\n'

    def report_miss_value(self, result_miss):
        for data_set_name in datasets:
            for model_name in ml:
                key = (data_set_name, model_name,)
                metric_result = result_miss[key]
                print 'key {} metric {}'.format(key, np.average(metric_result))

    def process(self):
        result_miss = {}
        for data_set_name in datasets:
            x, y = self.create_data_set(data_set_name)
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
            result = self.array_k_fold(0.5, data_set_name, x_train, y_train)
            for model_name, model in result.iteritems():
                key = (0.5, data_set_name, model_name,)
                start = time.time()
                y_pred = model.predict(x_test)
                total_time = time.time()-start
                lst_metric = self.find_metric(y_test, y_pred)
                result_miss[key] = lst_metric
        pickle.dump(result_miss, open('datafile/result/result_miss', 'wb'))
        self.report_by_datasets(result_miss)
        self.report_by_metric(result_miss)
        # print result_miss
        # self.report_miss_value(result_miss)

if __name__ == '__main__':
    start = time.time()
    missing = MissingTest()
    missing.process()
    total = (time.time()-start)/(60*60)
    print 'total time {} hour'.format(total)

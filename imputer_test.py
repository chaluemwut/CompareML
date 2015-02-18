from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *
from svm import LibSVMWrapper
from sklearn.cross_validation import train_test_split, KFold
from sklearn.preprocessing import Imputer
from load_data import SDDataSets
import time, pickle, math, copy
import numpy as np

ml = ['bagging', 'boosted', 'randomforest', 'nb', 'knn', 'decsiontree', 'svm']
metrics = ['acc', 'fsc', 'roc', 'apr', 'rms']
datasets = ['adult', 'cov_type', 'letter.p1', 'letter.p2']
test_size = [0.01, 0.05, 0.1]
line_header = '-'*99
line_header_metric = '-'*108

class ImputerTest(object):

    def report(self, result):
        pickle.dump(result, open('datafile/result/result_report', 'wb'))
        self.report_by_metric(result)
        self.report_by_datasets(result)

    def report_by_datasets(self, result):
        # pickle.dump(result, open('result/result_report', 'wb'))
        print '***************************** Report by datasets ****************\n'
        header = '{:<12}'.format('datasets')
        for data_name in datasets:
            header=header+' | {:<14}'.format(data_name)
        header=header+' | average '

        for rate in test_size:
            print '******************* add error rate {} %'.format(rate*100)
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
            print '******************* add error rate {} %'.format(rate*100)
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
                    # print key, value
    def report_time(self, result_time):
        pickle.dump(result_time, open('datafile/result/result_time', 'wb'))
        print '***************************** Report by time ****************\n'
        header = '{:<12}'.format('datasets')
        for data_name in datasets:
            header=header+' | {:<16}'.format(data_name)
        header=header+' | average '

        for rate in test_size:
            print '******************* add error rate {} %'.format(rate*100)
            print line_header
            print header
            print line_header
            for m in ml:
                out_data = '{:<12} | '.format(m)
                out_lst = []
                for data_name in datasets:
                    key = (rate, data_name, m)
                    value = result_time[key]
                    time_per_one = value[0]/value[1]
                    out_lst.append(time_per_one)
                    out_data = out_data+'{:<16} | '.format(time_per_one)
                out_data = out_data+str(np.average(out_lst))
                print out_data
            print '\n'

    def find_avg_metric(self, y_true, y_pred):
        return np.average(self.find_metric(y_true, y_pred))

    def find_metric(self, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        fsc = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred)
        apr = average_precision_score(y_true, y_pred)
        rms = 1-mean_squared_error(y_true, y_pred)
        return [acc, fsc, roc_auc, apr, rms]

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

    def generate_model(self, rate, data_set_name):
        # base_model = 2
        # base_model_lst = [2]
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
        for i in range(1, base_model):
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

    def compare(self):
        result_report_dup = {}
        result_time_dup = {}
        result_report_missing = {}
        result_time_missing = {}
        # max_model = []
        imputerCreateData = ImputerCreateDataFile()
        for rate in test_size:
            for data_set_name in datasets:
                for imputer_type in ['missing']:
                    x_train, x_test, y_train, y_test = imputerCreateData.create_bytype(rate, data_set_name, imputer_type)
                    result = self.array_k_fold(rate, data_set_name, x_train, y_train)
                    for model_name, model in result.iteritems():
                        key = (rate, data_set_name, model_name,)
                        print key
                        start = time.time()
                        y_pred = model.predict(x_test)
                        # max_model.append(model)
                        total_time = time.time()-start
                        lst_metric = self.find_metric(y_test, y_pred)
                        if imputer_type == 'dup':
                            result_report_dup[key] = lst_metric #5 metric
                            result_time_dup[key] = (total_time, len(y_pred),)
                        else:
                            result_report_missing[key] = lst_metric #5 metric
                            result_time_missing[key] = (total_time, len(y_pred),)

        # pickle.dump(max_model, open('datafile/result/max_model','wb'))
        # print '*'*30+' dup report '+'*'*30
        # self.report(result_report_dup)
        # self.report_time(result_time_dup)
        # print '*'*60

        print '*'*30+' missing report '+'*'*30
        self.report(result_report_missing)
        self.report_time(result_time_missing)
        print '*'*60


class ImputerCreateDataFile(object):
    training_datasize = 0.5
    sd = SDDataSets()
    def create_datafile_duplication(self, rate, data_set_name):
        x, y = self.sd.loadAll(data_set_name)
        x_train, x_test, y_train, y_test = train_test_split(x, y , test_size=self.training_datasize, random_state=0)
        data_len = len(x_train)
        len_extend = math.ceil(rate*data_len)
        target_x = x_train[0]
        target_y = y_train[0]

        for i in range(0, int(len_extend)):
            np.append(x_train, target_x)
            np.append(y_train, target_y)

        return x_train, x_test, y_train, y_test

    def create_datafile_missing_value(self, rate, data_set_name):
        lst_feature = [2,4]
        base_missing_value = -10021
        x, y = self.sd.loadAll(data_set_name)
        x_train, x_test, y_train, y_test = train_test_split(x, y , test_size=self.training_datasize, random_state=0)
        # print 'x train',x_train
        x_data_len = len(x_train)
        x_miss_len = x_data_len*rate
        x_train_missing = x_train[0:x_miss_len]
        x_train_new = x_train[x_miss_len:x_data_len]

        imp = Imputer(missing_values=base_missing_value, strategy='mean', axis=0)
        imp.fit(x_train_new)
        for feature in lst_feature:
            x_train_missing[:,feature] = base_missing_value
        x_train_missing_new = imp.transform(x_train_missing)

        x_train_transform = []
        x_train_transform.extend(x_train_missing_new)
        x_train_transform.extend(x_train_new)

        return np.array(x_train_transform).astype(int), x_test, y_train, y_test

    def create_bytype(self, rate, data_set_name, imputer_type):
        if imputer_type == 'dup':
            return self.create_datafile_duplication(rate, data_set_name)
        else:
            return self.create_datafile_missing_value(rate, data_set_name)

    def create_datafile(self):
        for rate in test_size:
            for data_set_name in datasets:
                self.create_datafile_duplication(rate, data_set_name)
                self.create_datafile_missing_value(rate, data_set_name)

if __name__ == '__main__':
    start = time.time()
    # impMLTest = ImputerCreateDataFile()
    # print impMLTest.create_datafile_missing_value(0.01, 'cov_type')[0]
    # impMLTest.create_datafile()
    imp = ImputerTest()
    imp.compare()
    total = (time.time()-start)/(60.0*60.0)
    print 'Total time {} minute'.format(total)

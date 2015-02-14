from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *
from svm import LibSVMWrapper
import time, pickle, copy
import numpy as np

ml = ['bagging', 'boosted', 'randomforest', 'nb', 'knn', 'decsiontree', 'svm']
metrics = ['acc', 'fsc', 'roc', 'apr', 'rms']
datasets = ['adult', 'cov_type', 'letter.p1', 'letter.p2']
libsvm_path = '/home/off/libsvm-3.20'
base_file_path = 'datafile'
base_file_train_path = base_file_path+'/train'
base_file_test_path = base_file_path+'/test'
base_file_model_path = base_file_train_path+'/model'
test_size = [.75, .50, .25]
line_header = '-'*99
line_header_metric = '-'*108

class MainCompare(object):

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
                    # print key, value
    def report_time(self, result_time):
        pickle.dump(result_time, open('datafile/result/result_time', 'wb'))
        print '***************************** Report by time ****************\n'
        header = '{:<12}'.format('datasets')
        for data_name in datasets:
            header=header+' | {:<16}'.format(data_name)
        header=header+' | average '

        for rate in test_size:
            print '******************* training rate {} %'.format((1-rate)*100)
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

    def generate_model(self, rate, data_set_name):
        # base_model = 201
        # base_model_lst = [2,4,8,16,32,64,128,256,1024,2048,4096,8192]
        base_model = 6
        base_model_lst = [2,4,8,32]
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

    def find_max_index(self, metric_lst, model_lst):
        max_metric = max(metric_lst)
        max_index = metric_lst.index(max_metric)
        return max_metric, model_lst[max_index]

    def array_k_fold(self, rate, data_set_name):
        result = {}
        for model_name, model_lst in self.generate_model(rate, data_set_name).iteritems():
            k_model = []
            k_metric = []
            for m in model_lst:
                print 'm',m
                ki_metric = []
                ki_model = []
                for k in range(1, 6):
                    print 'k {} , rate {} , dataset {}'.format(k, rate, data_set_name)
                    x_train_path = 'datafile/train/obj/train_k{}_x_{}_{}'.format(k,
                                                                           data_set_name,
                                                                           rate)
                    y_train_path = 'datafile/train/obj/train_k{}_y_{}_{}'.format(k,
                                                                           data_set_name,
                                                                           rate)
                    x_test_path = 'datafile/train/obj/test_k{}_x_{}_{}'.format(k,
                                                                               data_set_name,
                                                                               rate)
                    y_test_path = 'datafile/train/obj/test_k{}_y_{}_{}'.format(k,
                                                                               data_set_name,
                                                                               rate)
                    x_train = pickle.load(open(x_train_path, 'rb'))
                    y_train = pickle.load(open(y_train_path, 'rb'))
                    x_test = pickle.load(open(x_test_path, 'rb'))
                    y_test = pickle.load(open(y_test_path, 'rb'))
                    mc = copy.deepcopy(m)
                    if model_name == ml[6]:
                        mc.fit(x_train, y_train, k)
                    else:
                        mc.fit(x_train, y_train)
                        num_n_estimator = ''
                        try:
                            num_n_estimator = mc.n_estimators
                        except AttributeError:
                            pass
                        try:
                            num_n_estimator = mc.n_neighbors
                        except AttributeError:
                            pass
                        path_save_model = 'datafile/result/model/{}_rate{}_dataset{}_k{}_n_estimator{}'.format(model_name,
                                                                                                 rate,
                                                                                                 data_set_name,
                                                                                                 k,
                                                                                                 num_n_estimator)
                        pickle.dump(mc, open(path_save_model, 'wb'))
                    y_pred = mc.predict(x_test)
                    avg_metric = self.find_avg_metric(y_test, y_pred)
                    ki_metric.append(avg_metric)
                    ki_model.append(mc)
                ki_max_metric, ki_max_model = self.find_max_index(ki_metric, ki_model)
                k_metric.append(ki_max_metric)
                k_model.append(ki_max_model)
            max_metric, max_model = self.find_max_index(k_metric, k_model)
            print 'max model ', max_model
            result[model_name] = max_model
        return result

    def compare(self):
        result_report = {}
        result_time = {}
        max_model = []
        for rate in test_size:
            for data_set_name in datasets:
                result = self.array_k_fold(rate, data_set_name)
                for model_name, model in result.iteritems():
                    x_test_path = 'datafile/test/obj/test_x_{}_{}'.format(data_set_name, rate)
                    y_test_path = 'datafile/test/obj/test_y_{}_{}'.format(data_set_name, rate)
                    x_test = pickle.load(open(x_test_path, 'rb'))
                    y_test = pickle.load(open(y_test_path, 'rb'))
                    key = (rate, data_set_name, model_name,)
                    start = time.time()
                    y_pred = model.predict(x_test)
                    max_model.append(model)
                    total_time = time.time()-start
                    lst_metric = self.find_metric(y_test, y_pred)
                    result_report[key] = lst_metric #5 metric
                    result_time[key] = (total_time, len(y_pred),)

        pickle.dump(max_model, open('datafile/result/max_model','wb'))
        self.report(result_report)
        self.report_time(result_time)


if __name__ == '__main__':
    start = time.time()
    cmp = MainCompare()
    cmp.compare()
    total = (time.time()-start)/60.0
    print 'Total time {} m'.format(total)
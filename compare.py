from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from load_data import SDDataSets
from sklearn.svm import SVC
import numpy as np
import pickle, copy, time
from sklearn.metrics import *

ml = ['bagging', 'boosted', 'randomforest', 'nb', 'knn', 'decsiontree']
line_header = '-'*99
line_header_metric = '-'*108

class Compare(object):

    def __init__(self):
        self.setup()

    def setup(self):
        self.datasets = ['adult', 'cov_type', 'letter.p1', 'letter.p2']
        self.metrics = ['acc', 'fsc', 'roc', 'apr', 'rms']
        self.test_size = [.75, .50, .25]
        self.sd = SDDataSets()

    def generate_model_2(self, data_len):
        base_model = 201
        bagging_lst = []
        for i in range(1, base_model):
            bagging_lst.append(BaggingClassifier(DecisionTreeClassifier(), n_estimators=i))

        boosted_lst = []
        for i in range(1 , base_model):
            boosted_lst.append(GradientBoostingClassifier(n_estimators=i))

        random_lst = []
        for i in range(1, base_model):
            random_lst.append(RandomForestClassifier(n_estimators=i))
        knn_lst = []
        for i in range(1, base_model):
            knn_lst.append(KNeighborsClassifier(n_neighbors=i))

        svm_lst = []
        svm_lst.append(SVC(kernel='linear'))
        svm_lst.append(SVC(kernel='poly', degree=2))
        svm_lst.append(SVC(kernel='poly', degree=3))
        svm_lst.append(SVC(kernel='sigmoid'))

        return {ml[0]:bagging_lst,
                ml[1]:boosted_lst,
                ml[2]:random_lst,
                ml[3]:[GaussianNB()],
                ml[4]:knn_lst,
                ml[5]:[DecisionTreeClassifier()]
                # ml[6]:svm_lst
        }

    def generate_model(self, data_len):
        base_model = 201
        base_model_lst = [2,4,8,16,32,64,128,256,1024,2048,4096,8192]
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

        svm_lst = []
        svm_lst.append(SVC(kernel='linear'))
        svm_lst.append(SVC(kernel='poly', degree=2))
        svm_lst.append(SVC(kernel='poly', degree=3))
        svm_lst.append(SVC(kernel='sigmoid'))

        return {ml[0]:bagging_lst,
                ml[1]:boosted_lst,
                ml[2]:random_lst,
                ml[3]:[GaussianNB()],
                ml[4]:knn_lst,
                ml[5]:[DecisionTreeClassifier()]
                # ml[6]:svm_lst
        }

    def find_avg_metric(self, y_true, y_pred):
        return np.average(self.find_metric(y_true, y_pred))

    def find_metric(self, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        fsc = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred)
        apr = average_precision_score(y_true, y_pred)
        rms = 1-mean_squared_error(y_true, y_pred)
        return [acc, fsc, roc_auc, apr, rms]

    def find_best_model(self, x, y, data_name, rate):
        data_len = len(x)
        for model_name, model_lst in self.generate_model(data_len).iteritems():
            k_model = []
            k_metric = []
            for m in model_lst:
                print 'n estimator ',m.n_estimators
                kf = KFold(len(y), n_folds=5)
                j_metric_lst = []
                j_model_lst = []
                for train, test in kf:
                    m_i = copy.deepcopy(m)
                    x_train, y_train = x[train], y[train]
                    x_test, y_test = x[test], y[test]
                    m_i.fit(x_train, y_train)
                    y_pred = m_i.predict(x_test)
                    avg_metric = self.find_avg_metric(y_test, y_pred)
                    j_metric_lst.append(avg_metric)
                    j_model_lst.append(m_i)
                j_max_metric = max(j_metric_lst).item()
                j_max_index = j_metric_lst.index(j_max_metric)
                j_model = j_model_lst[j_max_index]
                k_model.append(j_model)
                k_metric.append(j_max_metric)

            max_metric = max(k_metric)
            max_metric_index = k_metric.index(max_metric)
            model_of_max_metric = k_model[max_metric_index]
            path_model_name = 'result/{}_{}_{}'.format(model_name, data_name, rate)
            path_metric = 'result/data/{}_{}_{}_data'.format(model_name, data_name, rate)
            pickle.dump(model_of_max_metric, open(path_model_name, 'wb'))
            pickle.dump(k_metric, open(path_metric, 'wb'))

    def test_model(self):
        for rate in self.test_size:
            for data_name in self.datasets:
                path = 'result/bagging_{}_{}'.format(data_name, rate)
                m = pickle.load(open(path, 'rb'))
                print path,': ',m.n_estimators

    def load_metric(self):
        for rate in self.test_size:
            for data_name in self.datasets:
                path = 'result/data/bagging_{}_{}_data'.format(data_name, rate)
                lst = pickle.load(open(path, 'rb'))
                print lst

    def array_k_fold(self, x, y, data_name, rate):
        result = {}
        data_len = len(x)
        for model_name, model_lst in self.generate_model(data_len).iteritems():
            print 'model name ',model_name
            k_model = []
            k_metric = []
            for m in model_lst:
                # print 'n estimator ',m.n_estimators
                print 'rate : {} data set : {} ml : {}'.format(rate, data_name, m)
                kf = KFold(len(y), n_folds=5)
                j_metric_lst = []
                j_model_lst = []
                for train, test in kf:
                    m_i = copy.deepcopy(m)
                    x_train, y_train = x[train], y[train]
                    x_test, y_test = x[test], y[test]
                    m_i.fit(x_train, y_train)
                    y_pred = m_i.predict(x_test)
                    avg_metric = self.find_avg_metric(y_test, y_pred)
                    j_metric_lst.append(avg_metric)
                    j_model_lst.append(m_i)
                j_max_metric = max(j_metric_lst).item()
                j_max_index = j_metric_lst.index(j_max_metric)
                j_model = j_model_lst[j_max_index]
                k_model.append(j_model)
                k_metric.append(j_max_metric)

            max_metric = max(k_metric)
            max_metric_index = k_metric.index(max_metric)
            model_of_max_metric = k_model[max_metric_index]
            path_model_name = 'result/model/{}_{}_{}'.format(model_name, data_name, rate)
            path_metric = 'result/metric/{}_{}_{}_data'.format(model_name, data_name, rate)
            pickle.dump(model_of_max_metric, open(path_model_name, 'wb'))
            pickle.dump(k_metric, open(path_metric, 'wb'))
            result[model_name] = model_of_max_metric
        return result

    def log_data(self, data_name, rate, x_train, x_test, y_train, y_test):
        path_train_x = 'result/data/train/x_{}_{}'.format(data_name, rate)
        path_train_y = 'result/data/train/y_{}_{}'.format(data_name, rate)
        path_test_x = 'result/data/test/x_{}_{}'.format(data_name, rate)
        path_test_y = 'result/data/test/y_{}_{}'.format(data_name, rate)
        pickle.dump(x_train, open(path_train_x, 'wb'))
        pickle.dump(y_train, open(path_train_y, 'wb'))
        pickle.dump(x_test, open(path_test_x, 'wb'))
        pickle.dump(y_test, open(path_test_y, 'wb'))

    def report(self, result):
        pickle.dump(result, open('result/result_report', 'wb'))
        self.report_by_metric(result)
        self.report_by_datasets(result)

    def report_by_datasets(self, result):
        # pickle.dump(result, open('result/result_report', 'wb'))
        print '***************************** Report by datasets ****************\n'
        header = '{:<12}'.format('datasets')
        for data_name in self.datasets:
            header=header+' | {:<14}'.format(data_name)
        header=header+' | average '

        for rate in self.test_size:
            print '******************* training rate {} %'.format((1-rate)*100)
            print line_header
            print header
            print line_header
            for m in ml:
                str = '{:<12} | '.format(m)
                lst_datasets = []
                for data_name in self.datasets:
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
        for data_name in self.metrics:
            header=header+'{:<14} |'.format(data_name)
        header=header+'average '

        for rate in self.test_size:
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
                for data_name in self.datasets:
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
        pickle.dump(result_time, open('result/result_time', 'wb'))
        print '***************************** Report by time ****************\n'
        header = '{:<12}'.format('datasets')
        for data_name in self.datasets:
            header=header+' | {:<16}'.format(data_name)
        header=header+' | average '

        for rate in self.test_size:
            print '******************* training rate {} %'.format((1-rate)*100)
            print line_header
            print header
            print line_header
            for m in ml:
                out_data = '{:<12} | '.format(m)
                out_lst = []
                for data_name in self.datasets:
                    key = (rate, data_name, m)
                    value = result_time[key]
                    time_per_one = value[0]/value[1]
                    out_lst.append(time_per_one)
                    out_data = out_data+'{:<16} | '.format(time_per_one)
                out_data = out_data+str(np.average(out_lst))
                print out_data
            print '\n'

    def write_data_file(self, f_train, x_data_train, y_data_train):
        data = str(y_data_train) + ' '
        counter = 1
        for i in x_data_train:
            data = data + str(counter) + ':' + str(i) + ' '
            counter = counter + 1
        f_train.write(data[:-1]+'\n')

    def create_dataset(self):
        for rate in self.test_size:
            for data_set_name in self.datasets:
                path_train = 'libsvm/train_{}_{}'.format(data_set_name,rate)
                path_test = 'libsvm/test_{}_{}'.format(data_set_name, rate)
                f_train = open(path_train, 'w')
                f_test = open(path_test, 'w')
                x, y = self.sd.loadAll(data_set_name)
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=rate, random_state=0)
                for x_data_train, y_data_train in zip(x_train, y_train):
                    self.write_data_file(f_train, x_data_train, y_data_train)

                for x_data_test, y_data_test in zip(x_test, y_test):
                    self.write_data_file(f_test, x_data_test, y_data_test)

                f_train.close()
                f_test.close()

    def libsvm(self):
        pass


    def create_model(self):
        result_report = {}
        result_time = {}
        for rate in self.test_size:
            key = ()
            print 'traing data size {} %'.format((1-rate)*100)
            for data_name in self.datasets:
                print 'datasets name ',data_name
                x, y = self.sd.loadAll(data_name)
                x_train, x_test, y_train, y_test = train_test_split(x, y , test_size=rate, random_state=0)
                self.log_data(data_name, rate, x_train, x_test, y_train, y_test)
                result = self.array_k_fold(x_train, y_train, data_name, rate)
                for model_name, model in result.iteritems():
                    key = (rate, data_name, model_name,)
                    start = time.time()
                    y_pred = model.predict(x_test)
                    total_time = time.time()-start
                    lst_metric = self.find_metric(y_test, y_pred)
                    result_report[key] = lst_metric #5 metric
                    result_time[key] = (total_time, len(y_pred),)
                    # print model_name, lst_metric

        self.report(result_report)
        self.report_time(result_time)

if __name__ == '__main__':
    start = time.time()
    obj = Compare()
    obj.create_dataset()
    # obj.create_model()
    total = (time.time()-start)/(60.0)
    print 'Total time execute {} second'.format(total)
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from load_data import SDDataSets
import numpy as np
import pickle
from sklearn.metrics import *

class Compare(object):

    def setup(self):
        self.datasets = ['adult', 'cov_type', 'letter.p1', 'letter.p2']
        self.train_rate = [.25, .5, .75]
        self.sd = SDDataSets()

    def variance_training_data(self):
        self.setup()
        for var_value in self.train_rate:
            for data_name in self.datasets:
                pass
                # x_train, x_test, y_train, y_test = self.sd.loadAll(data_name)
                # print var_value,' : ',data_name

    def generate_model(self):
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
        return {'bagging':bagging_lst}
        # return {'bagging':bagging_lst,
        #         'randomforest':random_lst,
        #         'boosted':boosted_lst,
        #         'knn':knn_lst,
        #         'dtree':[DecisionTreeClassifier()],
        #         'svm':svm_lst,
        #         'nb':[GaussianNB()]
        #         }

    def find_avg_metric(self, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        fsc = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred)
        apr = average_precision_score(y_true, y_pred)
        rms = 1-mean_squared_error(y_true, y_pred)
        return np.average([acc, fsc, roc_auc, apr, rms])

    def find_best_model(self, x, y, data_name, rate):
        for model_name, model_lst in self.generate_model().iteritems():
            metric_lst_data = []
            model_lst_data = []
            for m in model_lst:
                kf = KFold(len(y), n_folds=5)
                k_metric_lst = []
                for train, test in kf:
                    x_train, y_train = x[train], y[train]
                    x_test, y_test = x[test], y[test]
                    m.fit(x_train, x_test)
                    y_pred = m.predict(x_test)
                    avg_metric = self.find_avg_metric(y_test, y_pred)
                    k_metric_lst.append(avg_metric)
                max_metric = max(k_metric_lst)
                metric_lst_data.append(max_metric)
                model_lst_data.append(m)
            max_metric = max(metric_lst_data)
            max_index = metric_lst_data.index(max_metric)
            best_model = model_lst_data[max_metric]
            path_model_name = '{}_{}_{}'.formate(model_name, data_name, rate)
            path_metric = 'data/{}_{}_{}_data'.formate(model_name, data_name, rate)
            pickle.dump(best_model, open(path_model_name, 'wb'))
            pickle.dump(metric_lst_data, open(path_metric, 'wb'))

    def create_model(self):
        self.setup()
        for rate in self.train_rate:
            for data_name in self.datasets:
                x, y = self.sd.loadAll(data_name)
                x_train, x_test, y_train, y_test = train_test_split(x, y , test_size=rate, random_state=0)
                self.find_best_model(x_train, y_train, data_name, rate)

    def compare(self):
        print 'compare'

if __name__ == '__main__':
    obj = Compare()
    obj.create_model()
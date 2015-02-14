from load_data import SDDataSets
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
import time, pickle, os

ml = ['bagging', 'boosted', 'randomforest', 'nb', 'knn', 'decsiontree']
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

class Cml(object):

    def __init__(self):
        self.sd = SDDataSets()

    def process_cmp_libsvm(self):
        # file_train = 'datafile/libsvm/train_{}_{}'.format('adult',0.25)
        # str_cmd = '{} {} {}'.format('/home/off/libsvm-3.20/svm-train',
        #                             file_train,
        #                             'datafile/libsvm/model/{}_{}'.format('adult',0.25))
        # os.system(str_cmd)
        # file_train = 'datafile/libsvm/model/{}_{}'.format('adult',0.25)
        # str_cmd = '{} {} {} {}'.format('/home/off/libsvm-3.20/svm-predict',
        #                          'datafile/libsvm/test_{}_{}'.format('adult', 0.25),
        #                          'datafile/libsvm/model/{}_{}'.format('adult',0.25),
        #                          'datafile/libsvm/result/{}_{}'.format('adult',0.25)
        #                          )
        # os.system(str_cmd)

        # create model
        # for rate in test_size:
        #     for data_set_name in datasets:
        #         for k in range(1, 6):
        #             model_path = '{}/kfold/k{}_{}_{}'.format(base_file_model_path, k, data_set_name, rate)
        #             train_path = '{}/libsvm/train_k{}_{}_{}'.format(base_file_train_path,
        #                                                             k,
        #                                                             data_set_name,
        #                                                             rate)
        #             cmd_create_model = '{}/svm-train {} {}'.format(libsvm_path,
        #                                                            train_path,
        #                                                            model_path
        #                                                            )
        #             print cmd_create_model
        #             os.system(cmd_create_model)

        # predict model
        for rate in test_size:
            for data_set_name in datasets:
                for k in range(1, 6):
                    model_path = '{}/kfold/k{}_{}_{}'.format(base_file_model_path, k, data_set_name, rate)
                    test_file_path = '{}/libsvm/test_k{}_{}_{}'.format(base_file_train_path,
                                                                       k,
                                                                       data_set_name,
                                                                       rate)
                    result_file_path = '{}/predict/result_k{}_{}_{}'.format(base_file_train_path,
                                                                                   k,
                                                                                   data_set_name,
                                                                                   rate)
                    cmd_predict = '{}/svm-predict {} {} {}'.format(libsvm_path,
                                                                   test_file_path,
                                                                   model_path,
                                                                   result_file_path
                                                                   )
                    print cmd_predict
                    os.system(cmd_predict)


    def gen_str_line(self, x_data_train, y_data_train):
        data = str(y_data_train) + ' '
        counter = 1
        for i in x_data_train:
            data = data + str(counter) + ':' + str(i) + ' '
            counter = counter + 1
        return data+'\n'

    def write_data_file(self, f_file, x_data_train, y_data_train):
        data = ''
        for x_data, y_data in zip(x_data_train, y_data_train):
            data = data+self.gen_str_line(x_data, y_data)
        f_file.write(data)

    def write_test_data_file(self, f_file, x_data_train, y_data_train):
        data = ''
        for x_data, y_data in zip(x_data_train, y_data_train):
            data = data+self.gen_str_line(x_data, y_data)
        f_file.write(data)

    def write_libsvm_y(self, f_file, y_data):
        for i in y_data:
            f_file.write(str(i)+'\n')

    def create_data_file(self):
        for rate in test_size:
            for data_set_name in datasets:
                print '{} {}'.format(rate, data_set_name)
                x, y = self.sd.loadAll(data_set_name)
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=rate, random_state=0)
                obj_test_x = '{}/obj/test_x_{}_{}'.format(base_file_test_path, data_set_name, rate)
                obj_test_y = '{}/obj/test_y_{}_{}'.format(base_file_test_path, data_set_name, rate)
                pickle.dump(x_test, open(obj_test_x, 'wb'))
                pickle.dump(y_test, open(obj_test_y, 'wb'))

                libsvm_x = '{}/libsvm/test_x_{}_{}'.format(base_file_test_path, data_set_name, rate)
                libsvm_y = '{}/libsvm/test_y_{}_{}'.format(base_file_test_path, data_set_name, rate)

                f_libsvm_test_x = open(libsvm_x, 'w')
                self.write_data_file(f_libsvm_test_x, x_test, y_test)
                f_libsvm_test_x.close()

                f_libsvm_test_y = open(libsvm_y, 'w')
                self.write_libsvm_y(f_libsvm_test_y, y_test)
                f_libsvm_test_y.close()

                kf = KFold(len(y_train), n_folds=5)
                k_counter = 1
                for train, test in kf:
                    print 'write k ',k_counter
                    kx_train, ky_train = x_train[train], y_train[train]
                    kx_test, ky_test = x_train[test], y_train[test]

                    #write libsvm path
                    path_libsvm_train = '{}/libsvm/train_k{}_{}_{}'.format(base_file_train_path, k_counter, data_set_name,rate)
                    path_libsvm_test = '{}/libsvm/test_k{}_{}_{}'.format(base_file_train_path, k_counter, data_set_name, rate)
                    f_train = open(path_libsvm_train, 'w')
                    f_test = open(path_libsvm_test, 'w')

                    self.write_data_file(f_train, kx_train, ky_train)
                    self.write_data_file(f_test, kx_test, ky_test)

                    f_train.close()
                    f_test.close()

                    #write obj path
                    path_obj_train_x = '{}/obj/train_k{}_x_{}_{}'.format(base_file_train_path, k_counter, data_set_name, rate)
                    path_obj_train_y = '{}/obj/train_k{}_y_{}_{}'.format(base_file_train_path, k_counter, data_set_name, rate)
                    path_obj_test_x = '{}/obj/test_k{}_x_{}_{}'.format(base_file_train_path, k_counter, data_set_name, rate)
                    path_obj_test_y = '{}/obj/test_k{}_y_{}_{}'.format(base_file_train_path, k_counter, data_set_name, rate)

                    pickle.dump(x_train, open(path_obj_train_x,'wb'))
                    pickle.dump(y_train, open(path_obj_train_y, 'wb'))
                    pickle.dump(x_test, open(path_obj_test_x, 'wb'))
                    pickle.dump(y_test, open(path_obj_test_y, 'wb'))

                    k_counter = k_counter+1


if __name__ == '__main__':
    start = time.time()
    cml = Cml()
    cml.process_cmp_libsvm()
    total = (time.time()-start)/60.0
    print 'Total time {} m'.format(total)
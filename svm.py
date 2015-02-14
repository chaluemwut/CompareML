import os
import numpy as np
libsvm_path = '/home/off/libsvm-3.20'
data_file = 'datafile/result/model/svm_kernel{}_degree{}_rate{}_dataset{}_k{}'

class LibSVMWrapper(object):

    def __gen_str_line(self, x_data_train, y_data_train):
        data = str(y_data_train) + ' '
        counter = 1
        for i in x_data_train:
            data = data + str(counter) + ':' + str(i) + ' '
            counter = counter + 1
        return data+'\n'

    def __write_data_file(self, f_file, x_data_train, y_data_train):
        data = ''
        for x_data, y_data in zip(x_data_train, y_data_train):
            data = data+self.__gen_str_line(x_data, y_data)
        f_file.write(data)

    def __read_result(self):
        lst = []
        with open(self.path_result, 'r') as f:
            for line in f:
                lst.append(int(line[:-1]))
        return lst

    def __init__(self, kernel=None, degree=None, rate=None, data_set_name=None):
        self.kernel = kernel
        self.degree = degree
        self.rate = rate
        self.data_set_name = data_set_name
        lst = ['a','b','c','d','e','f','g','h','i']
        np.random.shuffle(lst)
        random = (''.join(lst))+str(np.random.random())
        self.path_model_file = '/tmp/{}'.format(random)
        self.path_test_data = '/tmp/{}.test'.format(random)
        self.path_result = '/tmp/{}.result'.format(random)

    def fit(self, x, y, k):
        self.path_model_result = data_file.format(self.kernel,
                                                  self.degree,
                                                  self.rate,
                                                  self.data_set_name,
                                                  k)
        f_file = open(self.path_model_file, 'w')
        self.__write_data_file(f_file, x, y)
        create_model = libsvm_path+'/svm-train'
        if self.kernel != None:
            create_model = create_model+' -t '+str(self.kernel)
        if self.degree != None:
            create_model = create_model+' -d '+str(self.degree)
        create_model = create_model+' {} {}'.format(self.path_model_file,
                                                    self.path_model_result)
        os.system(create_model)


    def predict(self, x):
        f_result = open(self.path_test_data, 'w')
        self.__write_data_file(f_result, x, [0]*len(x))
        f_result.close()
        create_predict = libsvm_path+'/svm-predict'+' {} {} {}'.format(self.path_test_data,
                                                                    self.path_model_result,
                                                                    self.path_result)
        os.system(create_predict)
        return self.__read_result()

if __name__ == '__main__':
    import pickle
    from sklearn.metrics import accuracy_score
    x_train = pickle.load(open('datafile/train/obj/train_k1_x_adult_0.75','rb'))
    y_train = pickle.load(open('datafile/train/obj/train_k1_y_adult_0.75', 'rb'))
    x_test = pickle.load(open('datafile/train/obj/test_k1_x_adult_0.75', 'rb'))
    y_test = pickle.load(open('datafile/train/obj/test_k1_y_adult_0.75','rb'))
    lst = [LibSVMWrapper(kernel=0),
           LibSVMWrapper(kernel=1, degree=2),
           LibSVMWrapper(kernel=1, degree=3),
           LibSVMWrapper(kernel=2),
           LibSVMWrapper(kernel=3)
           ]
    for m in lst:
        m.fit(x_train, y_train)
        y_pred = m.predict(x_test)
        print accuracy_score(y_test, y_pred)





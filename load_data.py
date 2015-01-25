import numpy as np
from sklearn.cross_validation import train_test_split
from numpy import dtype
from sklearn import datasets
import random
import logging

is_tranfer_data = True

logging.basicConfig(level=logging.INFO)

class DataLoader(object):
    
    def __init__(self):
        self.x = np.loadtxt('data/fselect.txt', delimiter=',', dtype=int)
        self.y = np.loadtxt('data/fresult.txt', dtype=int)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.3, random_state=0)
    
    def tranform_y(self, y):
        lst = []
        for i in y:
            if i > 5:
                lst.append(1)
            else:
                lst.append(0)
        return lst
                
    def load_train(self):
        return self.x_train, [self.y_train, self.tranform_y(self.y_train)][is_tranfer_data]
    
    def load_test(self):
        return self.x_test, [self.y_test, self.tranform_y(self.y_test)][is_tranfer_data]

def tranform_y(y):
    lst = []
    for i in y:
        if i > 5:
            lst.append(1)
        else:
            lst.append(0)
    return lst

class MultiDataLoader(object):
    
    def __init__(self):
        self.x = np.loadtxt('data/fselect.txt', delimiter=',', dtype=int)
        self.y = np.loadtxt('data/fresult.txt', dtype=int)
        self.load()
    

    
    def load(self):
#         def template_load(x1, y1):
#             out = []
#             for xo1, yo1 in zip(x1, y1):
#                 out_data = (xo1, yo1)
#                 out.append(out_data)
#             return out
        self.x1, xi, self.y1, yi = train_test_split(self.x, self.y, test_size=0.5, random_state=0)
        self.x2, self.x3, self.y2, self.y3 = train_test_split(xi, yi, test_size=0.5, random_state=0)
#         return template_load(x1, y1), template_load(x2, y2), template_load(x3, y3)
    def train(self):
        return np.array(self.x1, dtype='float'), tranform_y(self.y1)
    
    def test(self):
        return np.array(self.x2, dtype='float'), tranform_y(self.y2)
    
    def validation(self):
        return np.array(self.x3, dtype='float'), tranform_y(self.y3)
 
class IrisLoader(object):
    
    def __init__(self):
        iris = datasets.load_digits()
        self.x = iris.data
        self.y = iris.target
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=0)
    
    def load_train(self):
        return self.x_train, self.y_train
    
    def load_test(self):
        return self.x_test, self.y_test

class SDDataSets(object):
    
    class BaseSD(object):
        def tranform_data(self, name , lst, data):
            data_map = {}
            for index in lst:
                data_map[index] = data[name] == index
            
            for i in range(0, len(lst)):
                data[name][data_map[lst[i]]] = int(i)
        
        def reshape_y(self, y_train):
            ret = y_train.reshape(1, len(y_train))
            return ret[0]
                        
    class ADULT(object):
        
        def tranform_data(self, name , lst):
            data_map = {}
            for index in lst:
                data_map[index] = self.data[name] == index
            
            for i in range(0, len(lst)):
                self.data[name][data_map[lst[i]]] = int(i)
        
        def __init__(self):
            return
            print 'no pass'
            data_headers=['age',
                          'workclass','fnlwgt','education',
                          'education_num',
                          'marital_status',
                          'occupation',
                          'relationship',
                          'race',
                          'sex',
                          'capital_gain',
                          'capital_loss',
                          'hours_per_week',
                          'native_country',
                          'salary']
            self.data=np.array(np.genfromtxt('data/adult.data', dtype=(int, 'S32',int, 'S32',int,'S32','S32','S32','S32','S32',int,int,int,'S32','S32'),
                                        delimiter=',',
                                        autostrip=True,
                                        names=data_headers))
#             self.data = data
#             print self.data
#             print data.shape
            row_idx_to_delete=[]
            for i in range(0,len(self.data)):
                if "?" in self.data[i]:
                    row_idx_to_delete.append(i)
            
            self.data=np.delete(self.data,row_idx_to_delete)
            
            #age
            idx_age_0=self.data['age']<30
            idx_age_1=(self.data['age']>=30) & (self.data['age']<=55)
            idx_age_2=self.data['age']>55                                    
            self.data['age'][idx_age_0]=0
            self.data['age'][idx_age_1]=1
            self.data['age'][idx_age_2]=2
            
            #workclass
            workclass_attributes=["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",\
             "Local-gov", "State-gov", "Without-pay", "Never-worked"]
            self.tranform_data("workclass", workclass_attributes)
            
            #fnlwgt
            idx_wgt_0=self.data['fnlwgt']<=105702
            idx_wgt_1=(self.data['fnlwgt']>=105702) & (self.data['fnlwgt']<=289569) 
            idx_wgt_2=self.data['fnlwgt']>=289569                        
            self.data['fnlwgt'][idx_wgt_0]=0
            self.data['fnlwgt'][idx_wgt_1]=1
            self.data['fnlwgt'][idx_wgt_2]=2
            
            education_attributes=["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc"\
                                  , "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"]
            
            self.tranform_data("education", education_attributes)
    
            a=np.histogram(self.data['education_num'],4)
            means=[(a[1][i]+a[1][i+1])/2.0 for i in range(0,len(a[1])-1)]
            idx_enum_0=self.data['education_num']<=means[0]
            idx_enum_1=(self.data['education_num']>means[0]) & (self.data['education_num']<=means[1]) 
            idx_enum_2=(self.data['education_num']>=means[1]) & (self.data['education_num']<=means[2])
            idx_enum_3=self.data['education_num']>means[2]
            
            
            self.data['education_num'][idx_enum_0]=0
            self.data['education_num'][idx_enum_1]=1
            self.data['education_num'][idx_enum_2]=2
            self.data['education_num'][idx_enum_3]=3
            
            marital_status_attributes=['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed',\
                                       'Married-spouse-absent','Married-AF-spouse']
            self.tranform_data("marital_status", marital_status_attributes)
            
            occupation_attributes=['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',\
                                   'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces']
            self.tranform_data("occupation", occupation_attributes)
            
            relationship_attributes=['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried']
            self.tranform_data("relationship", relationship_attributes)
            
            race_attributes=['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
            self.tranform_data("race", race_attributes)
            
            sex_attributes=['Female', 'Male']
            self.tranform_data("sex", sex_attributes)
            
            a=np.histogram(self.data['capital_gain'],2)
            means=[(a[1][i]+a[1][i+1])/2.0 for i in range(0,len(a[1])-1)]
            
            idx_cap_gain_0=self.data['capital_gain']<=means[0]
            idx_cap_gain_1=(self.data['capital_gain']>means[0]) & (self.data['capital_gain']<=means[1]) 


            self.data['capital_gain'][idx_cap_gain_0]=0
            self.data['capital_gain'][idx_cap_gain_1]=1
            
            a=np.histogram(self.data['capital_loss'],2)
            means=[(a[1][i]+a[1][i+1])/2.0 for i in range(0,len(a[1])-1)]
            idx_cap_loss_0=self.data['capital_loss']<=means[0]
            idx_cap_loss_1=(self.data['capital_loss']>means[0]) & (self.data['capital_loss']<=means[1]) 

            self.data['capital_loss'][idx_cap_loss_0]=0
            self.data['capital_loss'][idx_cap_loss_1]=1
            
            a=np.histogram(self.data['hours_per_week'],4)
            means=[(a[1][i]+a[1][i+1])/2.0 for i in range(0,len(a[1])-1)]
            
            idx_hours_0=self.data['hours_per_week']<=means[0]
            idx_hours_1=(self.data['hours_per_week']>means[0]) & (self.data['hours_per_week']<=means[1]) 
            idx_hours_2=(self.data['hours_per_week']>=means[1]) & (self.data['hours_per_week']<=means[2])
            idx_hours_3=self.data['hours_per_week']>means[2]
            
            
            self.data['hours_per_week'][idx_hours_0]=0
            self.data['hours_per_week'][idx_hours_1]=1
            self.data['hours_per_week'][idx_hours_2]=2
            self.data['hours_per_week'][idx_hours_3]=3
            
            native_country_attributes=['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)'\
                                       ,'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam'
                                       , 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala',
                                       'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']
            self.tranform_data("native_country", native_country_attributes)
            
            salary_attributes=['<=50K','>50K']
            self.tranform_data("salary", salary_attributes)
            np.savetxt('data/filtered.txt',self.data,fmt="%s",delimiter=',')

        def load(self):
            filtered_data=np.array(np.genfromtxt('data/t_adult.data', dtype='int', delimiter=',',autostrip=True))
            random.shuffle(filtered_data)
            filtered_data[filtered_data[:,14]!=1, 14] = 0
            y_train = filtered_data[:,[14]]
            return filtered_data[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13]], y_train.reshape(1, len(y_train))[0]
                    
    class CovType(BaseSD):
        def __init__(self):
            self.dataset = np.array(np.genfromtxt('data/t_covtype2.data', dtype='int', delimiter=',',autostrip=True))
            logging.debug(self.dataset)
            random.shuffle(self.dataset)
            logging.debug(self.dataset)

        def load_x(self):
            return self.dataset[:,list(range(0,54))]
            
        def load_y(self):
            return self.reshape_y(self.dataset[:,54])

        def load(self):
            # return self.dataset
            return self.dataset[:,list(range(0,54))], self.reshape_y(self.dataset[:,54])
        
        def load_test(self):
            self.dataset = np.genfromtxt('data/test/covtype.data', delimiter=',', dtype=int)
            train_index = list(range(0,53))
            self.dataset[self.dataset[:,54]==-1, 54] = 0
            y_train = self.dataset[:,[54]]
            return self.dataset[:, train_index], y_train.reshape(1, len(y_train))[0] 
    
    class Letter(BaseSD):
        def __init__(self):
            header = ['letter','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16']
            self.dataset = np.genfromtxt('data/letter.data', delimiter=',', 
                                         dtype=('S32', int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int),
                                         names=header)
            
        def save(self):
            import string
            lst = list(string.ascii_uppercase)
            self.tranform_data('letter', lst, self.dataset)
            np.savetxt('data/t_letter.data', self.dataset,fmt="%s",delimiter=',')
            return self.dataset
        
        def load(self):
            obj = np.array(np.genfromtxt('data/t_letter.data', dtype='int', delimiter=',',autostrip=True))
            lst = list(range(1,16))
            return obj[:,lst], self.reshape_y(obj[:,[0]])

    class LetterP(BaseSD):

        def __init__(self, letter_name):
            self.name = letter_name
            self.data = np.loadtxt('data/'+self.name, dtype=int, delimiter=',')
            random.shuffle(self.data)

        def load(self):
            lst = list(range(1,16))
            self.data[self.data[:,0]==-1,0]=0
            return self.data[:,lst], self.reshape_y(self.data[:,[0]])

    class FBCredibility(BaseSD):

        def __init__(self):
            self.x = np.loadtxt('data/fbcredibility/training.data', dtype=int, delimiter=',')
            self.y = np.loadtxt('data/fbcredibility/result.data', dtype=int, delimiter=',')

        def load(self):
            return self.x, self.y

    # def __init__(self):
    #     pass
    #
    # def load_train(self):
    #     pass
    #
    # def load_test(self):
    #     pass
    #

    def split_data(self, x, y, size=0):
        size = [size+5001, len(y)][size==0]
        x_train, y_train = x[0:5000], y[0:5000]
        x_test, y_test = x[5000:size], y[5000:size]
        return x_train, y_train, x_test, y_test

    def load(self, dataset_name):
        if 'adult' == dataset_name:
            adult = self.ADULT()
            x, y = adult.load()
            return self.split_data(x, y)
        elif 'iris' == dataset_name:
            from sklearn import datasets
            iris = datasets.load_iris()
            x_train, y_train = iris.data[0:100], iris.target[0:100]
            x_test, y_test = iris.data[100:150], iris.target[100:150]
            return x_train, y_train, x_test, y_test
        elif 'cov_type' == dataset_name:
            cov_obj = self.CovType()
            x,y = cov_obj.load()
            return self.split_data(x, y)
        elif dataset_name in ['letter.p1', 'letter.p2']:
            letter = self.LetterP(dataset_name)
            x, y = letter.load()
            return self.split_data(x, y)
        elif 'letter' == dataset_name:
            letter_obj = self.Letter()
            return letter_obj.load()
        elif 'fbcredibility' == dataset_name:
            fb_load = self.FBCredibility()
            x, y = fb_load.load()
            size = len(y)
            return x[0:600], y[0:600], x[600:size], y[600:size]

    
if __name__ == '__main__':
    sdDataSet = SDDataSets()
    print sdDataSet.load('letter.p1')[1][11]
#     loader = DataLoader()
#     x_train, y_train = loader.load_test()
#     print x_train

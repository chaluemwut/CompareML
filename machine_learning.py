from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree

from pystruct.models import DirectionalGridCRF
import pystruct.learners as ssvm

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import BernoulliRBM
from sklearn import linear_model
from sklearn.pipeline import Pipeline

class BaseML(object):
    pass

class MLDeepLearning(BaseML):
    pass

class MLNeuralNetwork(BaseML):
    logistic = linear_model.LogisticRegression()
    rbm = BernoulliRBM(n_components=2)
    classifier = Pipeline(steps=[('rbm',rbm),('logistic', logistic)])
    
    def __init__(self, x_train, y_train):
        self.classifier.fit(x_train, y_train)
        
    def predict(self, x_test):
        return self.classifier.predict(x_test)
    
    def __str__(self, *args, **kwargs):
        return "neural network"
        
class MLSVM(BaseML):
    clf = SVC()
    def __init__(self, x_train, y_train):
        self.clf = self.clf.fit(x_train, y_train)
    
    def predict(self, x_test):
        return self.clf.predict(x_test)
    
    def __str__(self, *args, **kwargs):
        return "svm"
    
class MLDecisionTree(BaseML):
    clf = tree.DecisionTreeClassifier()
    
    def __init__(self, x_train, y_train):
        self.clf = self.clf.fit(x_train, y_train)
    
    def predict(self, x_test):
        return self.clf.predict(x_test)
    
    def __str__(self, *args, **kwargs):
        return "decision tree"

class MLKNN(BaseML):
    clf = KNeighborsClassifier()
    
    def __init__(self, x_train, y_train):
        self.clf.fit(x_train, y_train) 
    
    def predict(self, x_test):
        return self.clf.predict(x_test)
    
    def __str__(self, *args, **kwargs):
        return "k-NN"
    
class MLCRF(BaseML):
    clf = DirectionalGridCRF(inference_method='gpbo', neighborhood=4) 
    
    def __init__(self, x_train, y_train):
        self.clf = ssvm.OneSlackSSVM(model=self.clf, C=1, n_jobs=-1, inference_cache=100, tol=.1,
                        show_loss_every=10)
        self.clf.fit(x_train, y_train)
        
    def predict(self, x_test):
        return self.clf.predict(x_test)
    
class MLGaussianNaiveBayes(BaseML):
    clf = GaussianNB()
    
    def __init__(self, x_train, y_train):
        self.clf = self.clf.fit(x_train, y_train)
    
    def predict(self, x_test):
        return self.clf.predict(x_test)
    
    def __str__(self, *args, **kwargs):
        return "Naive Bayes"
    
class MLGeneticAlgorithms(BaseML):
    pass

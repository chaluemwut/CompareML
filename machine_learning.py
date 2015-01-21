from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree

from pystruct.models import DirectionalGridCRF
import pystruct.learners as ssvm

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import BernoulliRBM
from sklearn import linear_model
from sklearn.pipeline import Pipeline

from sklearn.linear_model import Perceptron
from sklearn.ensemble.forest import RandomForestClassifier

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

class LinearNeuralNetwork(BaseML):
    clf = Perceptron()
    def __init__(self, x_train, y_train):
        self.clf.fit_transform(x_train, y_train)
    
    def predict(self, x_test):
        return self.clf.predict(x_test)

    def __str__(self, *args, **kwargs):
        return "Linear NN"
        
# ok
class MLSVM(BaseML):
    clf = SVC()
    def __init__(self, x_train, y_train):
        self.clf = self.clf.fit(x_train, y_train)
    
    def predict(self, x_test):
        return self.clf.predict(x_test)
    
    def __str__(self, *args, **kwargs):
        return "svm"

class MLSVMKernel(BaseML):
    
    def __init__(self, x_train, y_train, kernel):
        self.clf = SVC(kernel=kernel)
        self.clf = self.clf.fit(x_train, y_train)
    
    def predict(self, x_test):
        return self.clf.predict(x_test)
    
    def __str__(self, *args, **kwargs):
        return "svm kernel"
 
# ok   
class MLDecisionTree(BaseML):
    clf = tree.DecisionTreeClassifier()
    
    def __init__(self, x_train, y_train):
        self.clf = self.clf.fit(x_train, y_train)
    
    def predict(self, x_test):
        return self.clf.predict(x_test)
    
    def __str__(self, *args, **kwargs):
        return "decision tree"

# ok
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

# ok    
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

class MLRandomForest(BaseML):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier()
    
    def __init__(self):
        pass
    
    def fit(self, x_train, y_train):
        self.clf = self.clf.fit(x_train, y_train)
    
    def predict(self, x_test):
        return self.clf.predict(x_test)
    
    def __str__(self, *args, **kwargs):
        return "RF"

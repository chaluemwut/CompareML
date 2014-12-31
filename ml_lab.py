from load_data import *
from sklearn import tree
from sklearn.externals.six import StringIO
from sklearn.metrics import *
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import pylab as pl
import pydot

import theano
import theano.tensor as T
import numpy as np

class LabTheano(object):
    def __init__(self):
        pass
    
    def test_scalar(self):
        x = T.scalar('x')
        y = T.scalar('y')
        f = theano.function([x,y], x+y)
        print f(3,5)
    
    def test_vector(self):
        x = T.vector('x')
        y = T.vector('y')
        f = theano.function([x, y], x*y)
        print f(np.array([2,3,4]), np.array([0.2,0.2,0.2]))
    
    def test_matrix(self):
        x = T.matrix('x')
        y = T.matrix('y')
        f = theano.function([x, y], x * y)
        print f(np.array([[1, 2, 3, 4], [2, 3, 4, 5]]),
                np.array([[0.2, 0.2, 0.2, 0.2], [0.2, 0.2, 0.2, 0.2]])
                )
    
    def test_share(self):
        x = theano.shared(np.array([[1,2],[2,4]]))
        x_x = x+2
        f = theano.function([], x_x)
#         f()
        print T.sum(x+2)
#         print f()
        
class MultilayerPerceptron(object):
    
    def __init__(self):
        pass
    
class LabBase(object):
    
    def __init__(self):
        self.load = DataLoader()
        self.x_train, self.y_train = self.load.load_train()
        self.x_test, self.y_test = self.load.load_test()
        self.clf = tree.DecisionTreeClassifier()
        self.clf = self.clf.fit(self.x_train, self.y_train)

class ManualNeuralNework(LabBase):
    
    def __init__(self):
        self.x_train = [[0,0],
                        [0,1],
                        [1,0],
                        [1,1]]
        self.y_train = [0,
                        0,
                        0,
                        1]
        self.weight = [0, 0]
        self.learning_rate = 0.1
        self.threshold = 1
    
    def process(self):
        
        while True:
            self.error_count = 0
            for i in range(0, len(self.x_train)):
                weight_sum = self.x_train[i][0]*self.weight[0] + self.x_train[i][1]*self.weight[1]
                output = 0
                if self.threshold == weight_sum :
                    output = 1
                
                self.error = self.y_train[i] - output
                
                if self.error != 0:
                    self.error_count += 1
                
                self.weight[0] += self.learning_rate*self.error*self.x_train[i][0]
                self.weight[1] += self.learning_rate*self.error*self.x_train[i][1]
                print 'loop i ',i,' : ',self.weight
            
            if self.error_count == 0:
                print self.weight
                break
    
class LabkNN(LabBase):
    
    def __init__(self):
        super(LabkNN, self).__init__()

    def test(self):
        from sklearn.neighbors import NearestNeighbors
        clf = NearestNeighbors()
        clf.fit(self.x_train)
#         print clf.predict(self.x_test)
        print clf.kneighbors(self.x_test)[0]
        print clf.kneighbors(self.x_test, return_distance=False)[0]
#         for x in self.x_test:
#             print clf.kneighbors(self.x_test)
        
class LabDecsionTree(object):
    
    def __init__(self):
        super(LabkNN, self).__init__()
#         super.__init__()
#         self.load = DataLoader()
#         self.x_train, self.y_train = self.load.load_train()
#         self.x_test, self.y_test = self.load.load_test()
#         self.clf = tree.DecisionTreeClassifier()
#         self.clf = self.clf.fit(self.x_train, self.y_train)
            
    def create_tree(self):
#         print clf
        dot_data = StringIO()
        tree.export_graphviz(self.clf, out_file=dot_data)
        graph = pydot.graph_from_dot_data(dot_data.getvalue())
        graph.write_pdf("fb.pdf")
    
    def plot3D(self):
        pca = PCA(n_components=3)
        new_x = pca.fit(self.x_train).transform(self.x_train)
        fig = pl.figure()
        ax = Axes3D(fig)        
        for i in range(0,100):
            y = self.y_train[i]
            l_color = ['red','green'][y==1]
            ax.scatter3D(new_x[i,0],new_x[i,1],new_x[i,2], color=l_color)
        pl.show()
#         fig = pl.figure()
#         ax = Axes3D(fig)
#         ax.scatter3D(new_x[:,0],new_x[:,1],new_x[:,2])
#         pl.show()
                
    def test(self):
        print self.clf
        y = self.clf.predict(self.x_test)
        print accuracy_score(self.y_test, y)
        pl.plot(y)
        pl.plot(self.y_test)
        pl.show()

if __name__ == "__main__":
    lab = LabTheano()
    lab.test_share()  

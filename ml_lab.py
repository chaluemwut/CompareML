from load_data import *
from sklearn import tree
from sklearn.externals.six import StringIO
from sklearn.metrics import *
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import pylab as pl
import pydot

class LabDecsionTree(object):
    
    def __init__(self):
        self.load = DataLoader()
        self.x_train, self.y_train = self.load.load_train()
        self.x_test, self.y_test = self.load.load_test()
        self.clf = tree.DecisionTreeClassifier()
        self.clf = self.clf.fit(self.x_train, self.y_train)
            
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


lab = LabDecsionTree()
lab.plot3D()
            
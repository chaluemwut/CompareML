from sklearn.neighbors import  KNeighborsClassifier
x = [[1,2,3,3],[2,3,4,5]]
y = [0,1]
x_test = [12,3,4,5]
cls = KNeighborsClassifier(n_neighbors=1)
cls.fit(x,y)
print cls.predict(x_test)
from load_data import SDDataSets
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC

def test_svm():
    test_rate = [.75, .50, .25]
    datasets = ['adult', 'cov_type', 'letter.p1', 'letter.p2']
    sd = SDDataSets()
    svm_lst = []
    # svm_lst.append(SVC(kernel='linear'))
    svm_lst.append(SVC(kernel='poly', degree=2))
    svm_lst.append(SVC(kernel='poly', degree=3))
    svm_lst.append(SVC(kernel='sigmoid'))

    for rate in test_rate:
        for data_name in datasets:
            x, y = sd.loadAll(data_name)
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=rate, random_state=0)
            for m in svm_lst:
                print m
                m.fit(x_train, y_train)
                y_pred = m.predict(x_test)
                print y_pred

if __name__ == '__main__':
    test_svm()
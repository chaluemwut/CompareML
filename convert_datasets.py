__author__ = 'Chaluemwut'
import numpy as np

def letter_weka():
    import string
    data = np.loadtxt('data/letter.data', dtype='S32', delimiter=',')
    lst_a_m = list(string.uppercase)[0:13]
    data_lst = []
    for i in range(0, len(data)):
        d = data[i]
        if d[0] in lst_a_m:
            d = np.append(d,'am')
        else:
            d = np.append(d,'nz')
        d = d[1:len(d)]
        data[i] = d
    np.savetxt('data/letter.p2.weka', data, fmt='%s', delimiter=',')

if __name__ == '__main__':
    letter_weka()
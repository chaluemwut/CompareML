import numpy as np

def count_class():
	print 'count class'
	data = np.loadtxt('data/covtype.data', dtype=int, delimiter=',')
	# data[data[:,54]==1, 54] = 1
	data[data[:,54]!=1, 54] = -1
	np.savetxt('data/t_covtype.data', data, fmt='%s', delimiter=',')
	# print len([i for i in data if i[54]==-1])
	# print len([i for i in data if i[54]==1])
	# print len([i for i in data if i[54] == 2])
	# type1 = sum(1 for i in data if i[54]==1)
	# counter = 0
	# for i in range(1,8):
	# 	type_count = len([d for d in data if d[54] == i])
	# 	counter += type_count
	# 	print 'type : ',i,' count : ',type_count
	# print 'counter : ',counter

if __name__ == '__main__':
	count_class()
from logistic_sgd import load_data

def test_mlp2():
    datasets = load_data('mnist.pkl.gz')
    train_set_x, train_set_y = datasets[0]
#     valid_set_x, valid_set_y = datasets[1]
#     test_set_x, test_set_y = datasets[2]
    print train_set_x.get_value()[0]
    print 'y : ', train_set_y  

if __name__ == '__main__':
    test_mlp2()
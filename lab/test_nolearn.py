# import the necessary packages
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from nolearn.dbn import DBN
import numpy as np

from load_data import DataLoader
# grab the MNIST dataset (if this is the first time you are running
# this script, this make take a minute -- the 55mb MNIST digit dataset
# will be downloaded)
print "[X] downloading data..."
# dataset = datasets.fetch_mldata("MNIST Original")

# scale the data to the range [0, 1] and then construct the training
# and testing splits
loader = DataLoader()
x_train, y_train = loader.load_train()
x_test, y_test = loader.load_test()
# (trainX, testX, trainY, testY) = train_test_split(
#     dataset.data / 255.0, dataset.target.astype("int0"), test_size = 0.33)

# train the Deep Belief Network with 784 input units (the flattened,
#  28x28 grayscale image), 800 hidden units in the 1st hidden layer,
# 800 hidden nodes in the 2nd hidden layer, and 10 output units (one
# for each possible output classification, which are the digits 1-10)
dbn = DBN(
    [np.array(x_train).shape[1], 800, 800, 10],
    learn_rates = 0.3,
    learn_rate_decays = 0.9,
    epochs = 10,
    verbose = 1)
dbn.fit(np.array(x_train), np.array(y_train))

# compute the predictions for the test data and show a classification
# report
preds = dbn.predict(np.array(x_test))
# print classification_report(testY, preds)
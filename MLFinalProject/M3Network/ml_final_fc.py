# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 16:32:10 2018

Machine Learning Final Project 
2018 summer semester
A solution with monolithic MLP classifier

student: Sijun Li

@author: Lamorak_Lee
"""
import numpy as np
from PIL import Image
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn import preprocessing

data_num = 60000 #The number of figures
test_num = 10000
fig_w = 45       #width of each figure

data = np.fromfile("mnist\mnist_train\mnist_train_data",dtype=np.uint8)
label = np.fromfile("mnist\mnist_train\mnist_train_label",dtype=np.uint8)
test_data = np.fromfile("mnist\mnist_test\mnist_test_data",dtype=np.uint8)
test_label = np.fromfile("mnist\mnist_test\mnist_test_label",dtype=np.uint8)

#reshape the matrix
data = data.reshape(data_num,fig_w*fig_w)
label = label.reshape(data_num,1)
test_data = test_data.reshape(test_num,fig_w*fig_w)
test_label = test_label.reshape(test_num,1)

#n_layer can be 4,5,6,7,8,9
clf = MLPClassifier(activation='tanh', hidden_layer_sizes=(512,5))


def main():
    x,y = data, label
    for i in range(1):
        clf.fit(x,y.ravel())
    pred = clf.predict(test_data)
    train_pred = clf.predict(data)


    print (pred)
    print ('#Accuracy score: ',metrics.accuracy_score(test_label, pred))
    print ('#Training accuracy score: ',metrics.accuracy_score(label, train_pred))


if __name__ == '__main__':
	main()

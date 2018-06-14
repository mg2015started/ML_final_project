# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 16:32:10 2018

Machine Learning Final Project in 2018 summer semester
A solution with min-max modular network

student:Sijun Li

@author: Lamorak_Lee
"""
import numpy as np
from PIL import Image
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn import metrics
from sklearn import preprocessing
import multiprocessing as mp
import time

data_num = 60000 #The number of figures
test_num = 10000
fig_w = 45       #width of each figure

data = np.fromfile("mnist\mnist_train\mnist_train_data",dtype=np.uint8)
label = np.fromfile("mnist\mnist_train\mnist_train_label",dtype=np.uint8)
test_data = np.fromfile("mnist\mnist_test\mnist_test_data",dtype=np.uint8)
test_label = np.fromfile("mnist\mnist_test\mnist_test_label",dtype=np.uint8)

#reshape the data matrix
data = preprocessing.scale(data.reshape(data_num,fig_w*fig_w))
label = label.reshape(data_num,1)
test_data = preprocessing.scale(test_data.reshape(test_num,fig_w*fig_w))
test_label = test_label.reshape(test_num,1)

#modify the original to 2-classes classification problem
#return a 10*60000 array,each row represents a 2-class problem
def get_new_label(label,n_samples):
    mod_label = np.zeros((10,n_samples))
    for i in range(n_samples):
        mod_label[label[i][0]][i] = 1
        
    return mod_label

#get the indices of the positive items in each 2-class problem
#return a dict containing the indices of the positive items
def get_pos_idx(label):
    idx = {}
    for i in range(10):
        idx['idx'+str(i)] = []
    for i in range(data_num):
        idx['idx'+str(label[i][0])].append(i)
    return idx

#get the indices of the negative items in each 2-class problem
#return a dict
def get_neg_idx(label):
    idx = {}
    for i in range(10):
        idx['idx'+str(i)] = []
    for i in range(data_num):
        for j in range(10):
            if j!=label[i][0]:
                idx['idx'+str(j)].append(i)
    return idx

#build MLP classifiers for the decomposed problem
m3_net = []
for i in range(100):
    clf = MLPClassifier(hidden_layer_sizes=(512,4),
                       activation='tanh',tol=0.005,
                       random_state=1)
    m3_net.append(clf)
    
#training step
#parameter i indicates the ith 2-class prob,as the positive class
#param j indicates the jth prob,as the negative class
#decomposition on the base of a priori knowledge
def train_step(data,label,i,j):
    print ('Classifier '+str(10*j+i)+' is running')
    m3_net[10*j+i].fit(data,label.ravel())
    return

#testing step
def predict_step(new_test_label):
    res = []
    for i in range(10):
        res_min = []
        for j in range(10):
            if j==i:
                pass
            else:
                print (m3_net[10*j+i].score(test_data,new_test_label[i]))
                res_min.append(m3_net[10*j+i].predict_proba(test_data)[:,1:2].ravel())
        res_min = np.array(res_min)
        #act as a MIN unit
        res_min = res_min.min(axis=0)
        res.append(res_min)
    res = np.array(res)
    #act as a MAX unit
    res = res.argmax(axis=0)
    return res

def main():
    mod_label = get_new_label(label,data_num)
    mod_test_label = get_new_label(test_label,test_num)
    pos_idx = get_pos_idx(label)
    start = time.time()
    #i indicates the positive items,and j indicates the negative
    for i in range(10):
        for j in range(10):
            if j==i:pass
            else:
                #make a batch containing a positive class and a negative class
                idx_pos = np.array(pos_idx['idx'+str(i)])
                idx_neg = np.array(pos_idx['idx'+str(j)])
                data_pos = data[idx_pos,:]
                data_neg = data[idx_neg,:]
                label_pos = (mod_label[i].reshape(data_num,1))[idx_pos,:]
                label_neg = (mod_label[i].reshape(data_num,1))[idx_neg,:]
                batch_data = np.append(data_pos,data_neg,axis=0)
                batch_label = np.append(label_pos,label_neg,axis=0)
               
                train_step(batch_data,batch_label,i,j)

    inter = time.time()
    test_res = predict_step(mod_test_label)
    ed = time.time()

    print ('Test Accuracy: ',np.mean(np.equal(test_res,test_label.ravel())))
    print ('Training time: ',inter-start)
    print ('Testing time: ',ed-inter)
    
if __name__ == '__main__':
	main()
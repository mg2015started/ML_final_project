# Standard scientific Python imports
import matplotlib.pyplot as plt
import numpy as np
from time import time

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, pipeline
from sklearn.kernel_approximation import (RBFSampler, Nystroem)
from sklearn.decomposition import PCA
import sys

from sklearn import svm
import numpy as np
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import time
import datetime as dt
import sys

logfile = "KernalAP.txt"
f = open(logfile,'w')
old = sys.stdout
sys.stdout = f

#read train Part
data_num = 60000 #The number of figures
fig_w = 45       #width of each figure
train_data = np.fromfile("mnist/mnist_train/mnist_train_data",dtype=np.uint8)
train_label = np.fromfile("mnist/mnist_train/mnist_train_label",dtype=np.uint8)
print(train_data.shape)
print(train_label.shape)
#reshape the matrix
train_data = train_data.reshape(data_num,fig_w*fig_w)
print("After reshape:",train_data.shape)

#read test Part
data_num = 10000 #The number of figures
fig_w = 45       #width of each figure
test_data = np.fromfile("mnist/mnist_test/mnist_test_data",dtype=np.uint8)
test_label = np.fromfile("mnist/mnist_test/mnist_test_label",dtype=np.uint8)
print(test_data.shape)
print(test_label.shape)
#reshape the matrix
test_data = test_data.reshape(data_num,fig_w*fig_w)
print("After reshape:",test_data.shape)



# Create a classifier: a support vector classifier
kernel_svm = svm.SVC(gamma=.2,verbose=1)
linear_svm = svm.LinearSVC(verbose=1)

# create pipeline from kernel approximation
# and linear svm
feature_map_fourier = RBFSampler(gamma=.2, random_state=1)
feature_map_nystroem = Nystroem(gamma=.2, random_state=1)

fourier_approx_svm = pipeline.Pipeline([("feature_map", feature_map_fourier),
                                        ("svm", svm.LinearSVC())])

nystroem_approx_svm = pipeline.Pipeline([("feature_map", feature_map_nystroem),
                                        ("svm", svm.LinearSVC())])

# fit and predict using linear and kernel svm:

import datetime as dt
# We learn the digits on train part

kernel_svm_start_time1 = dt.datetime.now()
print ('Start kernel svm learning at {}'.format(str(kernel_svm_start_time1)))
kernel_svm.fit(train_data, train_label)
kernel_svm_end_time1 = dt.datetime.now()
elapsed_time1 = kernel_svm_end_time1 - kernel_svm_start_time1
print ('End kernel svm learning at {}'.format(str(kernel_svm_end_time1)))
print ('Elapsed learning {}'.format(str(elapsed_time1)))

kernel_svm_start_time = dt.datetime.now()
kernel_svm_score = kernel_svm.score(test_data, test_label)
elapsed_time = dt.datetime.now() - kernel_svm_start_time
print ('Prediction takes {}'.format(str(elapsed_time)))

linear_svm_time = dt.datetime.now()
print ('Start linear svm learning at {}'.format(str(linear_svm_time)))
linear_svm.fit(train_data, train_label)
linear_svm_score = linear_svm.score(test_data, test_label)
linear_svm_time = dt.datetime.now() - linear_svm_time
print ('Elapsed learning {}'.format(str(linear_svm_time)))


#aprox sample sizes, used for ploting 
sample_sizes = 30 * np.arange(1, 10)
fourier_scores = []
nystroem_scores = []
fourier_times = []
nystroem_times = []


for D in sample_sizes:
    print(D)
    fourier_approx_svm.set_params(feature_map__n_components=D)
    nystroem_approx_svm.set_params(feature_map__n_components=D)
    start = time()
    nystroem_approx_svm.fit(train_data, train_label)
    nystroem_times.append(time() - start)

    start = time()
    fourier_approx_svm.fit(train_data, train_label)
    fourier_times.append(time() - start)

    fourier_score = fourier_approx_svm.score(test_data, test_label)
    nystroem_score = nystroem_approx_svm.score(test_data, test_label)
    nystroem_scores.append(nystroem_score)
    fourier_scores.append(fourier_score)


print("kernal{}{}{}{}\n".format(kernel_svm_start_time1,kernel_svm_end_time1,elapsed_time1,kernel_svm_score))
print("linear{}{}\n".format(linear_svm_time,linear_svm_score))

print(sample_sizes)
print(nystroem_times)
print(fourier_times)
print("score")
print(nystroem_scores)
print(fourier_scores)
print("finshed")

sys.stdout = old  # 还原系统输出
f.close()

print(open("KernalAP.txt", 'r').read())
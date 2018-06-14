from sklearn import svm
import numpy as np
import numpy as np
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import time
import datetime as dt
import sys

logfile = "Grid.txt"
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


gamma_range = [0.01,0.05, 0.1, 0.5]

C_range =[0.1, 0.5, 1.0, 5, 10]

parameters = {'kernel':['rbf'], 'C':C_range, 'gamma': gamma_range}

svm_clsf = svm.SVC()
grid_clsf = GridSearchCV(estimator=svm_clsf,param_grid=parameters,n_jobs=20, verbose=2)

print("begin fit")
start_time = dt.datetime.now()
grid_clsf.fit(train_data, train_label)
elapsed_time= dt.datetime.now() - start_time
print('Elapsed time, param searching {}'.format(str(elapsed_time)))

classifier = grid_clsf.best_estimator_
params = grid_clsf.best_params_

print("cvresults")
print(grid_clsf.cv_results_)

print("bestparams")
print(params)

scores = grid_clsf.cv_results_['mean_test_score'].reshape(len(C_range),
                                                     len(gamma_range))

print(scores)

predictLabel = classifier.predict(test_data)

print(metrics.classification_report(test_label, predictLabel))
print(metrics.accuracy_score(test_label, predictLabel))

sys.stdout = old  # 还原系统输出
f.close()

print(open("Grid.txt", 'r').read())
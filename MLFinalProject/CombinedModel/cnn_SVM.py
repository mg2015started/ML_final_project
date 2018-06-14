import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import svm
import time
from sklearn import metrics
from tools import *


NUMBER_OF_FEATURES = 256
NUMBER_OF_EPOCHS = 3
NUMBER_OF_EXPERIMENTS = 1

#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
converter = np.array([0,1,2,3,4,5,6,7,8,9])

batch_size = 100

svm_results = {
    "LK-SVM-ACCU":0, "GK-SVM-ACCU":0, "LK-SVM-TIME":0, "GK-SVM-TIME":0,}
experiment_results = {
    "ConvNet-ACCU":0, "ConvNetSVM-ACCU":0,
   "ConvNet-TIME":0, "ConvNetSVM-TIME":0,}


def compute_accuracy():
    global accuracy
    total_acc = 0
    data_len = 0
    batch_test = next_batch_test_cut(batch_size)
    for v_x, v_y in batch_test:
        batch_len = len(v_x)
        result = sess.run(model_accuracy, feed_dict={x: v_x, y_: v_y,keep_prob:1.0})
        total_acc += result * batch_len
        data_len += batch_len

    return total_acc/data_len

def print_debug(ndarrayinput, stringinput):
    print("\n"+stringinput)
    print(ndarrayinput.shape)
    print(type(ndarrayinput))
    print(np.mean(ndarrayinput))
    print(ndarrayinput)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def SVM(krnl):
    print("\n###############################\n", krnl, "Kernel SVM Train/Test\n###############################")
#concat
    # read train Part
    data_num = 60000  # The number of figures
    fig_w = 45  # width of each figure
    train_data = np.fromfile("mnist/mnist_train/mnist_train_data", dtype=np.uint8)
    train_label = np.fromfile("mnist/mnist_train/mnist_train_label", dtype=np.uint8)
    # reshape the matrix
    train_data = train_data.reshape(data_num, fig_w * fig_w)
    train_data = train_data / 255.0

    # read test Part
    data_num = 10000  # The number of figures
    fig_w = 45  # width of each figure
    test_data = np.fromfile("mnist/mnist_test/mnist_test_data", dtype=np.uint8)
    test_label = np.fromfile("mnist/mnist_test/mnist_test_label", dtype=np.uint8)
    # reshape the matrix
    test_data = test_data.reshape(data_num, fig_w * fig_w)
    test_data = test_data / 255.0

    print("readfinished")


    initial_time = time.time()

    clf = svm.SVC(kernel=krnl)
    clf.fit(train_data, train_label)
    training_time = time.time()-initial_time
    print("\nTraining Time = ", training_time)

    accuracy = clf.score(test_data, test_label)
#    test_time = time.time() - (training_time + initial_time)
#    print("\nTest Time = ", test_time)

    print("\n", krnl, "kernel SVM accuracy =", accuracy)

    print("begin predict")
    initial_time = time.time()
    predictLabel = clf.predict(test_data)
    training_time = time.time() - initial_time
    print("\nTesting Time = ", training_time)

    print(metrics.classification_report(test_label, predictLabel))
    print(metrics.accuracy_score(test_label, predictLabel))

    return accuracy, training_time


def ConvNet(number_of_training_epochs):
    print("\n#########################\nConvNet Train/Test\n#########################")
    initial_time = time.time()

    epoch = 16
    batch = 0
    display_batch = 600

    batch_l = []
    train_loss_l = []
    train_acc_l = []
    test_acc_l = []
    svm_acc_l = []

    for ep in range(epoch):
        print("_" * 25)
        print("epoch:{}".format(ep))
        batch_train = next_batch_train_cut(batch_size)
        for batch_x, batch_y in batch_train:
            if batch % display_batch == 0:
                loss, acc = sess.run([cross_entropy, model_accuracy], feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0})
                tacc = compute_accuracy()
                batch_l.append(batch)
                train_loss_l.append(loss)
                train_acc_l.append(acc)
                test_acc_l.append(tacc)
                print("batch:{}, train loss:{}, train accuracy:{}, test accuracy:{}".format(batch, loss, acc, tacc))
            if batch % 1800 == 0:
                sacc = ConvNetSVM()
                svm_acc_l.append(sacc)
            sess.run(train_step, feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
            batch = batch + 1

    training_time = time.time()-initial_time
    print("\nTraining Time = ", training_time)

    tacc = compute_accuracy()
#    test_time = time.time() - (training_time + initial_time)
#    print("\nTest Time = ", test_time)

    print("\nConvNet accuracy =", tacc)

    bunch = Bunch(batch_l=batch_l, train_loss_l=train_loss_l, train_acc_l=train_acc_l, test_acc_l=test_acc_l, svm_acc_l=svm_acc_l)
    writebunchobj("conv.dat", bunch)

    return tacc, training_time

def ConvNetSVM():
    print("\n#########################\nConvNetSVM Train/Test\n#########################")
    initial_time = time.time()

    # read train Part
    data_num = 60000  # The number of figures
    fig_w = 45  # width of each figure
    train_data = np.fromfile("mnist/mnist_train/mnist_train_data", dtype=np.uint8)
    train_label = np.fromfile("mnist/mnist_train/mnist_train_label", dtype=np.uint8)
    # reshape the matrix
    train_data = train_data.reshape(data_num, fig_w * fig_w)
    train_data = train_data / 255.0

    num_batch = int((data_num - 1) / batch_size) + 1
    rst = []

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_num)
        c = train_data[start_id:end_id]
        l = train_label[start_id:end_id]
        tpa = sess.run(h_fc1, feed_dict={x: c})
        rst.append(tpa)

    train_features_cnn = np.concatenate(tuple(rst))
    print(train_features_cnn.shape)

    # read test Part
    data_num = 10000  # The number of figures
    fig_w = 45  # width of each figure
    test_data = np.fromfile("mnist/mnist_test/mnist_test_data", dtype=np.uint8)
    test_label = np.fromfile("mnist/mnist_test/mnist_test_label", dtype=np.uint8)
    # reshape the matrix
    test_data = test_data.reshape(data_num, fig_w * fig_w)
    test_data = test_data / 255.0

    num_batch = int((data_num - 1) / batch_size) + 1
    rst = []

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_num)
        c = test_data[start_id:end_id]
        l = test_label[start_id:end_id]
        tpa = sess.run(h_fc1, feed_dict={x: c})
        rst.append(tpa)

    test_features_cnn = np.concatenate(tuple(rst))
    print(test_features_cnn.shape)

    clf = svm.SVC(verbose=2)
    clf.fit(train_features_cnn, train_label)
    training_time = time.time()-initial_time
    print("\nTraining Time = ", training_time)

    accuracy = clf.score(test_features_cnn, test_label)

    print("\nConvNetSVM accuracy =", accuracy)

    print("begin predict")
    initial_time = time.time()
    predictLabel = clf.predict(test_features_cnn)
    training_time = time.time() - initial_time
    print("\nTesting Time = ", training_time)

    print(metrics.classification_report(test_label, predictLabel))
    print(metrics.accuracy_score(test_label, predictLabel))

    return accuracy

print("\n#########################\nStarting\n#########################\n")

sess = tf.InteractiveSession()

print("\n#########################\nBuilding ConvNet\n#########################")

x = tf.placeholder(tf.float32, shape=[None, 2025])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

x_image = tf.reshape(x, [-1,45,45,1])

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

h_pool2_flat = tf.reshape(h_pool2, [-1, 12*12*64])

W_fc1 = weight_variable([12 * 12 * 64, NUMBER_OF_FEATURES])
b_fc1 = bias_variable([NUMBER_OF_FEATURES])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([NUMBER_OF_FEATURES, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
model_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#sess.run(tf.initialize_all_variables())

print("\n#########################\nExecuting Experiments\n#########################")

dataframe_svm = pd.DataFrame()
dataframe_results = pd.DataFrame()

svm_results["LK-SVM-ACCU"], svm_results["LK-SVM-TIME"] = SVM("linear")
svm_results["GK-SVM-ACCU"], svm_results["GK-SVM-TIME"] = SVM("rbf")

dataframe_svm = dataframe_svm.append(svm_results, ignore_index=True)
dataframe_svm = dataframe_svm[["LK-SVM-ACCU", "GK-SVM-ACCU", "LK-SVM-TIME", "GK-SVM-TIME"]]

for index in range(NUMBER_OF_EXPERIMENTS):
    print("\n#########################\nExperiment", index+1, "of", NUMBER_OF_EXPERIMENTS, "\n#########################")
    sess.run(tf.global_variables_initializer())
    experiment_results["ConvNet-ACCU"], experiment_results["ConvNet-TIME"] = ConvNet(NUMBER_OF_EPOCHS)

dataframe_results = dataframe_results[["ConvNet-ACCU", "ConvNetSVM-ACCU",
                       "ConvNet-TIME", "ConvNetSVM-TIME",]]

print("\n#########################\nPrinting Results\n#########################\n")

print("\n", dataframe_svm)
print("\n", dataframe_results, "\n")
print(dataframe_results.describe())

print("\n#########################\nStoping\n#########################\n")

sess.close()


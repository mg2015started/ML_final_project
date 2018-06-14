import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pickle
from sklearn.datasets.base import Bunch

def writebunchobj(path, bunchobj):
    with open(path, "wb") as file_obj:
        pickle.dump(bunchobj, file_obj)

# 读取bunch对象
def readbunchobj(path):
    with open(path, "rb") as file_obj:
        bunch = pickle.load(file_obj)
    return bunch

def getpic():
    # read test Part
    data_num = 10000  # The number of figures
    fig_w = 45  # width of each figure
    test_data = np.fromfile("mnist/mnist_test/mnist_test_data", dtype=np.uint8)
    test_label = np.fromfile("mnist/mnist_test/mnist_test_label", dtype=np.uint8)
    # print(test_data.shape)
    # print(test_label.shape)
    # reshape the matrix
    test_data = test_data.reshape(data_num, fig_w * fig_w)
    test_data = test_data / 255.0

    return test_data[100:200]

def next_batch_train(batch_size=64):
    data_num = 60000  # The number of figures
    fig_w = 45  # width of each figure
    train_data = np.fromfile("mnist/mnist_train/mnist_train_data", dtype=np.uint8)
    train_label = np.fromfile("mnist/mnist_train/mnist_train_label", dtype=np.uint8)
    #print(train_data.shape)
    #print(train_label.shape)
    # reshape the matrix
    train_data = train_data.reshape(data_num, fig_w * fig_w)
    train_data = train_data / 255.0
    #print("After reshape:", train_data.shape)

    # onehot
    train_label = OneHotEncoder(sparse=False).fit_transform(train_label.reshape(-1, 1))

    #shuffle
    indices = np.random.permutation(np.arange(data_num))
    train_data = [train_data[i] for i in indices]
    train_label = [train_label[i] for i in indices]

    num_batch = int((data_num - 1) / batch_size) + 1

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_num)
        c = train_data[start_id:end_id]
        l = train_label[start_id:end_id]
        yield c,l


def next_batch_test(batch_size=64):
    # read test Part
    data_num = 10000  # The number of figures
    fig_w = 45  # width of each figure
    test_data = np.fromfile("mnist/mnist_test/mnist_test_data", dtype=np.uint8)
    test_label = np.fromfile("mnist/mnist_test/mnist_test_label", dtype=np.uint8)
    #print(test_data.shape)
    #print(test_label.shape)
    # reshape the matrix
    test_data = test_data.reshape(data_num, fig_w * fig_w)
    test_data = test_data / 255.0
    # onehot
    test_label = OneHotEncoder(sparse=False).fit_transform(test_label.reshape(-1, 1))

    # shuffle
    #indices = np.random.permutation(np.arange(data_num))
    #test_data = [test_data[i] for i in indices]
    #test_label = [test_label[i] for i in indices]

    num_batch = int((data_num - 1) / batch_size) + 1

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_num)
        c = test_data[start_id:end_id]
        l = test_label[start_id:end_id]
        yield c, l

def next_batch_train_cut(batch_size=64):
    data_num = 60000  # The number of figures
    fig_w = 45  # width of each figure
    train_data = np.fromfile("mnist/mnist_train/mnist_train_data", dtype=np.uint8)
    train_label = np.fromfile("mnist/mnist_train/mnist_train_label", dtype=np.uint8)
    #print(train_data.shape)
    #print(train_label.shape)
    # reshape the matrix
    train_data = train_data.reshape(data_num, fig_w * fig_w)
    train_data = train_data / 255.0
    #print("After reshape:", train_data.shape)

    # onehot
    train_label = OneHotEncoder(sparse=False).fit_transform(train_label.reshape(-1, 1))

    #shuffle
    indices = np.random.permutation(np.arange(data_num))
    train_data = [train_data[i] for i in indices]
    train_label = [train_label[i] for i in indices]

    num_batch = int((data_num - 1) / batch_size) + 1

    for i in range(num_batch):
        start_id = i * batch_size
        if ((i+1)*batch_size)>data_num:
            raise StopIteration
        end_id = min((i + 1) * batch_size, data_num)
        c = train_data[start_id:end_id]
        l = train_label[start_id:end_id]
        yield c,l


def next_batch_test_cut(batch_size=64):
    # read test Part
    data_num = 10000  # The number of figures
    fig_w = 45  # width of each figure
    test_data = np.fromfile("mnist/mnist_test/mnist_test_data", dtype=np.uint8)
    test_label = np.fromfile("mnist/mnist_test/mnist_test_label", dtype=np.uint8)
    #print(test_data.shape)
    #print(test_label.shape)
    # reshape the matrix
    test_data = test_data.reshape(data_num, fig_w * fig_w)
    test_data = test_data / 255.0
    # onehot
    test_label = OneHotEncoder(sparse=False).fit_transform(test_label.reshape(-1, 1))

    # shuffle
    #indices = np.random.permutation(np.arange(data_num))
    #test_data = [test_data[i] for i in indices]
    #test_label = [test_label[i] for i in indices]

    num_batch = int((data_num - 1) / batch_size) + 1

    for i in range(num_batch):
        start_id = i * batch_size
        if ((i+1)*batch_size)>data_num:
            raise StopIteration
        end_id = min((i + 1) * batch_size, data_num)
        c = test_data[start_id:end_id]
        l = test_label[start_id:end_id]
        yield c, l


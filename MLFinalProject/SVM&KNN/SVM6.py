from sklearn import svm
import numpy as np
import numpy as np
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import time
import datetime as dt
from PIL import Image
import sys

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError("""usage: python svm.py [c g]""")

    logfile = sys.argv[1]
    f = open(logfile,'w')
    old = sys.stdout
    sys.stdout = f


    # read train Part
    data_num = 60000  # The number of figures
    fig_w = 45  # width of each figure
    train_data = np.fromfile("mnist/mnist_train/mnist_train_data", dtype=np.uint8)
    train_label = np.fromfile("mnist/mnist_train/mnist_train_label", dtype=np.uint8)
    print(train_data.shape)
    print(train_label.shape)
    # reshape the matrix
    train_data = train_data.reshape(data_num, fig_w * fig_w)
    train_data = train_data/255.0
    print("After reshape:", train_data.shape)
    # choose a random index
    ind = np.random.randint(0, data_num)
    # print the index and label
    print("index: ", ind)
    print("label: ", train_label[ind])
    # save the figure
    im = Image.fromarray(train_data[ind].reshape(fig_w, fig_w))
    im.save("example.png")
    dd
    # read test Part
    data_num = 10000  # The number of figures
    fig_w = 45  # width of each figure
    test_data = np.fromfile("mnist/mnist_test/mnist_test_data", dtype=np.uint8)
    test_label = np.fromfile("mnist/mnist_test/mnist_test_label", dtype=np.uint8)
    print(test_data.shape)
    print(test_label.shape)
    # reshape the matrix
    test_data = test_data.reshape(data_num, fig_w * fig_w)
    test_data = test_data/255.0
    print("After reshape:", test_data.shape)
    # choose a random index
    ind = np.random.randint(0, data_num)
    # print the index and label
    print("index: ", ind)
    print("label: ", test_label[ind])
    # save the figure
    # im = Image.fromarray(data[ind])
    # im.save("example.png")

    clf = svm.SVC(kernel='linear',verbose=True)

    print("begin fit")

    start_time = dt.datetime.now()

    clf.fit(train_data, train_label)

    elapsed_time = dt.datetime.now() - start_time
    print('Elapsed time, param fitting {}'.format(str(elapsed_time)))

    print("begin predict")
    start_time = dt.datetime.now()
    print(start_time)

    predictLabel = clf.predict(test_data)

    elapsed_time = dt.datetime.now() - start_time
    print('Elapsed time, predict {}'.format(str(elapsed_time)))

    print(metrics.classification_report(test_label, predictLabel))
    print(metrics.accuracy_score(test_label, predictLabel))

    sys.stdout = old  # 还原系统输出
    f.close()

    print(open(logfile, 'r').read())
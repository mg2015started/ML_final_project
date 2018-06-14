from tools import  *
import matplotlib.pyplot as plt

def drawall():
    path = ["cnn.dat", "lstm.dat", "cnn_rnn.dat"]
    parent_dir = "result/"

    for p in path:
        bunch = readbunchobj(parent_dir + p)
        if (p=="cnn.dat"):
            batch_l = bunch.batch_l[:10]
            train_loss_l = bunch.train_loss_l[:10]
            train_acc_l = bunch.train_acc_l[:10]
            test_acc_l = bunch.test_acc_l[:10]
        else:
            batch_l = bunch.batch_l
            train_loss_l = bunch.train_loss_l
            train_acc_l = bunch.train_acc_l
            test_acc_l = bunch.test_acc_l

        print(max(test_acc_l))
        fig = plt.figure()
        ax1 = fig.add_subplot(2, 2, 1)
        plt.plot(batch_l, train_loss_l, 'o-', label="train loss", markersize=4)
        plt.xlabel("batch num")
        plt.ylabel("train loss")
        plt.title("train loss")
        plt.legend(loc='upper right')
        ax2 = fig.add_subplot(2, 2, 2)
        plt.plot(batch_l, train_acc_l, 'o-', label="train acc", markersize=4)
        plt.xlabel("batch num")
        plt.ylabel("train acc")
        plt.title("train acc")
        plt.legend(loc='upper right')
        ax4 = fig.add_subplot(2, 2, 3)
        plt.plot(batch_l, test_acc_l, 'o-', label="test acc", markersize=4)
        plt.xlabel("batch num")
        plt.ylabel("test acc")
        plt.title("test acc")
        plt.legend(loc='upper right')
        ax4 = fig.add_subplot(2, 2, 4)
        plt.plot(batch_l, train_acc_l, 'o-', label="train acc", markersize=4)
        plt.plot(batch_l, test_acc_l, '*-', label="test acc", markersize=4)
        plt.xlabel("batch num")
        plt.ylabel("acc")
        plt.legend(loc='upper right')
        plt.title("train and test acc")
        plt.show()

def drawPart():
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)

    path = ["cnn.dat", "lstm.dat", "cnn_rnn.dat"]
    parent_dir = "result/"

    name = ["cnn", "lstm", "cnn_lstm"]
    shape = ["o-","*-","d-"]

    mp = dict(zip(path,name))
    co = dict(zip(path,shape))

    for p in path:
        bunch = readbunchobj(parent_dir + p)
        if (p=="cnn.dat"):
            batch_l = bunch.batch_l[:10]
            train_loss_l = bunch.train_loss_l[:10]
            train_acc_l = bunch.train_acc_l[:10]
            test_acc_l = bunch.test_acc_l[:10]
        else:
            batch_l = bunch.batch_l
            train_loss_l = bunch.train_loss_l
            train_acc_l = bunch.train_acc_l
            test_acc_l = bunch.test_acc_l
        plt.plot(batch_l, train_acc_l, co[p], label=mp[p], markersize=4)

    plt.xlabel("batch num")
    plt.ylabel("train acc")
    plt.title("train acc")
    plt.legend(loc='bottom right')

    ax2 = fig.add_subplot(1, 2, 2)
    for p in path:
        bunch = readbunchobj(parent_dir + p)
        if (p=="cnn.dat"):
            batch_l = bunch.batch_l[:10]
            train_loss_l = bunch.train_loss_l[:10]
            train_acc_l = bunch.train_acc_l[:10]
            test_acc_l = bunch.test_acc_l[:10]
        else:
            batch_l = bunch.batch_l
            train_loss_l = bunch.train_loss_l
            train_acc_l = bunch.train_acc_l
            test_acc_l = bunch.test_acc_l

        plt.plot(batch_l, test_acc_l, co[p], label=mp[p], markersize=4)

    plt.xlabel("batch num")
    plt.ylabel("test acc")
    plt.title("test acc")
    plt.legend(loc='bottom right')

    plt.show()

if __name__ == "__main__":
    drawPart()

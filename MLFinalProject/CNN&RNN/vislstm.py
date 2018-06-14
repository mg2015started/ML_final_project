import tensorflow as tf
from tools import *
import matplotlib.pyplot as plt

def compute_accuracy():
    global accuracy
    total_acc = 0
    data_len = 0
    batch_test = next_batch_test_cut(batch_size)
    for v_x, v_y in batch_test:
        batch_len = len(v_x)
        result = sess.run(accuracy, feed_dict={x_image: v_x, y: v_y})
        total_acc += result * batch_len
        data_len += batch_len

    return total_acc / data_len


# parameters init
l_r = 0.001
training_iters = 100000
batch_size = 100

n_inputs = 45
n_steps = 45
n_hidden_units = 128
n_classes = 10

# define placeholder for input
x_image = tf.placeholder(tf.float32, [None, 2025])
x = tf.reshape(x_image, [-1, n_inputs, n_steps])
y = tf.placeholder(tf.float32, [None, n_classes])

# define w and b
weights = {
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}

X = tf.reshape(x, [-1, n_inputs])
X_in = tf.matmul(X, weights['in']) + biases['in']
X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # cell
lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
_init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state, time_major=False)

    # hidden layer for output as the final results
    # results = tf.matmul(states[1], weights['out']) + biases['out']
    # or
outputs1 = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
pred = tf.matmul(outputs1[-1], weights['out']) + biases['out']

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(l_r).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# init session
sess = tf.Session()
# init all variables
sess.run(tf.global_variables_initializer())
# start training

# for i in range(training_iters):
epoch = 15
batch = 0
display_batch = 600

batch_l = []
train_loss_l = []
train_acc_l = []
test_acc_l = []

for ep in range(epoch):
    print("_" * 25)
    print("epoch:{}".format(ep))

    dx = getpic()
    op = sess.run(outputs, feed_dict={x_image: dx})[0]
    op = np.reshape(op, [-1, 128])
    print(op.shape)
    bar_index = range(10)
    for i in range(op.shape[0]):
        plt.subplot(6, 8, i + 1)
        X_h_shate = op[i, :].reshape([-1, n_hidden_units])
        pro = sess.run(tf.nn.softmax(tf.matmul(X_h_shate, weights['out']) + biases['out']))
        plt.bar(bar_index, pro[0], width=0.2, align='center')
        plt.axis('off')
    plt.show()

    batch_train = next_batch_train_cut(batch_size)
    for batch_x, batch_y in batch_train:
        if batch % display_batch == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={x_image: batch_x, y: batch_y})
            tacc = compute_accuracy()
            batch_l.append(batch)
            train_loss_l.append(loss)
            train_acc_l.append(acc)
            test_acc_l.append(tacc)
            print("batch:{}, train loss:{}, train accuracy:{}, test accuracy:{}".format(batch, loss, acc, tacc))
        sess.run(train_op, feed_dict={x_image: batch_x, y: batch_y})
        batch = batch + 1



bunch = Bunch(batch_l=batch_l, train_loss_l=train_loss_l, train_acc_l=train_acc_l, test_acc_l=test_acc_l)
writebunchobj("result/lstm_vis.dat", bunch)

print("finished")
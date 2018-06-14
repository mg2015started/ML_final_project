import tensorflow as tf

from tools import *

lr = 0.001
training_iters = 100000
batch_size = 100
n_input = 144
n_steps = 64
n_hidden_units = 128
n_classes = 10

def compute_accuracy():
    global accuracy
    total_acc = 0
    data_len = 0
    batch_test = next_batch_test_cut(batch_size)
    for v_x, v_y in batch_test:
        batch_len = len(v_x)
        result = sess.run(accuracy, feed_dict={x: v_x, y: v_y, keep_prob:1.0})
        total_acc += result * batch_len
        data_len += batch_len

    return total_acc/data_len


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
    
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # strides=[1,x_movement,y_movement,1]
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def conv_pool_layer(X, img_len, img_hi, out_seq):
    W = weight_variable([img_len, img_len, img_hi, out_seq])
    b = bias_variable([out_seq])
    h_conv = tf.nn.relu(conv2d(X, W) + b)
    return max_pool_2x2(h_conv)

def lstm(X):
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    outputs,states = tf.nn.dynamic_rnn(lstm_cell, X, initial_state=_init_state, time_major=False)
    W = weight_variable([n_hidden_units, n_classes])
    b = bias_variable([n_classes])
    outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
    results = tf.matmul(outputs[-1], W) + b
    return results

x = tf.placeholder(tf.float32, [None,2025])
y = tf.placeholder(tf.float32, [None,10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(x, [-1,45,45,1])

h_pool1 = conv_pool_layer(x_image, 5, 1, 32)

h_pool2 = conv_pool_layer(h_pool1, 5, 32, 64)

# reshape data
X_in = tf.reshape(h_pool2, [-1,12*12,64])
X_in = tf.transpose(X_in, [0,2,1])

#put into a lstm layer
prediction = lstm(X_in)

# calculate the loss
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
# use Gradientdescentoptimizer
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

correct_pred = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# init session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

epoch=15
batch = 0
display_batch = 600

batch_l = []
train_loss_l = []
train_acc_l = []
test_acc_l = []

for ep in range(epoch):
    print("_"*25)
    print("epoch:{}".format(ep))
    batch_train = next_batch_train_cut(batch_size)
    for batch_x, batch_y in batch_train:
        if batch % display_batch == 0:
            loss, acc = sess.run([cross_entropy, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
            tacc = compute_accuracy()
            batch_l.append(batch)
            train_loss_l.append(loss)
            train_acc_l.append(acc)
            test_acc_l.append(tacc)
            print("batch:{}, train loss:{}, train accuracy:{}, test accuracy:{}".format(batch, loss, acc, tacc))
        sess.run(train_step, feed_dict={x: batch_x, y: batch_y, keep_prob:0.5})
        batch = batch + 1

bunch = Bunch(batch_l=batch_l,train_loss_l=train_loss_l,train_acc_l=train_acc_l,test_acc_l=test_acc_l)
writebunchobj("result/cnn_rnn.dat",bunch)

print("finished")
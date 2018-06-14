import tensorflow as tf
from tools import *

def compute_accuracy():
    global accuracy
    total_acc = 0
    data_len = 0
    batch_test = next_batch_test(100)
    for v_x, v_y in batch_test:
        batch_len = len(v_x)
        result = sess.run(accuracy, feed_dict={x: v_x, y: v_y, keep_prob: 1})
        total_acc += result * batch_len
        data_len += batch_len

    return total_acc/data_len

def calNewFeature():
    batch_train = next_batch_train_cut(100)
    rst = []
    for batch_x, batch_y in batch_train:
        tpa = sess.run(h_fc1,feed_dict={x:batch_x})
        rst.append(tpa)
    conrst =  np.concatenate(tuple(rst))
    print(conrst.shape)
    return conrst


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


x = tf.placeholder(tf.float32, [None,2025])
y = tf.placeholder(tf.float32, [None,10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(x, [-1,45,45,1])

# ********************** conv1 *********************************
# transfer a 5*5*1 imagine into 32 sequence
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
# input a imagine and make a 5*5*1 to 32 with stride=1*1, and activate with relu
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 28*28*32
h_pool1 = max_pool_2x2(h_conv1) # output size 23*23*32

# ********************** conv2 *********************************
# transfer a 5*5*32 imagine into 64 sequence
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
# input a imagine and make a 5*5*32 to 64 with stride=1*1, and activate with relu
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 23*23*64
h_pool2 = max_pool_2x2(h_conv2) # output size 12*12*64

# ********************* func1 layer *********************************
W_fc1 = weight_variable([12*12*64, 1024])
b_fc1 = bias_variable([1024])
# reshape the image from 12,12,64 into a flat (12*12*64)
h_pool2_flat = tf.reshape(h_pool2, [-1,12*12*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

# ********************* func2 layer *********************************
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction,1), tf.argmax(y,1)), tf.float32))

# calculate the loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y*tf.log(prediction), reduction_indices=[1]))
# use Gradientdescentoptimizer
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
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
    batch_train = next_batch_train(100)
    for batch_x, batch_y in batch_train:
        if batch % display_batch == 0:
            loss, acc = sess.run([cross_entropy, accuracy],feed_dict={x: batch_x, y: batch_y,keep_prob: 1.0})
            tacc = compute_accuracy()
            batch_l.append(batch)
            train_loss_l.append(loss)
            train_acc_l.append(acc)
            test_acc_l.append(tacc)
            print("batch:{}, train loss:{}, train accuracy:{}, test accuracy:{}".format(batch,loss,acc,tacc))
        sess.run(train_step, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
        batch = batch + 1

bunch = Bunch(batch_l=batch_l,train_loss_l=train_loss_l,train_acc_l=train_acc_l,test_acc_l=test_acc_l)
writebunchobj("result/cnn.dat",bunch)

print("finished")

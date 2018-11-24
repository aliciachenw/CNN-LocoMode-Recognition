import scipy.io as scio
import tensorflow as tf
import numpy as np
import copy
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
import time

# Parameters for data
n_channels = 1
n_classes = 3

# Load Data
dataFile = 'D://python_program/ANN_terrain/PAPER/DATA/New/data0125_3cla.mat'
modelFile = "Liu0125/3class/model6/fix-pace.ckpt"
resultFile = "D:/python_program/ANN_terrain/PAPER/Liu0125/3class/model6/predict_time.txt"
data = scio.loadmat(dataFile)
X_tr_temp = data['trainData']
lab_tr = data['trainLabel']
X_vld_temp = data['vldData']
lab_vld = data['vldLabel']
X_test_temp = data['testData']
labels_test = data['testLabel']
enc = OneHotEncoder()
enc.fit(lab_tr)
y_tr = enc.transform(lab_tr).toarray()
enc.fit(lab_vld)
y_vld = enc.transform(lab_vld).toarray()
enc.fit(labels_test)
y_test = enc.transform(labels_test).toarray()
num_tr = y_tr.shape[0]
num_vld = y_vld.shape[0]
num_test = y_test.shape[0]
X_tr = np.zeros([num_tr, 1000, n_channels], dtype=float)
for i in range(num_tr):
    X_tr[i, :, 0] = copy.deepcopy(X_tr_temp[i, :])
X_vld = np.zeros([num_vld, 1000, n_channels], dtype=float)
for i in range(num_vld):
    X_vld[i, :, 0] = copy.deepcopy(X_vld_temp[i, :])
X_test = np.zeros([num_test, 1000, n_channels], dtype=float)
for i in range(num_test):
     X_test[i, :, 0] = copy.deepcopy(X_test_temp[i, :])

# Construct CNN #
# hyperparameters
learning_rate = 0.01
epochs = 500
seq_len = 1000
dropout_keep_prob = 0.8

graph = tf.Graph()
with graph.as_default():
    inputs_ = tf.placeholder(tf.float32, [None, seq_len, n_channels], name='inputs')
    labels_ = tf.placeholder(tf.float32, [None, n_classes], name='labels')
    keep_prob_ = tf.placeholder(tf.float32, name='keep')
    learning_rate_ = tf.placeholder(tf.float32, name='learning_rate')
with graph.as_default():
    # (batch, 1000, 1000， 1)->(batch, 200, 200， 4)
    conv1 = tf.layers.conv1d(inputs=inputs_, filters=4, kernel_size=5, strides=1, padding='same',
                                 activation=tf.nn.relu, name='conv1')
    max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=5, strides=5, padding='same', name='pool1')
    # print(max_pool_1.shape)
    # (batch, 500, 16)->(batch, 250, 32)
    conv2 = tf.layers.conv1d(inputs=max_pool_1, filters=8, kernel_size=5, strides=1, padding='same',
                             activation=tf.nn.relu, name='conv2')
    max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=5, strides=5, padding='same', name='pool2')
    # print(max_pool_2.shape)
    # (batch, 250, 32)->(batch, 125, 64)
    conv3 = tf.layers.conv1d(inputs=max_pool_2, filters=16, kernel_size=3, strides=1, padding='same',
                              activation=tf.nn.relu, name='conv3')
    max_pool_3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2, padding='same', name='pool3')
    # print(max_pool_3.shape)
    # (batch, 125, 64)->(batch, 25, 256)
    conv4 = tf.layers.conv1d(inputs=max_pool_3, filters=32, kernel_size=3, strides=1, padding='same',
                              activation=tf.nn.relu, name='conv4')
    max_pool_4 = tf.layers.max_pooling1d(inputs=conv4, pool_size=2, strides=2, padding='same', name='pool4')
    conv5 = tf.layers.conv1d(inputs=max_pool_4, filters=64, kernel_size=3, strides=1, padding='same',
                              activation=tf.nn.relu, name='conv5')
    max_pool_5 = tf.layers.max_pooling1d(inputs=conv5, pool_size=2, strides=2, padding='same', name='pool5')
    # conv6 = tf.layers.conv1d(inputs=max_pool_5, filters=512, kernel_size=3, strides=1, padding='same',
    #                           activation=tf.nn.relu, name='conv6')
    # max_pool_6 = tf.layers.max_pooling1d(inputs=conv6, pool_size=2, strides=2, padding='same', name='pool6')

with graph.as_default():
    flat1 = tf.reshape(max_pool_5, (-1, 64*5))
    flat1 = tf.nn.dropout(flat1, keep_prob=keep_prob_)
    flat2 = tf.layers.dense(flat1, 100, activation=tf.nn.relu)
    logits = tf.layers.dense(flat2, n_classes)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_))
    optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost)
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
    pred = tf.argmax(logits, 1)

# Train the network
validation_acc = []
validation_loss = []

train_acc = []
train_loss = []

test_acc = []
test_loss = []
best_vld = 0
train_epochs = 0
p = 0
m = 20

with graph.as_default():
    saver = tf.train.Saver(max_to_keep=500)

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, modelFile)
    document = open(resultFile, "w+")
    starttime = time.clock()
    # Feed
    feed = {inputs_: X_tr, labels_: y_tr, keep_prob_: 1.0}
    pred_lab = sess.run([pred], feed_dict=feed)
    feed = {inputs_: X_vld, labels_: y_vld, keep_prob_: 1.0}
    pred_lab = sess.run([pred], feed_dict=feed)
    feed = {inputs_: X_test, labels_: y_test, keep_prob_: 1.0}
    pred_lab = sess.run([pred], feed_dict=feed)
    # Loss
    # loss_test, acc_test = sess.run([cost, accuracy], feed_dict=feed)
    # Print info
    # print("Test loss: {:6f}".format(loss_test), "Test acc: {:.6f}".format(acc_test))

    endtime = time.clock()
    t = (endtime - starttime)
    t2 = t / (X_tr.shape[0] + X_vld.shape[0] + X_test.shape[0])
    # pred_lab = np.transpose(np.asarray(pred_lab, int))
    # con_table = confusion_matrix(labels_test, pred_lab)
    # print(con_table)
    document.write("%lf %lf"%(t,t2))
    document.close()

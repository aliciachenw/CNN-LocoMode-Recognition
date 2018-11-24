import scipy.io as scio
import tensorflow as tf
import numpy as np
import copy
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

# Parameters for data
n_channels = 1
n_classes = 5

# Load Data
dataFile = 'D://python_program/ANN_terrain/PAPER/DATA/New/leave-one/data0614_5cla.mat'
modelFile = "Wang0614/5class/kfold.ckpt"
resultFile = "D:/python_program/ANN_terrain/PAPER/Wang0614/5class/kfold.txt"
data = scio.loadmat(dataFile)
X_all = data['rawData']
lab_all = data['rawLabel']

enc = OneHotEncoder()
enc.fit(lab_all)
y_all = enc.transform(lab_all).toarray()
num_all = y_all.shape[0]

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

m = 20
kf = StratifiedKFold(n_splits=5)
kf.get_n_splits(X_all, lab_all)
con_table = np.zeros([n_classes, n_classes], int)
acc_loo = 0
loss_loo = 0
document = open(resultFile, "w+")

with graph.as_default():
    saver = tf.train.Saver(max_to_keep=500)

for train_index, test_index in kf.split(X_all, lab_all):
    X_train, X_test_temp = X_all[train_index], X_all[test_index]
    labels_train, labels_test = lab_all[train_index], lab_all[test_index]
    y_test = y_all[test_index]
    X_tr_temp, X_vld_temp, lab_tr, lab_vld = train_test_split(X_train, labels_train,
                                                              stratify=labels_train, test_size=0.2)
    enc.fit(lab_tr)
    y_tr = enc.transform(lab_tr).toarray()
    enc.fit(lab_vld)
    y_vld = enc.transform(lab_vld).toarray()
    enc.fit(labels_test)
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

    best_vld = 0
    train_epochs = 0
    p = 0
    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        # Loop over epochs
        for e in range(epochs):
            # Feed dictionary
            feed = {inputs_: X_tr, labels_: y_tr, keep_prob_: dropout_keep_prob, learning_rate_: learning_rate}
            # Loss
            loss, _, acc = sess.run([cost, optimizer, accuracy], feed_dict=feed)
            # Feed
            feed = {inputs_: X_vld, labels_: y_vld, keep_prob_: 1.0}
            # Loss
            loss_v, acc_v = sess.run([cost, accuracy], feed_dict=feed)
            if(e==0):
                saver.save(sess, modelFile)
                best_vld = loss_v
                print(best_vld, e+1)
                train_epochs = e+1
            elif(loss_v < best_vld):
                saver.save(sess, modelFile)
                best_vld = loss_v
                train_epochs = e + 1
                print(best_vld, e + 1)
                p = e
            elif(e-p > m):
                break
        # Feed
        feed = {inputs_: X_test, labels_: y_test, keep_prob_: 1.0}
        # Loss
        loss_test, acc_test = sess.run([cost, accuracy], feed_dict=feed)
        pred_lab = sess.run([pred], feed_dict=feed)
        pred_lab = np.transpose(np.asarray(pred_lab, int))
        con_table_temp = confusion_matrix(labels_test, pred_lab)
        con_table += con_table_temp
        document.write("%f %f\n" % (loss_test, acc_test))
        acc_loo = acc_loo + acc_test
        loss_loo = loss_loo + loss_test


acc_loo = acc_loo/5
loss_loo = loss_loo/5

document.write("%f\n" % (loss_loo))
document.write("%f\n" % (acc_loo))
document.write("%d %d %d %d %d\n %d %d %d %d %d\n %d %d %d %d %d\n %d %d %d %d %d\n %d %d %d %d %d\n" % (
    con_table[0][0], con_table[0][1], con_table[0][2], con_table[0][3], con_table[0][4],
    con_table[1][0], con_table[1][1], con_table[1][2], con_table[1][3], con_table[1][4],
    con_table[2][0], con_table[2][1], con_table[2][2], con_table[2][3], con_table[2][4],
    con_table[3][0], con_table[3][1], con_table[3][2], con_table[3][3], con_table[3][4],
    con_table[4][0], con_table[4][1], con_table[4][2], con_table[4][3], con_table[4][4],
))
document.close()

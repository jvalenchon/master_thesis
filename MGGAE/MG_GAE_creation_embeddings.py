# based on the Github code of Van den Berg https://github.com/riannevdberg/gc-mc
#modified by Juliette Valenchon

from __future__ import division
from __future__ import print_function

import argparse
import datetime
import time

import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import sys
import json
from sklearn.metrics import roc_curve, auc
import csv

from preprocessing import sparse_to_tuple, load_data_monti_tadpole
from model import MG_GAE
from utils import construct_feed_dict
import read_tadpole
import preprocessing_dataset

VERBOSE = True

path_dataset_matrix ='bl_data_MCI_v1.csv'
labels_path = 'labels_comma.csv'

#load the .csv files
M_str, nrRows, nrCols = read_tadpole.load_csv_no_header(path_dataset_matrix)
labels, _, _ = read_tadpole.load_csv_no_header(labels_path)
gender_file_path = 'gender.csv'
gender_column, nrRowsg, nrColsg = read_tadpole.load_csv_no_header(gender_file_path)
sex=[]
for i in range(len(gender_column)):
    if gender_column[i] == 'F':
        sex.append(1)
    else:
        sex.append(0)

sex=np.array(sex)
gender_column=sex.reshape(-1)

age_file_path = 'age.csv'
age_column, nrRowsa, nrColsa = preprocessing_dataset.load_csv_no_header_float(age_file_path)
age_column=age_column.reshape(-1)
#parameters/preprocessing step that do not change during the running
labels = preprocessing_dataset.str_to_float(labels)
labels_save=labels.reshape(-1)
M_init= np.concatenate((M_str, labels), axis=1)
M_support = np.concatenate((M_str, labels), axis=1)

#M normalized with -10 for missing values
M = preprocessing_dataset.normalization_gae(M_init)
M_sup=preprocessing_dataset.normalization_for_supportreal(M_support)

seed=0
training_set_mask, testing_set_mask, idx_training, idx_testing=preprocessing_dataset.split_train_test(0.8, M_str, seed, labels)
#create a training and test mask on the data
Otraining = preprocessing_dataset.load_mask(training_set_mask, M_str, nrRows, nrCols)
Otest= preprocessing_dataset.load_mask(testing_set_mask, M_str, nrRows, nrCols)

new_labels_train=np.copy(labels)
new_labels_train[idx_testing]=-1
#split train set into 4 parts to create a validation set
training_set_mask, validation_set_mask, idx_training, idx_validation=preprocessing_dataset.split_train_validation_4(3, M_str, seed, new_labels_train)
Otraining = preprocessing_dataset.load_mask(training_set_mask, M_str, nrRows, nrCols)
Ovalidation= preprocessing_dataset.load_mask(validation_set_mask, M_str, nrRows, nrCols)

Otraining = np.concatenate((Otraining, training_set_mask), axis=1)
Ocol = np.zeros((Otest.shape[0], 1 ))
Otest_support = np.concatenate((Otest, Ocol), axis=1)
Ovalidation_support = np.concatenate((Ovalidation, Ocol), axis=1)
Osupport_t = Otraining + Otest_support + Ovalidation_support
Ovalidation = np.concatenate((Ovalidation, validation_set_mask), axis=1)
Otest = np.concatenate((Otest, testing_set_mask), axis=1)

u_features, v_features, train_labels, train_u_indices, train_v_indices, val_labels, val_u_indices, val_v_indices, test_labels, test_u_indices, test_v_indices = load_data_monti_tadpole(M, Otraining, Otest, Ovalidation)

indexes_train = np.concatenate((idx_testing, idx_validation), axis=0)
indexes_validation = np.concatenate((idx_training, idx_testing), axis=0)
indexes_test = np.concatenate((idx_training, idx_validation), axis=0)


gender_column_train=np.delete(gender_column, indexes_train, 0)
gender_column_test=np.delete(gender_column, indexes_test, 0)
gender_column_val=np.delete(gender_column, indexes_validation, 0)

age_column_train=np.delete(age_column, indexes_train, 0)
age_column_test=np.delete(age_column, indexes_test, 0)
age_column_val=np.delete(age_column, indexes_validation, 0)

labels_train = np.delete(labels_save, indexes_train, 0)
labels_test= np.delete(labels_save, indexes_test, 0)
labels_val = np.delete(labels_save, indexes_validation, 0)
np.savetxt('age_column_train_tadpole.csv', age_column_train, fmt='%s',delimiter=',')
np.savetxt('gender_column_train_tadpole.csv' , gender_column_train, fmt='%s',delimiter=',')
np.savetxt('labels_column_train_tadpole.csv' , labels_train, fmt='%s',delimiter=',')
np.savetxt('age_column_test_tadpole.csv', age_column_test, fmt='%s',delimiter=',')
np.savetxt('gender_column_test_tadpole.csv' , gender_column_test, fmt='%s',delimiter=',')
np.savetxt('labels_column_test_tadpole.csv' , labels_test, fmt='%s',delimiter=',')
np.savetxt('age_column_val_tadpole.csv', age_column_val, fmt='%s',delimiter=',')
np.savetxt('gender_column_val_tadpole.csv' , gender_column_val, fmt='%s',delimiter=',')
np.savetxt('labels_column_val_tadpole.csv' , labels_val, fmt='%s',delimiter=',')

m, n = M.shape

# global normalization
support = []
support_t = []

path_support_women = "sex_women.csv"
women_support, _, _ = read_tadpole.load_csv_no_header(path_support_women)
women_support = preprocessing_dataset.str_to_float(women_support)
women_support=women_support*M_sup
women_support = sp.csr_matrix(women_support, dtype=np.float32)
support.append(women_support)
support_t.append(women_support.T)

path_support_men = "sex_men.csv"
men_support, _, _ = read_tadpole.load_csv_no_header(path_support_men)
men_support = preprocessing_dataset.str_to_float(men_support)
men_support=men_support*M_sup
men_support = sp.csr_matrix(men_support, dtype=np.float32)
support.append(men_support)
support_t.append(men_support.T)

path_support_women_84 = "age_84_92_women.csv"
women_84_support, _, _ = read_tadpole.load_csv_no_header(path_support_women_84)
women_84_support = preprocessing_dataset.str_to_float(women_84_support)
women_84_support=women_84_support*M_sup
women_84_support = sp.csr_matrix(women_84_support, dtype=np.float32)
support.append(women_84_support)
support_t.append(women_84_support.T)

path_support_men_84 = "age_84_92_men.csv"
men_84_support, _, _ = read_tadpole.load_csv_no_header(path_support_men_84)
men_84_support = preprocessing_dataset.str_to_float(men_84_support)
men_84_support=men_84_support*M_sup
men_84_support = sp.csr_matrix(men_84_support, dtype=np.float32)
support.append(men_84_support)
support_t.append(men_84_support.T)

path_support_84 = "age84_92.csv"
age84_support, _, _ = read_tadpole.load_csv_no_header(path_support_84)
age84_support = preprocessing_dataset.str_to_float(age84_support)
age84_support=age84_support*M_sup
age84_support = sp.csr_matrix(age84_support, dtype=np.float32)
support.append(age84_support)
support_t.append(age84_support.T)

path_support_women_79 = "age_79_84_women.csv"
women_79_support, _, _ = read_tadpole.load_csv_no_header(path_support_women_79)
women_79_support = preprocessing_dataset.str_to_float(women_79_support)
women_79_support=women_79_support*M_sup
women_79_support = sp.csr_matrix(women_79_support, dtype=np.float32)
support.append(women_79_support)
support_t.append(women_79_support.T)

path_support_men_79 = "age_79_84_men.csv"
men_79_support, _, _ = read_tadpole.load_csv_no_header(path_support_men_79)
men_79_support = preprocessing_dataset.str_to_float(men_79_support)
men_79_support=men_79_support*M_sup
men_79_support = sp.csr_matrix(men_79_support, dtype=np.float32)
support.append(men_79_support)
support_t.append(men_79_support.T)

path_support_79 = "age79_84.csv"
age79_support, _, _ = read_tadpole.load_csv_no_header(path_support_79)
age79_support = preprocessing_dataset.str_to_float(age79_support)
age79_support=age79_support*M_sup
age79_support = sp.csr_matrix(age79_support, dtype=np.float32)
support.append(age79_support)
support_t.append(age79_support.T)

path_support_women_74 = "age_74_79_women.csv"
women_74_support, _, _ = read_tadpole.load_csv_no_header(path_support_women_74)
women_74_support = preprocessing_dataset.str_to_float(women_74_support)
women_74_support=women_74_support*M_sup
women_74_support = sp.csr_matrix(women_74_support, dtype=np.float32)
support.append(women_74_support)
support_t.append(women_74_support.T)

path_support_men_74 = "age_74_79_men.csv"
men_74_support, _, _ = read_tadpole.load_csv_no_header(path_support_men_74)
men_74_support = preprocessing_dataset.str_to_float(men_74_support)
men_74_support=men_74_support*M_sup
men_74_support = sp.csr_matrix(men_74_support, dtype=np.float32)
support.append(men_74_support)
support_t.append(men_74_support.T)

path_support_74 = "age74_79.csv"
age74_support, _, _ = read_tadpole.load_csv_no_header(path_support_74)
age74_support = preprocessing_dataset.str_to_float(age74_support)
age74_support=age74_support*M_sup
age74_support = sp.csr_matrix(age74_support, dtype=np.float32)
support.append(age74_support)
support_t.append(age74_support.T)

path_support_women_69 = "age_69_74_women.csv"
women_69_support, _, _ = read_tadpole.load_csv_no_header(path_support_women_69)
women_69_support = preprocessing_dataset.str_to_float(women_69_support)
women_69_support=women_69_support*M_sup
women_69_support = sp.csr_matrix(women_69_support, dtype=np.float32)
support.append(women_69_support)
support_t.append(women_69_support.T)

path_support_men_69 = "age_69_74_men.csv"
men_69_support, _, _ = read_tadpole.load_csv_no_header(path_support_men_69)
men_69_support = preprocessing_dataset.str_to_float(men_69_support)
men_69_support=men_69_support*M_sup
men_69_support = sp.csr_matrix(men_69_support, dtype=np.float32)
support.append(men_69_support)
support_t.append(men_69_support.T)

path_support_69 = "age69_74.csv"
age69_support, _, _ = read_tadpole.load_csv_no_header(path_support_69)
age69_support = preprocessing_dataset.str_to_float(age69_support)
age69_support=age69_support*M_sup
age69_support = sp.csr_matrix(age69_support, dtype=np.float32)
support.append(age69_support)
support_t.append(age69_support.T)

path_support_women_64 = "age_64_69_women.csv"
women_64_support, _, _ = read_tadpole.load_csv_no_header(path_support_women_64)
women_64_support = preprocessing_dataset.str_to_float(women_64_support)
women_64_support=women_64_support*M_sup
women_64_support = sp.csr_matrix(women_64_support, dtype=np.float32)
support.append(women_64_support)
support_t.append(women_64_support.T)

path_support_men_64 = "age_64_69_men.csv"
men_64_support, _, _ = read_tadpole.load_csv_no_header(path_support_men_64)
men_64_support = preprocessing_dataset.str_to_float(men_64_support)
men_64_support=men_64_support*M_sup
men_64_support = sp.csr_matrix(men_64_support, dtype=np.float32)
support.append(men_64_support)
support_t.append(men_64_support.T)

path_support_64 = "age64_69.csv"
age64_support, _, _ = read_tadpole.load_csv_no_header(path_support_64)
age64_support = preprocessing_dataset.str_to_float(age64_support)
age64_support=age64_support*M_sup
age64_support = sp.csr_matrix(age64_support, dtype=np.float32)
support.append(age64_support)
support_t.append(age64_support.T)

path_support_women_59 = "age_59_64_women.csv"
women_59_support, _, _ = read_tadpole.load_csv_no_header(path_support_women_59)
women_59_support = preprocessing_dataset.str_to_float(women_59_support)
women_59_support=women_59_support*M_sup
women_59_support = sp.csr_matrix(women_59_support, dtype=np.float32)
support.append(women_59_support)
support_t.append(women_59_support.T)

path_support_men_59 = "age_59_64_men.csv"
men_59_support, _, _ = read_tadpole.load_csv_no_header(path_support_men_59)
men_59_support = preprocessing_dataset.str_to_float(men_59_support)
men_59_support=men_59_support*M_sup
men_59_support = sp.csr_matrix(men_59_support, dtype=np.float32)
support.append(men_59_support)
support_t.append(men_59_support.T)

path_support_59 = "age59_64.csv"
age59_support, _, _ = read_tadpole.load_csv_no_header(path_support_59)
age59_support = preprocessing_dataset.str_to_float(age59_support)
age59_support=age59_support*M_sup
age59_support = sp.csr_matrix(age59_support, dtype=np.float32)
support.append(age59_support)
support_t.append(age59_support.T)

path_support_women_54 = "age_54_59_women.csv"
women_54_support, _, _ = read_tadpole.load_csv_no_header(path_support_women_54)
women_54_support = preprocessing_dataset.str_to_float(women_54_support)
women_54_support=women_54_support*M_sup
women_54_support = sp.csr_matrix(women_54_support, dtype=np.float32)
support.append(women_54_support)
support_t.append(women_54_support.T)

path_support_men_54 = "age_54_59_men.csv"
men_54_support, _, _ = read_tadpole.load_csv_no_header(path_support_men_54)
men_54_support = preprocessing_dataset.str_to_float(men_54_support)
men_54_support=men_54_support*M_sup
men_54_support = sp.csr_matrix(men_54_support, dtype=np.float32)
support.append(men_54_support)
support_t.append(men_54_support.T)

path_support_54 = "age54_59.csv"
age54_support, _, _ = read_tadpole.load_csv_no_header(path_support_54)
age54_support = preprocessing_dataset.str_to_float(age54_support)
age54_support=age54_support*M_sup
age54_support = sp.csr_matrix(age54_support, dtype=np.float32)
support.append(age54_support)
support_t.append(age54_support.T)

num_support = len(support)
mask_support_t=[]
Osupport_t=sp.csr_matrix(Osupport_t, dtype=np.int)
for i in range(num_support):
    mask_support_t.append(Osupport_t.T)

mask_support_t = sp.hstack(mask_support_t, format='csr')

support = sp.hstack(support, format='csr')
support_t = sp.hstack(support_t, format='csr')

# Collect all subject and feature nodes for test set
test_u = list(set(test_u_indices))
test_v = list(set(test_v_indices))
test_u_dict = {n: i for i, n in enumerate(test_u)}
test_v_dict = {n: i for i, n in enumerate(test_v)}

test_u_indices = np.array([test_u_dict[o] for o in test_u_indices])
test_v_indices = np.array([test_v_dict[o] for o in test_v_indices])
test_support = support[np.array(test_u)]
for i in range(test_support.shape[0]):
    for j in range(563, test_support.shape[1], 564):
        test_support[i,j] = 0.0
test_support_t=sp.csr_matrix.multiply(support_t, mask_support_t)

# Collect all subject and feature nodes for validation set
val_u = list(set(val_u_indices))
val_v = list(set(val_v_indices))
val_u_dict = {n: i for i, n in enumerate(val_u)}
val_v_dict = {n: i for i, n in enumerate(val_v)}

val_u_indices = np.array([val_u_dict[o] for o in val_u_indices])
val_v_indices = np.array([val_v_dict[o] for o in val_v_indices])

val_support = support[np.array(val_u)]
for i in range(val_support.shape[0]):
    for j in range(563, val_support.shape[1], 564):
        val_support[i,j] = 0.0
val_support_t=sp.csr_matrix.multiply(support_t, mask_support_t)

# Collect all subject and feature nodes for training set
train_u = list(set(train_u_indices))
train_v = list(set(train_v_indices))
train_u_dict = {n: i for i, n in enumerate(train_u)}
train_v_dict = {n: i for i, n in enumerate(train_v)}

train_u_indices = np.array([train_u_dict[o] for o in train_u_indices])
train_v_indices = np.array([train_v_dict[o] for o in train_v_indices])

train_support = support[np.array(train_u)]
train_support_t = sp.csr_matrix.multiply(support_t, mask_support_t)


placeholders = {
    'u_features': tf.sparse_placeholder(tf.float32, shape=np.array(u_features.shape, dtype=np.int64)),
    'v_features': tf.sparse_placeholder(tf.float32, shape=np.array(v_features.shape, dtype=np.int64)),
    'u_features_nonzero': tf.placeholder(tf.int32, shape=()),
    'v_features_nonzero': tf.placeholder(tf.int32, shape=()),
    'labels': tf.placeholder(tf.float32, shape=(None,)),
    'indices_labels': tf.placeholder(tf.int32, shape=(None,)),

    'user_indices': tf.placeholder(tf.int32, shape=(None,)),
    'item_indices': tf.placeholder(tf.int32, shape=(None,)),

    'dropout': tf.placeholder_with_default(0., shape=()),
    'weight_decay': tf.placeholder_with_default(0., shape=()),

    'support': tf.sparse_placeholder(tf.float32, shape=(None, None)),
    'support_t': tf.sparse_placeholder(tf.float32, shape=(None, None)),
}

gamma, beta, hidden_a, hidden_b, lr =[0.576360939, 0.913519177, 99, 25, 0.00996575458]
hidden_a=hidden_a*23
hidden=[hidden_a, hidden_b]

div = hidden[0] // num_support
if hidden[0] % num_support != 0:
    print("""\nWARNING: HIDDEN[0] (=%d) of stack layer is adjusted to %d such that
                      it can be evenly split in %d splits.\n""" % (hidden[0], num_support * div, num_support))
hidden[0] = num_support * div

# create model
model = MG_GAE(placeholders,
                           input_dim=u_features.shape[1],
                           num_support=num_support,
                           hidden=hidden,
                           num_users=m,
                           num_items=n,
                           learning_rate=lr,
                           gamma= gamma,
                           beta=beta,
                           logging=True)

# Convert sparse placeholders to tuples to construct feed_dict
test_support = sparse_to_tuple(test_support)
test_support_t = sparse_to_tuple(test_support_t)

val_support = sparse_to_tuple(val_support)
val_support_t = sparse_to_tuple(val_support_t)

train_support = sparse_to_tuple(train_support)
train_support_t = sparse_to_tuple(train_support_t)

u_features = sparse_to_tuple(u_features)
v_features = sparse_to_tuple(v_features)

assert u_features[2][1] == v_features[2][1], 'Number of features of users and items must be the same!'

num_features = u_features[2][1]
u_features_nonzero = u_features[1].shape[0]
v_features_nonzero = v_features[1].shape[0]

indices_labels = [563]*train_labels.shape[0]
indices_labels_val = [563]*val_labels.shape[0]
indices_labels_test = [563]*test_labels.shape[0]

# Feed_dicts for validation and test set stay constant over different update steps
train_feed_dict = construct_feed_dict(placeholders, u_features, v_features, u_features_nonzero,
                                      v_features_nonzero, train_support, train_support_t,
                                      train_labels,indices_labels, train_u_indices, train_v_indices, 0.)

val_feed_dict = construct_feed_dict(placeholders, u_features, v_features, u_features_nonzero,
                                    v_features_nonzero, val_support, val_support_t,
                                    val_labels, indices_labels_val, val_u_indices, val_v_indices, 0.)

test_feed_dict = construct_feed_dict(placeholders, u_features, v_features, u_features_nonzero,
                                     v_features_nonzero, test_support, test_support_t,
                                     test_labels, indices_labels_test, test_u_indices, test_v_indices, 0.)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

print('Training...')
auc_val=[]
auc_train=[]
auc_test=[]
gcn_u_train_array=[]
gcn_v_train_array=[]
gcn_u_test_array=[]
gcn_v_test_array=[]
gcn_u_val_array=[]
gcn_v_val_array=[]
NB_EPOCH=100
for epoch in range(NB_EPOCH):

    t = time.time()
    outs = sess.run([model.training_op, model.loss, model.indices, model.labels, model.outputs, model.labels_class, model.classification, model.inputs, model.gcn_u, model.gcn_v, model.loss_frob, model.binary_entropy], feed_dict=train_feed_dict)
    train_avg_loss = outs[1]
    label_train = outs[5]
    output_train = outs[6]

    gcn_u_train_array.append(outs[-4])
    gcn_v_train_array.append(outs[-3])
    fpr_train, tpr_train, thresholds_train=roc_curve(label_train, output_train)
    roc_auc_train= auc(fpr_train, tpr_train)
    auc_train.append(roc_auc_train)

    val_avg_loss, val_classification, val_labels_corres, gcn_u_val, gcn_v_val = sess.run([model.loss, model.classification, model.labels_class, model.gcn_u, model.gcn_v], feed_dict=val_feed_dict)
    fpr_val, tpr_val, thresholds_train=roc_curve(val_labels_corres, val_classification)
    roc_auc_val= auc(fpr_val, tpr_val)
    auc_val.append(roc_auc_val)
    gcn_u_val_array.append(gcn_u_val)
    gcn_v_val_array.append(gcn_v_val)

    test_avg_loss, test_classification, test_labels_corres, gcn_u_test, gcn_v_test = sess.run([model.loss, model.classification, model.labels_class, model.gcn_u, model.gcn_v], feed_dict=test_feed_dict)
    fpr_test, tpr_test, thresholds_test=roc_curve(test_labels_corres, test_classification, pos_label=label_train.max())
    roc_auc_test = auc(fpr_test, tpr_test)
    auc_test.append(roc_auc_test)
    gcn_u_test_array.append(gcn_u_test)
    gcn_v_test_array.append(gcn_v_test)

    if VERBOSE:
        print("[*] Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_avg_loss),
              "train_auc=", "{:.5f}".format(roc_auc_train),
              "val_loss=", "{:.5f}".format(val_avg_loss),
              "val_auc=", "{:.5f}".format(roc_auc_val),
              "\t\ttime=", "{:.5f}".format(time.time() - t))
        print('test auc = ', roc_auc_test)



iteration_max_val = np.argmax(auc_val)
auc_test_final=auc_test[iteration_max_val]
auc_train_final=auc_train[iteration_max_val]
auc_val_final=auc_val[iteration_max_val]
print('At the best validation AUC, we have a train AUC of %0.3f, a validation AUC of %0.3f and a test AUC of %0.3f' %(auc_train_final, auc_val_final, auc_test_final))
gcn_u_train = gcn_u_train_array[iteration_max_val]
gcn_v_train = gcn_v_train_array[iteration_max_val]
gcn_u_test = gcn_u_test_array[iteration_max_val]
gcn_v_test = gcn_v_test_array[iteration_max_val]
gcn_u_val = gcn_u_val_array[iteration_max_val]
gcn_v_val = gcn_v_val_array[iteration_max_val]

np.savetxt('gcn_u_train_emb%i_tadpole.csv' %iteration_max_val, gcn_u_train, fmt='%s',delimiter=',')
np.savetxt('gcn_v_train_emb%i_tadpole.csv' %iteration_max_val, gcn_v_train, fmt='%s',delimiter=',')
np.savetxt('gcn_u_val_emb%i_tadpole.csv' %iteration_max_val, gcn_u_val, fmt='%s',delimiter=',')
np.savetxt('gcn_v_val_emb%i_tadpole.csv' %iteration_max_val, gcn_v_val, fmt='%s',delimiter=',')
np.savetxt('gcn_u_test_emb%i_tadpole.csv' %iteration_max_val, gcn_u_test, fmt='%s',delimiter=',')
np.savetxt('gcn_v_test_emb%i_tadpole.csv' %iteration_max_val, gcn_v_test, fmt='%s',delimiter=',')

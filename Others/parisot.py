#######################################
##
## REQUIRED LIBRARIES + program preprocessing_dataset.py
##
######################################
import os,sys,inspect
import joblib
import numpy as np
import tensorflow as tf
import h5py
import scipy.sparse as sp
import scipy
import time
from scipy.sparse import csgraph
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import Imputer
from scipy.spatial import distance
import rbfopt #FOR OPTIMIZATION

import preprocessing_dataset
import read_tadpole


###############################
##
## csv files required to run the program
##
##############################
path_dataset_matrix = 'bl_data_MCI_v1.csv'
path_dataset_affinity_matrix = 'affinity_mat_parisot_v1.csv'
labels_path = 'labels_comma.csv'
#load the .csv files
M_str, nrRows, nrCols = read_tadpole.load_csv_no_header(path_dataset_matrix)
Wrow, _, _ = read_tadpole.load_csv_no_header(path_dataset_affinity_matrix)
labels, _, _ = read_tadpole.load_csv_no_header(labels_path)

#parameters/preprocessing step that do not change during the running
Wrow=preprocessing_dataset.str_to_float(Wrow)
labels = preprocessing_dataset.str_to_float(labels)
M_float = preprocessing_dataset.preprocessing_nan_normalization(M_str)

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp = imp.fit(M_float)
# Impute our data
M_float_imp = imp.transform(M_float)

M = M_float_imp

Wrow= preprocessing_dataset.normalize_adj(Wrow)
#computation of the normalized laplacians
Lrow = csgraph.laplacian(Wrow, normed=True)

ord_row = 3 # row for the Chebyshev polynomials


class Train_test_matrix_completion:

    """
    The neural network model.
    """

    def frobenius_norm_square(self, tensor):
        """
        Function that returns the squared Frobenius norm of tensor
        Input: tensor: the tensor that we would like to know the norm of
        Output: Frobenius norm of the tensor
        """
        square_tensor = tf.square(tensor)
        tensor_sum = tf.reduce_sum(square_tensor)
        return tensor_sum

    def mono_conv_cheby(self, list_lap, ord_conv, A, W, b):
        """
        Function that returns the output of a GCNN layer
        Input: list_lap: list of the Chebyshev polynomials
               ord_conv: number of Chebyshev polynomial used
               A: input matrix
               W: weight matrix
               b: bias vector
        Output: conv_feat : output of GCNN layer
        """
        feat = []
        #collect features
        for k in range(ord_conv):
            c_lap = list_lap[k]
            c_feat = tf.matmul(c_lap, A, a_is_sparse=False)
            feat.append(c_feat)
        all_feat = tf.concat(feat, 1)
        conv_feat = tf.matmul(all_feat, W) + b
        #conv_feat = tf.nn.relu(conv_feat)
        return conv_feat

    def mono_conv(self, adjacency, X, W, b):
        """
        Function that returns the output of a GCN layer
        Input: adjacency: adjacency matrix of the graph
               X: input matrix
               W: weight matrix
               b: bias vector
        Output: conv_feat : output of GCN layer
        """
        conv_feat_1 = tf.matmul(adjacency, X)
        conv_feat = tf.matmul(conv_feat_1, W) + b
        #conv_feat = tf.nn.relu(conv_feat)
        return conv_feat

    def compute_cheb_polynomials(self, L, ord_cheb, list_cheb):
        """
        Function that computes the list of the Chebyshev polynomials list_cheb
        Input: L: Laplacian of the graph
               ord_conv: number of Chebyshev polynomial used
               list_cheb: list of the Chebyshev polynomials
        """
        for k in range(ord_cheb):
            if (k==0):
                list_cheb.append(tf.cast(tf.diag(tf.ones([L.get_shape().as_list()[0],])), 'float32'))
            elif (k==1):
                list_cheb.append(tf.cast(L, 'float32'))
            else:
                list_cheb.append(2*tf.matmul(L, list_cheb[k-1])  - list_cheb[k-2])

    def __init__(self, M, L, A, labels_train, labels_test, labels_val, testing_set_mask, training_set_mask,validation_set_mask,
                 order_chebyshev_row = 3, n_conv_feat=32, dropout=0.98,
                 learning_rate=0.01, l2_regu=0.00001, nb_layers=5, idx_gpu = '/gpu:1'):
        """
                 Neural network architecture. Output classification result for each row of M.
                 Inputs:
                    M: initial matrix with all the known values,
                    A : adjacency matrices, respectively for the age, sex and age and sex graphs,
                    L: laplacian matrices, respectively for the age, sex and age and sex graphs,
                    labels_train, labels_test, labels_val: labels for each set for every subject,
                    training_set_mask, testing_set_mask, validation_set_mask: indexes of subjects that respectively belong to the training, testing and validation sets,
                    order_chebyshev_row: order to use for the Chebyshev polynomials. Default value = 18,
                    n_conv_feat: number of weights to use for the GCNN layer. Default value = 36,
                    l2_regu: coefficient to use in front of the l2 regularization term. Default value = 1,
                    dropout: dropout rate on the GCN output. Default = 0.5,
                    learning_rate: learning rate. Default value = 0.001,
                    nb_layers: number of GCNN layers. Default value: 5
        """
        self.ord_row = order_chebyshev_row
        self.n_conv_feat = n_conv_feat

        with tf.Graph().as_default() as g:
                tf.logging.set_verbosity(tf.logging.ERROR)
                self.graph = g
                tf.set_random_seed(0)
                with tf.device(idx_gpu):

                        #loading of the laplacians
                        self.L = tf.cast(L, 'float32')
                        self.norm_L = self.L - tf.diag(tf.ones([L.shape[0], ]))

                        #compute all chebyshev polynomials a priori
                        self.list_row_cheb_pol = list()
                        self.compute_cheb_polynomials(self.norm_L, self.ord_row, self.list_row_cheb_pol)

                        #definition of constant matrices
                        self.A=tf.constant(A, dtype=tf.float32)

                        self.M = tf.constant(M, dtype=tf.float32)
                        self.labels_train=tf.constant(labels_train, dtype=tf.float32)
                        self.labels_test=tf.constant(labels_test, dtype=tf.float32)
                        self.labels_val=tf.constant(labels_val, dtype=tf.float32)
                        self.testing_mask=tf.constant(testing_set_mask, dtype=tf.float32)
                        self.training_mask=tf.constant(training_set_mask, dtype=tf.float32)
                        self.validation_mask=tf.constant(validation_set_mask, dtype=tf.float32)

                        ##################################definition of the NN variables#####################################
                        #definition of the weights for extracting the global features
                        weights=[]
                        bias=[]
                        if nb_layers==1:
                            weights.append(tf.get_variable('weights_W', shape=[self.ord_row*M.shape[1], 1], initializer=tf.contrib.layers.xavier_initializer()))
                            bias.append( tf.Variable(tf.zeros([1,])))
                        else:
                            weights.append(tf.get_variable('weights_W', shape=[self.ord_row*M.shape[1], self.n_conv_feat], initializer=tf.contrib.layers.xavier_initializer()))
                            bias.append(tf.Variable(tf.zeros([self.n_conv_feat,])))
                            for i in range(nb_layers-2):
                                weights.append(tf.get_variable('weights_%d' %i, shape=[self.ord_row*self.n_conv_feat, self.n_conv_feat], initializer=tf.contrib.layers.xavier_initializer()))
                                bias.append(tf.Variable(tf.zeros([self.n_conv_feat,])))
                            weights.append(tf.get_variable("weight_final", shape=[self.ord_row*self.n_conv_feat, 1], initializer=tf.contrib.layers.xavier_initializer()))
                            bias.append(tf.Variable(tf.zeros([1,])))

                        # GCNN architecture TRAINING
                        if nb_layers==1:
                            self.final_feat_users = self.mono_conv_cheby(self.list_row_cheb_pol, self.ord_row, self.M, weights[0], bias[0])
                            self.final_feat_users = tf.nn.dropout(self.final_feat_users, dropout)

                        else:
                            self.final_feat_users = self.mono_conv_cheby(self.list_row_cheb_pol, self.ord_row, self.M, weights[0], bias[0])
                            self.final_feat_users = tf.nn.dropout(self.final_feat_users, dropout)
                            self.final_feat_users = tf.nn.relu(self.final_feat_users)

                            for i in range(nb_layers-2):
                                self.final_feat_users = self.mono_conv_cheby(self.list_row_cheb_pol, self.ord_row, self.final_feat_users, weights[i+1], bias[i+1])
                                self.final_feat_users = tf.nn.dropout(self.final_feat_users, dropout)
                                self.final_feat_users = tf.nn.relu(self.final_feat_users)
                            self.final_feat_users = self.mono_conv_cheby(self.list_row_cheb_pol, self.ord_row, self.final_feat_users, weights[-1], bias[-1])
                            self.final_feat_users = tf.nn.dropout(self.final_feat_users, dropout)

                        self.classification=tf.sigmoid(self.final_feat_users)

                        #########loss definition

                        #computation of the accuracy term
                        self.classification_train = tf.multiply(self.training_mask, self.classification)
                        self.classification_test= tf.multiply(self.testing_mask, self.classification)
                        self.classification_val= tf.multiply(self.validation_mask, self.classification)


                        self.binary_entropy = tf.losses.sigmoid_cross_entropy(multi_class_labels = self.labels_train, logits = self.classification_train)
                        self.l2=0
                        for i in range(len(weights)):
                            self.l2+= tf.nn.l2_loss(weights[i])
                        #training loss definition
                        self.loss = self.binary_entropy+ l2_regu* self.l2

                        # GCNN architecture TESTING

                        self.binary_entropy_test = tf.losses.sigmoid_cross_entropy(multi_class_labels = self.labels_test, logits = self.classification_test)
                        self.predictions_error = self.binary_entropy_test

                        #definition of the solver
                        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

                        self.var_grad = tf.gradients(self.loss, tf.trainable_variables())
                        self.norm_grad = self.frobenius_norm_square(tf.concat([tf.reshape(g, [-1]) for g in self.var_grad], 0))

                        # Create a session for running Ops on the Graph.
                        config = tf.ConfigProto(allow_soft_placement = True)
                        config.gpu_options.allow_growth = True
                        self.session = tf.Session(config=config)

                        # Run the Op to initialize the variables.
                        init = tf.initialize_all_variables()
                        #print(init)
                        self.session.run(init)


def running_one_time(n, dropout, seed, n_conv_feat,lr, l2_regu, nb_layers, ord_row):
    """
    function to run the architecture one time
    Inputs:
       n: number of runs of the architecture for the same initialization with the weights and biases of the last run,
       seed: seed to use for the random sampling between training, testing and validation sets,
       n_conv_feat: number of weights to use for the GCNN layer. Default value = 36,
       l2_regu: coefficient to use in front of the l2 regularization term. Default value = 1,
       dropout: dropout rate on the GCNN output. Default = 0.5,
       lr: learning rate. Default value = 0.001
       nb_layers: number of GCNN layers
       ord_row: number of Chebyshev polynomials
    Outputs:
       auc_test_list, auc_train_list, auc_val_list: lists of the AUC values on the test, train and validation sets for the n runs,
       pred_train_list, pred_test_list, pred_val_list: lists of the prediction values on the test, train and validation sets for the n runs,
       labels_test, labels_train, labels_val: labels of the test, train and validation sets
    """
    #initialization of the matrix, of the training, testing and validation sets
    training_set_mask, testing_set_mask, idx_training, idx_testing=preprocessing_dataset.split_train_test(0.8, M_str, seed, labels)
    new_labels_train=np.copy(labels)
    new_labels_train[idx_testing]=-1
    #creation of a validation set from the training set. Split the training set into 4 parts, one for validation
    training_set_mask, validation_set_mask, idx_training, idx_validation=preprocessing_dataset.split_train_validation_4(3, M_str, seed, new_labels_train)

    labels_train = labels * training_set_mask
    labels_test = labels * testing_set_mask
    labels_val = labels * validation_set_mask

    indexes_train = np.concatenate((idx_testing, idx_validation), axis=0)
    indexes_validation = np.concatenate((idx_training, idx_testing), axis=0)
    indexes_test = np.concatenate((idx_training, idx_validation), axis=0)

    labels_test_reduce = np.delete(labels_test, indexes_test,0)
    labels_train_reduce = np.delete(labels_train, indexes_train,0)
    labels_val_reduce=np.delete(labels_val, indexes_validation, 0)

    learning_obj = Train_test_matrix_completion(M, Lrow, Wrow, labels_train, labels_test, labels_val, testing_set_mask, training_set_mask, validation_set_mask,
                                                        order_chebyshev_row = ord_row, n_conv_feat=n_conv_feat, dropout=dropout,learning_rate=lr, l2_regu=l2_regu, nb_layers=nb_layers)

    num_iter_test = 10
    num_total_iter_training = n
    num_iter = 0

    list_training_loss = list()
    list_training_norm_grad = list()
    list_test_pred_error = list()

    auc_train_list=[]
    pred_train_list=[]
    auc_test_list=[]
    pred_test_list=[]
    auc_val_list=[]
    pred_val_list=[]

    for k in range(num_iter, num_total_iter_training):

        tic = time.time()
        list_of_outputs = learning_obj.session.run([learning_obj.optimizer, learning_obj.loss, learning_obj.norm_grad, learning_obj.classification_train, learning_obj.binary_entropy] + learning_obj.var_grad)
        current_training_loss = list_of_outputs[1]
        norm_grad = list_of_outputs[2]
        pred_train = list_of_outputs[3]
        pred_train = np.delete(pred_train, indexes_train,0)

        fpr_train, tpr_train, thresholds_train=roc_curve(labels_train_reduce, pred_train)
        roc_auc_train = auc(fpr_train, tpr_train)

        X_grad = list_of_outputs[3:]
        training_time = time.time() - tic

        list_training_loss.append(current_training_loss)
        list_training_norm_grad.append(norm_grad)

        if (np.mod(num_iter, num_iter_test)==0):
            msg = "[TRN] iter = %03i, cost = %3.2e, |grad| = %.2e (%3.2es),auc = %3.2e" \
                                        % (num_iter, list_training_loss[-1], list_training_norm_grad[-1], training_time, roc_auc_train)
            print msg

            auc_train_list.append(roc_auc_train)
            pred_train_list.append(pred_train)

            tic = time.time()
            pred_val = learning_obj.session.run([learning_obj.classification_val])#
            test_time = time.time() - tic
            pred_val = np.delete(pred_val[0], indexes_validation,0)
            fpr, tpr, thresholds=roc_curve(labels_val_reduce, pred_val)
            roc_auc_val = auc(fpr, tpr)
            auc_val_list.append(roc_auc_val)
            pred_val_list.append(pred_val)

            msg =  "[VAL] iter = %03i, AUC = %3.2e" % (num_iter, roc_auc_val)
            print msg
            #Test Code
            tic = time.time()
            pred_error, pred = learning_obj.session.run([learning_obj.predictions_error, learning_obj.classification_test])
            test_time = time.time() - tic

            pred = np.delete(pred, indexes_test,0)

            list_test_pred_error.append(pred_error)
            fpr, tpr, thresholds=roc_curve(labels_test_reduce, pred)
            roc_auc = auc(fpr, tpr)

            auc_test_list.append(roc_auc)
            pred_test_list.append(pred)

            msg =  "[TST] iter = %03i, cost = %3.2e, AUC = %3.2e" % (num_iter, list_test_pred_error[-1], roc_auc)
            print msg

        num_iter += 1
    return (auc_test_list, auc_train_list, auc_val_list, pred_train_list, pred_test_list, pred_val_list, labels_test_reduce, labels_train_reduce, labels_val_reduce)


def optimize(x):
    """
    Function to optimize dropout, n_conv_feat (number of outputs of the GCNN layers), lr (learning rate), l2_regu (term in loss function to do the trade-off between l2-regularization and binary cross-entropy), nb_layers (number of GCNN layers), ord_row (number of Chebyshev polynomials fo the GCNN layer) with RBFOpt.
    Input:
        x: value given by RBFOpt for the hyperparameters  by running alg.optimize()
    Output: opposite of the max validation AUC (function to minimize)
    """
    seed=0
    nb_itera =200
    print(x)
    dropout, n_conv_feat, lr, l2_regu,nb_layers, ord_row=x
    n_conv_feat=int(n_conv_feat)
    nb_layers=int(nb_layers)
    ord_row=int(ord_row)

    auc_test_list, auc_train_list, auc_val_list, pred_train_list, pred_test_list, pred_val_list, labels_test_reduce, labels_train_reduce, labels_val_reduce= running_one_time(nb_itera, dropout, seed, n_conv_feat, lr, l2_regu, nb_layers, ord_row)

    iteration_max_val = np.argmax(auc_val_list)
    pred_test_list_final = pred_test_list[iteration_max_val]
    print(auc_train_list)
    print(auc_val_list)
    print(np.max(auc_val_list))
    return -np.max(auc_val_list)

"""bb=rbfopt.RbfoptUserBlackBox(6, np.array([0.5, 1, 0.0005, 0.000001, 1, 1]), np.array([1.0, 100, 0.05, 0.0001, 10, 5]), np.array(['R', 'I', 'R', 'R', 'I', 'I']), optimize)
settings=rbfopt.RbfoptSettings(max_iterations=2000,max_noisy_evaluations=2000,max_evaluations=2000, minlp_solver_path='~/TADPOLE/bonmin', nlp_solver_path='~/TADPOLE/ipopt', target_objval=-1)
alg=rbfopt.RbfoptAlgorithm(settings, bb)
objval, x, itercount, evalcount, fast_evalcount=alg.optimize()


"""
nb_seeds=3
save=False

auc_test=[]
auc_train=[]
auc_val=[]
pred_train=[]
pred_test=[]
pred_val=[]
labels_train=[]
labels_test=[]
labels_val=[]
for seed in range(nb_seeds):
    nb_itera =200
    dropout, n_conv_feat, lr, l2_regu,nb_layers=0.832077907, 76, 0.0005, 0.000001, 3
    n_conv_feat=int(n_conv_feat)
    nb_layers=int(nb_layers)

    auc_test_list, auc_train_list, auc_val_list, pred_train_list, pred_test_list, pred_val_list, labels_test_reduce, labels_train_reduce, labels_val_reduce= running_one_time(nb_itera, dropout, seed, n_conv_feat, lr, l2_regu, nb_layers, 3)

    iteration_max_val = np.argmax(auc_val_list)
    auc_test_value = auc_test_list[iteration_max_val]
    print(auc_train_list)
    print(auc_val_list)
    print('auc test', auc_test_value)
    pred_test_list_final = pred_test_list[iteration_max_val]
    auc_test.append(auc_test_value)
    auc_train.append(auc_train_list)
    auc_val.append(auc_val_list)
    pred_test.append(pred_test_list_final)
    pred_train.append(pred_train_list[iteration_max_val])
    pred_val.append(pred_val_list[iteration_max_val])
    labels_train.append(labels_train_reduce)
    labels_test.append(labels_test_reduce)
    labels_val.append(labels_val_reduce)


if save:
    np.savetxt('parisot_auc_values_test_hidden5_32.csv', auc_test, fmt='%f', delimiter=',')
    np.savetxt('parisot_auc_values_train_hidden5_32.csv', auc_train, fmt='%f', delimiter=',')
    np.savetxt('parisot_auc_values_val_hidden5_32.csv', auc_val, fmt='%f', delimiter=',')
    np.savetxt('parisot_pred_test.csv', pred_test, fmt='%f', delimiter=',')
    np.savetxt('parisot_pred_train.csv', pred_train, fmt='%f', delimiter=',')
    np.savetxt('parisot_pred_val.csv', pred_val, fmt='%f', delimiter=',')
    np.savetxt('parisot_labels_train.csv', labels_train, fmt='%f', delimiter=',')
    np.savetxt('parisot_labels_test.csv', labels_test,fmt='%s',delimiter=',')
    np.savetxt('parisot_labels_val.csv', labels_val, fmt='%s',delimiter=',')

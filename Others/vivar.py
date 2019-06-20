#######################################
##
## REQUIRED LIBRARIES + program preprocessing_dataset.py + read_tadpole.py
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

import rbfopt

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
M_init = preprocessing_dataset.normalization(M_str)
labels = preprocessing_dataset.str_to_float(labels)

M = np.concatenate((M_init, labels), axis=1)

#ADD A SIMILARITY MEASURE TO THE GRAPH
# Calculate all pairwise distances
distv = distance.pdist(M, metric='correlation')
    # Convert to a square symmetric distance matrix
dist = distance.squareform(distv)
sigma = np.mean(dist)
    # Get affinity from similarity matrix
sparse_graph = np.exp(- dist ** 2 / (2 * sigma ** 2))
#Wrow=Wrow*sparse_graph
Wrow= preprocessing_dataset.normalize_adj(Wrow)

#creation of a mask for the features: 1 for features and 0 for labels
M_features_ones = np.ones(M_init.shape)
M_labels_zeros=np.zeros(labels.shape)
mask_features = np.concatenate((M_features_ones, M_labels_zeros), axis=1)

#computation of the normalized laplacians
Lrow = csgraph.laplacian(Wrow, normed=True)


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
        conv_feat = tf.nn.relu(conv_feat)
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
        conv_feat = tf.nn.relu(conv_feat)
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

    def norm_tensor(self, X):
        """
        Compute the norm of tensor X per column. Put the values of X between -1 and 1. Do the same normalization
        as the one done for initialization.
        Input:
            X: tensor to normalize
        Output:
            Xnormed: tensor X normalized
        """
        max=tf.reduce_max(X, axis=0)
        min =tf.reduce_min(X, axis=0)
        temp=tf.slice(X, begin = [0, 0], size = [1, X.get_shape().as_list()[1]])
        Xnormed=(tf.divide(2*(temp-min),(max-min) -1))
        for i in range(1, X.shape[0]):
            temp=tf.slice(X, begin = [i, 0], size = [1, X.get_shape().as_list()[1]])
            Xnormed_col=tf.divide(2*(temp-min),(max-min) -1)
            Xnormed = tf.concat([Xnormed, Xnormed_col], 0)
        return Xnormed

    def __init__(self, M, Lr, Wrow, Otraining, initial_W, initial_H, labels, training_set_mask, testing_set_mask, validation_set_mask, mask_features, validation,
                 order_chebyshev_row = 18, cheby=1, n_conv_feat=36, l2_regu=10,
                 num_iterations = 10, gamma=563.39, gamma_H=688.85, gamma_W=248.91,  gamma_e=890.14, learning_rate=0.00089, idx_gpu = '/gpu:1'):
        """
                     Neural network architecture. Compute an update X of M.
                     Inputs:
                        M: initial matrix with all the known values,
                        initial_W, initial_H: initialization of W and H with the feature values from M and only the labels of the training set,
                        Wrow : adjacency matrix,
                        Lr, Lrow_sex, Lrow_agesex: laplacian matrices, respectively for the age, sex and age and sex graphs,
                        Otraining: mask on the training features to know which ones are on the training set for the loss function,
                        labels: labels for every subject,
                        training_set_mask, testing_set_mask, validation_set_mask: indexes of subjects that respectively belong to the training, testing and validation sets,
                        mask_features: mask composed of 1 values for all the features and 0 for the labels to compute the Frobenius loss term on the features,
                        validation: boolean, to include a validation set or not
                        order_chebyshev_row: order to use for the Chebyshev polynomials. Default value = 18,
                        cheby: boolean, use of a GCNN or a GCN layer. 0: GCN, 1: GCNN. Default value = 1,
                        n_conv_feat: number of weights to use for the GCNN layer. Default value = 36,
                        l2_regu: coefficient to use in front of the l2 regularization term. Default value = 1,
                        dropout: dropout rate on the GCN output. Default = 0.5,
                        num_iterations: number of times that the process GCNN+LSTM is done before updating X and computing the loss function. Default value = 10,
                        gamma, gamma_H, gamma_W, gamma_e: hyperparameters of the loss function in front of all the terms. Default value = 1,
                        learning_rate: learning rate. Default value = 0.001
        """
        self.ord_row = order_chebyshev_row
        self.num_iterations = num_iterations
        self.n_conv_feat = n_conv_feat

        with tf.Graph().as_default() as g:
                tf.logging.set_verbosity(tf.logging.ERROR)
                self.graph = g
                tf.set_random_seed(0)
                with tf.device(idx_gpu):

                        #loading of the laplacians
                        self.Lr = tf.cast(Lr, 'float32')

                        self.norm_Lr = self.Lr - tf.diag(tf.ones([Lr.shape[0], ]))

                        #compute all chebyshev polynomials a priori
                        self.list_row_cheb_pol = list()
                        self.compute_cheb_polynomials(self.norm_Lr, self.ord_row, self.list_row_cheb_pol)

                        #definition of constant matrices
                        self.Wrow=tf.constant(Wrow, dtype=tf.float32)
                        self.M = tf.constant(M, dtype=tf.float32)
                        self.Otraining = tf.constant(Otraining, dtype=tf.float32) #training mask
                        self.training_set_mask = tf.constant(training_set_mask, dtype=tf.float32)
                        self.testing_set_mask =tf.constant(testing_set_mask, dtype=tf.float32)
                        if validation:
                            self.validation_set_mask =tf.constant(validation_set_mask, dtype=tf.float32)

                        self.mask_features=tf.constant(mask_features, dtype=tf.float32)

                        self.output_nn=tf.zeros([training_set_mask.shape[0],])


                        ##################################definition of the NN variables#####################################
                        #cheby=0 #0 or 1. Using the Chebyshev decomposition for the GCNN or not
                        #definition of the weights for extracting the global features
                        if cheby==0:
                            self.W_conv_W = tf.get_variable("W_conv_W", shape=[initial_W.shape[1], self.n_conv_feat], initializer=tf.contrib.layers.xavier_initializer())
                        else:
                            self.W_conv_W = tf.get_variable("W_conv_W", shape=[self.ord_row*initial_W.shape[1], self.n_conv_feat], initializer=tf.contrib.layers.xavier_initializer())
                        self.b_conv_W = tf.Variable(tf.zeros([self.n_conv_feat,]))

                        #recurrent N parameters
                        self.W_f_u = tf.get_variable("W_f_u", shape=[self.n_conv_feat, self.n_conv_feat], initializer=tf.contrib.layers.xavier_initializer())
                        self.W_i_u = tf.get_variable("W_i_u", shape=[self.n_conv_feat, self.n_conv_feat], initializer=tf.contrib.layers.xavier_initializer())
                        self.W_o_u = tf.get_variable("W_o_u", shape=[self.n_conv_feat, self.n_conv_feat], initializer=tf.contrib.layers.xavier_initializer())
                        self.W_c_u = tf.get_variable("W_c_u", shape=[self.n_conv_feat, self.n_conv_feat], initializer=tf.contrib.layers.xavier_initializer())
                        self.U_f_u = tf.get_variable("U_f_u", shape=[self.n_conv_feat, self.n_conv_feat], initializer=tf.contrib.layers.xavier_initializer())
                        self.U_i_u = tf.get_variable("U_i_u", shape=[self.n_conv_feat, self.n_conv_feat], initializer=tf.contrib.layers.xavier_initializer())
                        self.U_o_u = tf.get_variable("U_o_u", shape=[self.n_conv_feat, self.n_conv_feat], initializer=tf.contrib.layers.xavier_initializer())
                        self.U_c_u = tf.get_variable("U_c_u", shape=[self.n_conv_feat, self.n_conv_feat], initializer=tf.contrib.layers.xavier_initializer())
                        self.b_f_u = tf.Variable(tf.zeros([self.n_conv_feat,]))
                        self.b_i_u = tf.Variable(tf.zeros([self.n_conv_feat,]))
                        self.b_o_u = tf.Variable(tf.zeros([self.n_conv_feat,]))
                        self.b_c_u = tf.Variable(tf.zeros([self.n_conv_feat,]))

                        #output parameters
                        self.W_out_W = tf.get_variable("W_out_W", shape=[self.n_conv_feat, initial_W.shape[1]], initializer=tf.contrib.layers.xavier_initializer())
                        self.b_out_W = tf.Variable(tf.zeros([initial_W.shape[1],]))

                        #########definition of the NN
                        #definition of W and H
                        self.W = tf.constant(initial_W.astype('float32'))
                        self.H = tf.Variable(initial_H.astype('float32'))

                        self.X = tf.matmul(self.W, self.H, transpose_b=True) #we may initialize it at random here
                        self.list_X = list()
                        self.list_X.append(tf.identity(self.X))

                        #RNN
                        self.h_u = tf.zeros([M.shape[0], self.n_conv_feat])
                        self.c_u = tf.zeros([M.shape[0], self.n_conv_feat])


                        for k in range(self.num_iterations):
                            #extraction of global features vectors
                            if cheby==0:
                                self.final_feat_users = self.mono_conv(self.Wrow, self.W, self.W_conv_W, self.b_conv_W)
                            else: #cheby = 1
                                self.final_feat_users = self.mono_conv_cheby(self.list_row_cheb_pol, self.ord_row, self.W, self.W_conv_W, self.b_conv_W)

                            # row RNN
                            self.f_u = tf.sigmoid(tf.matmul(self.final_feat_users, self.W_f_u) + tf.matmul(self.h_u, self.U_f_u) + self.b_f_u)
                            self.i_u = tf.sigmoid(tf.matmul(self.final_feat_users, self.W_i_u) + tf.matmul(self.h_u, self.U_i_u) + self.b_i_u)
                            self.o_u = tf.sigmoid(tf.matmul(self.final_feat_users, self.W_o_u) + tf.matmul(self.h_u, self.U_o_u) + self.b_o_u)

                            self.update_c_u = tf.sigmoid(tf.matmul(self.final_feat_users, self.W_c_u) + tf.matmul(self.h_u, self.U_c_u) + self.b_c_u)
                            self.c_u = tf.multiply(self.f_u, self.c_u) + tf.multiply(self.i_u, self.update_c_u)
                            self.h_u = tf.multiply(self.o_u, tf.sigmoid(self.c_u))  #

                            #compute update of matrix X
                            self.delta_W = tf.tanh(tf.matmul(self.c_u, self.W_out_W) + self.b_out_W)


                            self.W += self.delta_W

                            self.X = tf.matmul(self.W, self.H, transpose_b=True)
                            self.list_X.append(tf.identity(tf.reshape(self.X, [self.M.get_shape().as_list()[0], self.M.get_shape().as_list()[1]])))
                        self.X = tf.matmul(self.W, self.H, transpose_b=True)
                        #########loss definition

                        #computation of the accuracy term
                        self.norm_X = 1+4*(self.X-tf.reduce_min(self.X))/(tf.reduce_max(self.X-tf.reduce_min(self.X)))
                        self.Xnormed= self.norm_tensor(self.X)
                        frob_tensor = tf.multiply(self.Otraining , self.Xnormed - self.M)
                        frob_tensor = tf.multiply(self.mask_features, frob_tensor)
                        self.loss_frob = self.frobenius_norm_square(frob_tensor)/np.sum(Otraining)

                        #computation of the regularization terms
                        trace_row_tensor = tf.matmul(tf.matmul(self.X, self.Lr, transpose_a=True), self.X)
                        self.loss_trace_row = tf.trace(trace_row_tensor)/tf.cast(tf.shape(self.X)[0]*tf.shape(self.X)[1],'float32')

                        self.frob_norm_H = self.frobenius_norm_square(self.H)/tf.cast(tf.shape(self.H)[0]*tf.shape(self.H)[1], 'float32')
                        self.frob_norm_W =  self.frobenius_norm_square(self.W)/tf.cast(tf.shape(self.W)[0]*tf.shape(self.W)[1], 'float32')
                        self.output_nn = tf.slice(self.X, begin = [0, self.M.get_shape().as_list()[1]-1], size = [self.M.get_shape().as_list()[0], 1])
                        self.output_nn=tf.sigmoid(self.output_nn)
                        output_nn_train = (tf.multiply(self.training_set_mask , self.output_nn))
                        self.prediction_train = output_nn_train
                        self.labels_training = tf.multiply(self.training_set_mask , labels)

                        self.binary_entropy = tf.losses.sigmoid_cross_entropy(multi_class_labels = self.labels_training, logits = self.prediction_train)

                        #l2 regularization
                        self.l2_regu=tf.nn.l2_loss(self.W_f_u) + tf.nn.l2_loss(self.W_i_u ) + tf.nn.l2_loss( self.W_o_u) + tf.nn.l2_loss(self.W_c_u)+ tf.nn.l2_loss(self.U_f_u ) + tf.nn.l2_loss(self.U_i_u )+ tf.nn.l2_loss(self.U_o_u ) + tf.nn.l2_loss( self.U_c_u )+ tf.nn.l2_loss(self.W_out_W) + tf.nn.l2_loss(self.W_conv_W)

                        #training loss definition
                        self.loss = self.loss_frob + (gamma)*self.loss_trace_row  + (gamma_W)*self.frob_norm_W+ (gamma_H)*self.frob_norm_H + gamma_e* self.binary_entropy +l2_regu*self.l2_regu#

                        if validation:
                            output_nn_val = (tf.multiply(self.validation_set_mask, self.output_nn))
                            self.predictions_val = output_nn_val
                            self.labels_val = tf.multiply(self.validation_set_mask, labels)

                        output_nn_test = (tf.multiply(self.testing_set_mask, self.output_nn))
                        self.predictions = output_nn_test
                        self.labels_test = tf.multiply(self.testing_set_mask, labels)


                        self.binary_entropy_test = tf.losses.sigmoid_cross_entropy(multi_class_labels = self.labels_test, logits = self.predictions)
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


def running_one_time(validation, rank, l2_regu, gamma, gamma_H, gamma_W, gamma_e, n, seed, cheby, n_conv_feat, lr, ord_row):
    """
    function to run the architecture one time
    Inputs:
       validation :  boolean, to include a validation set or not
       rank : rank of the SVD decomposition
       l2_regu: coefficient to use in front of the l2 regularization term. Default value = 1,
       gamma, gamma_H, gamma_W, gamma_e: coefficients in front of the other loss terms of the loss function,
       n: number of runs of the architecture for the same initialization with the weights and biases of the last run,
       seed: seed to use for the random sampling between training, testing and validation sets,
       cheby: boolean, use of a GCNN or a GCN layer. 0: GCN, 1: GCNN. Default value = 1,
       n_conv_feat: number of weights to use for the GCNN layer. Default value = 36,
       lr: learning rate. Default value = 0.001
       ord_row: int, number of chebyshev polynomials to use for the GCNN
    Outputs:
       auc_test_list, auc_train_list, auc_val_list: lists of the AUC values on the test, train and validation sets for the n runs,
       pred_train_list, pred_test_list, pred_val_list: lists of the prediction values on the test, train and validation sets for the n runs,
       labels_test, labels_train, labels_val: labels of the test, train and validation sets
    """
    #initialization
    training_set_mask, testing_set_mask, idx_training, idx_testing=preprocessing_dataset.split_train_test(0.8, M_str, seed, labels)
    #create a training and test mask on the data
    Otraining = preprocessing_dataset.load_mask(training_set_mask, M_str, nrRows, nrCols)
    Otest= preprocessing_dataset.load_mask(testing_set_mask, M_str, nrRows, nrCols)

    if validation:
        #split train set into 4 parts to create a validation set
        new_labels_train=np.copy(labels)
        new_labels_train[idx_testing]=-1
        training_set_mask, validation_set_mask, idx_training, idx_validation=preprocessing_dataset.split_train_validation_4(3, M_str, seed, new_labels_train)
        Otraining = preprocessing_dataset.load_mask(training_set_mask, M_str, nrRows, nrCols)
        Ovalidation= preprocessing_dataset.load_mask(validation_set_mask, M_str, nrRows, nrCols)
    else:
        validation_set_mask= []
        idx_validation=[]

    Odata, _=preprocessing_dataset.split_train_into_2(Otraining, seed)
    Otraining = np.concatenate((Otraining, training_set_mask), axis=1)
    if validation:
        Oinitial=np.concatenate((Odata+Otest+Ovalidation, training_set_mask), axis=1)
    else:
        Oinitial=np.concatenate((Odata+Otest, training_set_mask), axis=1)


    M_init_final=preprocessing_dataset.init_mask(Oinitial, M_init, labels)
    #apply SVD initially for detecting the main components of our initialization
    initial_W, initial_H= preprocessing_dataset.svd_W_H(M_init_final, rank)

    learning_obj = Train_test_matrix_completion(M, Lrow, Wrow, Otraining,
                                                        initial_W, initial_H,  labels, training_set_mask, testing_set_mask, validation_set_mask, mask_features,validation,
                                                        order_chebyshev_row = ord_row, cheby=cheby, n_conv_feat=n_conv_feat, l2_regu=l2_regu,
                                                        gamma=gamma, gamma_H=gamma_H, gamma_W=gamma_W, gamma_e=gamma_e, learning_rate=lr)

    num_iter_test = 10
    num_total_iter_training = n
    num_iter = 0

    list_training_loss = list()
    list_training_norm_grad = list()
    list_test_pred_error = list()

    list_test_times = list()
    list_grad_X = list()


    auc_train_list=[]
    pred_train_list=[]

    auc_val_list=[]
    pred_val_list=[]
    auc_test_list=[]
    pred_test_list=[]


    num_iter = 0
    for k in range(num_iter, num_total_iter_training):

        tic = time.time()
        list_of_outputs = learning_obj.session.run([learning_obj.optimizer, learning_obj.loss, learning_obj.norm_grad, learning_obj.prediction_train, learning_obj.labels_training, learning_obj.loss_frob, learning_obj.loss_trace_row, learning_obj.frob_norm_H, learning_obj.frob_norm_W, learning_obj.binary_entropy] + learning_obj.var_grad)
        current_training_loss = list_of_outputs[1]
        norm_grad = list_of_outputs[2]
        pred_train = list_of_outputs[3]
        labels_train =list_of_outputs[4]
        loss_frob = list_of_outputs[5]
        loss_trace = list_of_outputs[6]
        loss_norm_H =list_of_outputs[7]
        loss_norm_W =list_of_outputs[8]
        loss_entropy = list_of_outputs[9]
        indexes_train = np.concatenate((idx_testing, idx_validation), axis=0)

        pred_train = np.delete(pred_train, indexes_train,0)
        labels_train = np.delete(labels_train, indexes_train,0)

        accuracy_train, tn_train, fn_train, tp_train, fp_train, spe_train, sen_train=preprocessing_dataset.accuracy_computation(pred_train, training_set_mask, labels)
        fpr_train, tpr_train, thresholds_train=roc_curve(labels_train, pred_train)
        roc_auc_train = auc(fpr_train, tpr_train)

        X_grad = list_of_outputs[10:]
        training_time = time.time() - tic

        list_training_loss.append(current_training_loss)
        list_training_norm_grad.append(norm_grad)

        if (np.mod(num_iter, num_iter_test)==0):
            msg = "[TRN] iter = %03i, cost = %3.2e, |grad| = %.2e (%3.2es), accuracy = %3.2e, auc = %3.2e" \
                                        % (num_iter, list_training_loss[-1], list_training_norm_grad[-1], training_time, accuracy_train, roc_auc_train)
            print msg

            auc_train_list.append(roc_auc_train)
            pred_train_list.append(pred_train)

            #Test Code
            tic = time.time()
            pred_error, pred, labels_test= learning_obj.session.run([learning_obj.predictions_error, learning_obj.predictions, learning_obj.labels_test])
            test_time = time.time() - tic

            if validation:
                pred_val, labels_val=learning_obj.session.run([learning_obj.predictions_val, learning_obj.labels_val])

                indexes_validation = np.concatenate((idx_training, idx_testing), axis=0)
                pred_val = np.delete(pred_val, indexes_validation,0)
                labels_val = np.delete(labels_val, indexes_validation,0)
                fpr, tpr, thresholds=roc_curve(labels_val, pred_val)
                roc_auc_val = auc(fpr, tpr)
                auc_val_list.append(roc_auc_val)
                pred_val_list.append(pred_val)
                msg =  "[VAL] iter = %03i, AUC = %3.2e" % (num_iter, roc_auc_val)
                print msg
            else:
                pred_val_list=[]
                labels_val=[]


            indexes_test = np.concatenate((idx_training, idx_validation), axis=0)

            pred = np.delete(pred, indexes_test,0)
            labels_test = np.delete(labels_test, indexes_test,0)

            list_test_pred_error.append(pred_error)

            accuracy, tn, fn, tp, fp, spe, sen= preprocessing_dataset.accuracy_computation(pred, testing_set_mask, labels)
            fpr, tpr, thresholds=roc_curve(labels_test, pred)
            roc_auc = auc(fpr, tpr)

            auc_test_list.append(roc_auc)
            pred_test_list.append(pred)

            msg =  "[TST] iter = %03i, cost = %3.2e, Accuracy = %3.2e (%3.2es), AUC = %3.2e" % (num_iter, list_test_pred_error[-1], accuracy, test_time, roc_auc)
            print msg

        num_iter += 1
    return (auc_test_list, auc_train_list, auc_val_list, pred_train_list, pred_test_list, pred_val_list, labels_test, labels_train,labels_val)

##############################################################################################################################
##
##
##code to optimize the hyperparameters with RBFOpt
##
##
################################################################################################################################
#opimization of the hyperparameters of the loss function
def optimize_loss(x):
    """
    Function to optimize the hyperparameters of the loss function with RBFOpt. The other hyperparameters are fixed
    Input:
        x: value given by RBFOpt for the hyperparameters  by running alg.optimize()
    Output: opposite of the max validation AUC (function to minimize)
    """
    print(x)
    l2_regu, gamma, gamma_H, gamma_W, gamma_e=x
    nb_itera =1000
    seed = 0
    auc_test_list, auc_train_list, auc_val_list, pred_train_list, pred_test_list, pred_val_list, labels_test, labels_train,labels_val= running_one_time(l2_regu, gamma, gamma_H, gamma_W, gamma_e, nb_itera, seed, 1, 36, 0.00089)
    auc_test.append(auc_test_list)
    auc_train.append(auc_train_list)
    pred_test.append(pred_test_list)
    pred_train.append(pred_train_list)
    auc_val.append(auc_val_list)
    pred_val.append(pred_val_list)
    x_list.append(x)

    max_auc_val=np.max(auc_val_list)
    print(max_auc_val)
    return -max_auc_val
"""
bb=rbfopt.RbfoptUserBlackBox(5, np.array([1]*5), np.array([100]*5), np.array(['R', 'R', 'R', 'R', 'R']), optimize_loss)
settings=rbfopt.RbfoptSettings(max_iterations=200,max_evaluations=200, minlp_solver_path='~/TADPOLE/bonmin', nlp_solver_path='~/TADPOLE/ipopt', target_objval=-1)
alg=rbfopt.RbfoptAlgorithm(settings, bb)
objval, x, itercount, evalcount, fast_evalcount=alg.optimize()
print(objval, x, itercount, evalcount, fast_evalcount)
"""

def optimize(x):
    """
    Function to optimize the number of hidden units of the GCNN layer with RBFOpt. The other hyperparameters are fixed
    Input:
        x: value given by RBFOpt for the hyperparameters  by running alg.optimize()
    Output: opposite of the max validation AUC (function to minimize)
    """
    print(x)
    nb_hidden=x
    nb_itera =1000
    seed = 0
    l2_regu, gamma, gamma_H, gamma_W, gamma_e= 17.07040641, 99.96903513, 67.66640563, 68.59572245, 69.86587889
    auc_test_list, auc_train_list, auc_val_list, pred_train_list, pred_test_list, pred_val_list, labels_test, labels_train,labels_val= running_one_time(l2_regu, gamma, gamma_H, gamma_W, gamma_e, nb_itera, seed, 1, nb_hidden, lr)
    max_auc_val=np.max(auc_val_list)
    print(max_auc_val)
    return -max_auc_val

"""
bb=rbfopt.RbfoptUserBlackBox(1, np.array([1]), np.array([50]), np.array(['I']), optimize)
settings=rbfopt.RbfoptSettings(max_iterations=50,max_evaluations=50, minlp_solver_path='~/TADPOLE/bonmin', nlp_solver_path='~/TADPOLE/ipopt', target_objval=-1)
alg=rbfopt.RbfoptAlgorithm(settings, bb)
objval, x, itercount, evalcount, fast_evalcount=alg.optimize()
print(objval, x, itercount, evalcount, fast_evalcount)
"""

def optimize(x):
    """
    Function to optimize the learning rate with RBFOpt. The other hyperparameters are fixed
    Input:
        x: value given by RBFOpt for the hyperparameters  by running alg.optimize()
    Output: opposite of the max validation AUC (function to minimize)
    """
    lr=x[0]
    print(lr)
    nb_hidden=17
    nb_itera =1000
    seed = 0
    l2_regu, gamma, gamma_H, gamma_W, gamma_e= 17.07040641, 99.96903513, 67.66640563, 68.59572245, 69.86587889
    auc_test_list, auc_train_list, auc_val_list, pred_train_list, pred_test_list, pred_val_list, labels_test, labels_train,labels_val= running_one_time(l2_regu, gamma, gamma_H, gamma_W, gamma_e, nb_itera, seed, 1, nb_hidden, lr)
    max_auc_val=np.max(auc_val_list)
    print(max_auc_val)
    return -max_auc_val

"""
bb=rbfopt.RbfoptUserBlackBox(1, np.array([0.005]), np.array([0.00005]), np.array(['R']), optimize)
settings=rbfopt.RbfoptSettings(max_iterations=100,max_evaluations=100, minlp_solver_path='~/TADPOLE/bonmin', nlp_solver_path='~/TADPOLE/ipopt', target_objval=-1)
alg=rbfopt.RbfoptAlgorithm(settings, bb)
objval, x, itercount, evalcount, fast_evalcount=alg.optimize()
print(objval, x, itercount, evalcount, fast_evalcount)
"""

def optimize(x):
    """
    Function to optimize the number of Chebyshev polynomials with RBFOpt. The other hyperparameters are fixed
    Input:
        x: value given by RBFOpt for the hyperparameters  by running alg.optimize()
    Output: opposite of the max validation AUC (function to minimize)
    """
    ord_row=x[0]
    ord_row=int(ord_row)
    print(ord_row)
    lr=0.00030738
    nb_hidden=17
    nb_itera =1000
    seed = 0
    l2_regu, gamma, gamma_H, gamma_W, gamma_e= 17.07040641, 99.96903513, 67.66640563, 68.59572245, 69.86587889
    auc_test_list, auc_train_list, auc_val_list, pred_train_list, pred_test_list, pred_val_list, labels_test, labels_train,labels_val= running_one_time(l2_regu, gamma, gamma_H, gamma_W, gamma_e, nb_itera, seed, 1, nb_hidden, lr, ord_row)
    max_auc_val=np.max(auc_val_list)
    print(max_auc_val)
    return -max_auc_val

"""
bb=rbfopt.RbfoptUserBlackBox(1, np.array([1]), np.array([50]), np.array(['I']), optimize)
settings=rbfopt.RbfoptSettings(max_iterations=50,max_evaluations=50, minlp_solver_path='~/TADPOLE/bonmin', nlp_solver_path='~/TADPOLE/ipopt', target_objval=-1)
alg=rbfopt.RbfoptAlgorithm(settings, bb)
objval, x, itercount, evalcount, fast_evalcount=alg.optimize()
print(objval, x, itercount, evalcount, fast_evalcount)
"""
def optimize(x):
    """
    Function to optimize the rank for SVD with RBFOpt. The other hyperparameters are fixed
    Input:
        x: value given by RBFOpt for the hyperparameters  by running alg.optimize()
    Output: opposite of the max validation AUC (function to minimize)
    """
    rank=x
    ord_row=18
    lr=0.00030738
    nb_hidden=17
    nb_itera =3500
    seed = 0
    l2_regu, gamma, gamma_H, gamma_W, gamma_e= 17.07040641, 99.96903513, 67.66640563, 68.59572245, 69.86587889
    auc_test_list, auc_train_list, auc_val_list, pred_train_list, pred_test_list, pred_val_list, labels_test, labels_train,labels_val= running_one_time(False, rank, l2_regu, gamma, gamma_H, gamma_W, gamma_e, nb_itera, seed, 1, nb_hidden, lr, ord_row)
    max_auc_val=np.max(auc_val_list)
    print(max_auc_val)
    return -max_auc_val
"""
bb=rbfopt.RbfoptUserBlackBox(1, np.array([1]), np.array([500]), np.array(['I']), optimize)
settings=rbfopt.RbfoptSettings(max_iterations=100,max_evaluations=100, minlp_solver_path='~/TADPOLE/bonmin', nlp_solver_path='~/TADPOLE/ipopt', target_objval=-1)
alg=rbfopt.RbfoptAlgorithm(settings, bb)
objval, x, itercount, evalcount, fast_evalcount=alg.optimize()
print(objval, x, itercount, evalcount, fast_evalcount)
"""

nb_seed= 3
save=False
auc_test=[]

for seed in range(nb_seed):
    print('seed', seed)
    dropout=1
    nb_itera = 1000
    lr=0.00030738
    ord_row=18
    lr=0.00030738
    nb_hidden=17
    rank=156
    l2_regu, gamma, gamma_H, gamma_W, gamma_e= 17.07040641, 99.96903513, 67.66640563, 68.59572245, 69.86587889
    auc_test_list, auc_train_list, auc_val_list, pred_train_list, pred_test_list, pred_val_list, labels_test, labels_train,labels_val= running_one_time(True, rank, l2_regu, gamma, gamma_H, gamma_W, gamma_e, nb_itera, seed, 1, nb_hidden, lr, ord_row)
    iteration_max_val = np.argmax(auc_val_list)
    auc_test_value = auc_test_list[iteration_max_val]
    print('auc test', auc_test_value)
    pred_test_list_final = pred_test_list[iteration_max_val]
    auc_test.append(auc_test_value)
    fpr, tpr, thresholds=roc_curve(labels_test, pred_test_list_final)
    roc_auc = auc(fpr, tpr)
    print('auc pred', roc_auc)
    if save:
        np.savetxt('pred_test_sRGCNN_simi_seed%i_0.8.csv' %(seed) , pred_test_list_final,fmt='%s',delimiter=',')
        np.savetxt('labels_test_set_index_sRGCNN_simi_seed%i_0.8.csv' %(seed), labels_test,fmt='%s',delimiter=',')


print(auc_test)
if save:
    np.savetxt('auc_test_sRGCNN_simi_seed%i_0.8.csv' , auc_test,fmt='%s',delimiter=',')

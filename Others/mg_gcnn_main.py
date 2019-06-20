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
#import rbfopt FOR OPTIMIZATION

import preprocessing_dataset
import read_tadpole


###############################
##
## csv files required to run the program
##
##############################
path_dataset_matrix = 'bl_data_MCI_v1.csv'
path_mask_X1 = 'mask_age_nolab.csv'
path_mask_X2 = 'mask_sex_nolab.csv'
path_mask_X3 = 'mask_agesex_nolab.csv'
path_mask_X4 = 'mask_nosignificance_nolab.csv'

path_dataset_affinity_matrix_age = 'affinity_mat_1.csv'
path_dataset_affinity_matrix_sex = 'affinity_mat_2.csv'
path_dataset_affinity_matrix_agesex = 'affinity_mat_3.csv'


labels_path = 'labels_comma.csv'
#load the .csv files
M_str, nrRows, nrCols = read_tadpole.load_csv_no_header(path_dataset_matrix)
mask_age, _, _ = read_tadpole.load_csv_no_header(path_mask_X1)
mask_sex, _, _ = read_tadpole.load_csv_no_header(path_mask_X2)
mask_agesex, _, _ = read_tadpole.load_csv_no_header(path_mask_X3)
mask_nosignificance, _, _ = read_tadpole.load_csv_no_header(path_mask_X4)

A1, _, _ = read_tadpole.load_csv_no_header(path_dataset_affinity_matrix_age)
A2, _, _ = read_tadpole.load_csv_no_header(path_dataset_affinity_matrix_sex)
A3, _, _ = read_tadpole.load_csv_no_header(path_dataset_affinity_matrix_agesex)
A_age=preprocessing_dataset.str_to_float(A1)
A_sex=preprocessing_dataset.str_to_float(A2)
A_sexage=preprocessing_dataset.str_to_float(A3)
labels, _, _ = read_tadpole.load_csv_no_header(labels_path)

labels = preprocessing_dataset.str_to_float(labels)


M_float = preprocessing_dataset.preprocessing_nan_normalization(M_str)

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp = imp.fit(M_float)

# Impute our data, then train
M_float_imp = imp.transform(M_float)

M = M_float_imp

A_age=preprocessing_dataset.normalize_adj(A_age)
A_sex=preprocessing_dataset.normalize_adj(A_sex)
A_sexage=preprocessing_dataset.normalize_adj(A_sexage)

mask_age=preprocessing_dataset.str_to_float(mask_age)
mask_sex= preprocessing_dataset.str_to_float(mask_sex)
mask_agesex=preprocessing_dataset.str_to_float(mask_agesex)
mask_nosignificance=preprocessing_dataset.str_to_float(mask_nosignificance)

#computation of the normalized laplacians
Lrow_age = csgraph.laplacian(A_age, normed=True)
Lrow_sex = csgraph.laplacian(A_sex, normed=True)
Lrow_agesex = csgraph.laplacian(A_sexage, normed=True)

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

    def __init__(self, M, L_a, L_s, L_sa, A_a, A_s, A_sa, mask_a, mask_s, mask_as, mask_ns,
                    labels_train, labels_test, labels_val, testing_set_mask, training_set_mask,validation_set_mask,
                 order_chebyshev_row = 3, n_conv_feat=32, dropout=0.98,
                 learning_rate=0.01, l2_regu=0.00001, nb_layers=5, idx_gpu = '/gpu:1'):
        """
                 Neural network architecture. Output classification result for each row of M.
                 Inputs:
                    M: initial matrix with all the known values,
                    A_a, A_s, A_sa : adjacency matrices, respectively for the age, sex and age and sex graphs,
                    mask_a, mask_s, mask_as, mask_ns: masks to apply to X for each one of the graphs,
                    L_a, L_s, L_sa: laplacian matrices, respectively for the age, sex and age and sex graphs,
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
                        self.L_s = tf.cast(L_s, 'float32')
                        self.norm_L_s = self.L_s - tf.diag(tf.ones([L_s.shape[0], ]))

                        #compute all chebyshev polynomials a priori
                        self.list_row_cheb_pol_sex = list()
                        self.compute_cheb_polynomials(self.norm_L_s, self.ord_row, self.list_row_cheb_pol_sex)

                        self.L_a = tf.cast(L_a, 'float32')
                        self.norm_L_a = self.L_a - tf.diag(tf.ones([L_a.shape[0], ]))

                        #compute all chebyshev polynomials a priori
                        self.list_row_cheb_pol_age = list()
                        self.compute_cheb_polynomials(self.norm_L_a, self.ord_row, self.list_row_cheb_pol_age)

                        self.L_sa = tf.cast(L_a, 'float32')
                        self.norm_L_sa = self.L_sa - tf.diag(tf.ones([L_sa.shape[0], ]))

                        #compute all chebyshev polynomials a priori
                        self.list_row_cheb_pol_agesex = list()
                        self.compute_cheb_polynomials(self.norm_L_sa, self.ord_row, self.list_row_cheb_pol_agesex)

                        #definition of constant matrices
                        self.A_a=tf.constant(A_a, dtype=tf.float32)
                        self.A_s=tf.constant(A_s, dtype=tf.float32)
                        self.A_sa=tf.constant(A_sa, dtype=tf.float32)

                        self.mask_a=tf.constant(mask_a, dtype=tf.float32)
                        self.mask_s=tf.constant(mask_s, dtype=tf.float32)
                        self.mask_ns=tf.constant(mask_ns, dtype=tf.float32)
                        self.mask_as=tf.constant(mask_as, dtype=tf.float32)

                        self.M = tf.constant(M, dtype=tf.float32)
                        self.labels_train=tf.constant(labels_train, dtype=tf.float32)
                        self.labels_test=tf.constant(labels_test, dtype=tf.float32)
                        self.labels_val=tf.constant(labels_val, dtype=tf.float32)
                        self.testing_mask=tf.constant(testing_set_mask, dtype=tf.float32)
                        self.training_mask=tf.constant(training_set_mask, dtype=tf.float32)
                        self.validation_mask=tf.constant(validation_set_mask, dtype=tf.float32)

                        ##################################definition of the NN variables#####################################
                        #definition of the weights for extracting the global features
                        weights_age=[]
                        bias_age=[]
                        weights_sex=[]
                        bias_sex=[]
                        weights_as=[]
                        bias_as=[]
                        weights_ns=[]
                        bias_ns=[]
                        if nb_layers==1:
                            weights_age.append(tf.get_variable('weights_Wa', shape=[self.ord_row*M.shape[1], self.n_conv_feat], initializer=tf.contrib.layers.xavier_initializer()))#self.n_conv_feat
                            bias_age.append( tf.Variable(tf.zeros([self.n_conv_feat,])))
                            weights_sex.append(tf.get_variable('weights_Ws', shape=[self.ord_row*M.shape[1], self.n_conv_feat], initializer=tf.contrib.layers.xavier_initializer()))#self.n_conv_feat
                            bias_sex.append( tf.Variable(tf.zeros([self.n_conv_feat,])))
                            weights_as.append(tf.get_variable('weights_Was', shape=[self.ord_row*M.shape[1], self.n_conv_feat], initializer=tf.contrib.layers.xavier_initializer()))#self.n_conv_feat
                            bias_as.append( tf.Variable(tf.zeros([self.n_conv_feat,])))
                            weights_ns.append(tf.get_variable('weights_Wns', shape=[M.shape[1], self.n_conv_feat], initializer=tf.contrib.layers.xavier_initializer()))#self.n_conv_feat
                            bias_ns.append( tf.Variable(tf.zeros([self.n_conv_feat,])))
                            weights_age.append(tf.get_variable("weight_final", shape=[4*self.n_conv_feat, 1], initializer=tf.contrib.layers.xavier_initializer()))
                            bias_age.append(tf.Variable(tf.zeros([1,])))
                        else:
                            weights_age.append(tf.get_variable('weights_Wa', shape=[self.ord_row*M.shape[1], self.n_conv_feat], initializer=tf.contrib.layers.xavier_initializer()))#self.n_conv_feat
                            bias_age.append(tf.Variable(tf.zeros([self.n_conv_feat,])))
                            for i in range(nb_layers-1):
                                weights_age.append(tf.get_variable('weightsa_%d' %i, shape=[self.ord_row*self.n_conv_feat, self.n_conv_feat], initializer=tf.contrib.layers.xavier_initializer()))#self.n_conv_feat
                                bias_age.append(tf.Variable(tf.zeros([self.n_conv_feat,])))
                            weights_sex.append(tf.get_variable('weights_Ws', shape=[self.ord_row*M.shape[1], self.n_conv_feat], initializer=tf.contrib.layers.xavier_initializer()))#self.n_conv_feat
                            bias_sex.append(tf.Variable(tf.zeros([self.n_conv_feat,])))
                            for i in range(nb_layers-1):
                                weights_sex.append(tf.get_variable('weightss_%d' %i, shape=[self.ord_row*self.n_conv_feat, self.n_conv_feat], initializer=tf.contrib.layers.xavier_initializer()))#self.n_conv_feat
                                bias_sex.append(tf.Variable(tf.zeros([self.n_conv_feat,])))
                            weights_as.append(tf.get_variable('weights_Was', shape=[self.ord_row*M.shape[1], self.n_conv_feat], initializer=tf.contrib.layers.xavier_initializer()))#self.n_conv_feat
                            bias_as.append(tf.Variable(tf.zeros([self.n_conv_feat,])))
                            for i in range(nb_layers-1):
                                weights_as.append(tf.get_variable('weightsas_%d' %i, shape=[self.ord_row*self.n_conv_feat, self.n_conv_feat], initializer=tf.contrib.layers.xavier_initializer()))#self.n_conv_feat
                                bias_as.append(tf.Variable(tf.zeros([self.n_conv_feat,])))
                            weights_ns.append(tf.get_variable('weights_Wns', shape=[M.shape[1], self.n_conv_feat], initializer=tf.contrib.layers.xavier_initializer()))#self.n_conv_feat
                            bias_ns.append(tf.Variable(tf.zeros([self.n_conv_feat,])))
                            for i in range(nb_layers-1):
                                weights_ns.append(tf.get_variable('weightsns_%d' %i, shape=[self.n_conv_feat, self.n_conv_feat], initializer=tf.contrib.layers.xavier_initializer()))#self.n_conv_feat
                                bias_ns.append(tf.Variable(tf.zeros([self.n_conv_feat,])))

                            weights_age.append(tf.get_variable("weight_final", shape=[4*self.n_conv_feat, 1], initializer=tf.contrib.layers.xavier_initializer()))
                            bias_age.append(tf.Variable(tf.zeros([1,])))

                        # GCNN architecture TRAINING
                        if nb_layers==1:
                            X1= tf.multiply(self.mask_s , self.M)
                            self.final_feat_sex = self.mono_conv_cheby(self.list_row_cheb_pol_sex, self.ord_row, X1, weights_sex[0], bias_sex[0])
                            self.final_feat_sex = tf.nn.relu(self.final_feat_sex)
                            self.final_feat_sex = tf.nn.dropout(self.final_feat_sex, dropout)
                            X2= tf.multiply(self.mask_a , self.M)
                            self.final_feat_age = self.mono_conv_cheby(self.list_row_cheb_pol_age, self.ord_row, X2, weights_age[0], bias_age[0])
                            self.final_feat_age = tf.nn.relu(self.final_feat_age)
                            self.final_feat_age = tf.nn.dropout(self.final_feat_age, dropout)
                            X3= tf.multiply(self.mask_as , self.M)
                            self.final_feat_agesex = self.mono_conv_cheby(self.list_row_cheb_pol_agesex, self.ord_row, X3, weights_as[0], bias_as[0])
                            self.final_feat_agesex = tf.nn.relu(self.final_feat_agesex)
                            self.final_feat_agesex = tf.nn.dropout(self.final_feat_agesex, dropout)
                            X4= tf.multiply(self.mask_ns , self.M)
                            self.final_feat_ns = tf.matmul(X4, weights_ns[0]) + bias_ns[0]
                            self.final_feat_ns = tf.nn.relu(self.final_feat_ns)
                            self.final_feat_ns = tf.nn.dropout(self.final_feat_ns, dropout)

                            self.final_feat=tf.concat([self.final_feat_age, self.final_feat_sex, self.final_feat_agesex, self.final_feat_ns], 1)
                            self.final_feat = tf.matmul(self.final_feat, weights_age[-1]) + bias_age[-1]
                        else:
                            X1= tf.multiply(self.mask_s , self.M)
                            self.final_feat_sex = self.mono_conv_cheby(self.list_row_cheb_pol_sex, self.ord_row, X1, weights_sex[0], bias_sex[0]) #shape 3000*32
                            self.final_feat_sex = tf.nn.relu(self.final_feat_sex)
                            self.final_feat_sex = tf.nn.dropout(self.final_feat_sex, dropout)
                            for i in range(nb_layers-1):
                                self.final_feat_sex = self.mono_conv_cheby(self.list_row_cheb_pol_sex, self.ord_row, self.final_feat_sex, weights_sex[i+1], bias_sex[i+1])
                                self.final_feat_sex = tf.nn.relu(self.final_feat_sex)
                                self.final_feat_sex = tf.nn.dropout(self.final_feat_sex, dropout)
                            X2= tf.multiply(self.mask_a , self.M)
                            self.final_feat_age = self.mono_conv_cheby(self.list_row_cheb_pol_age, self.ord_row, X2, weights_age[0], bias_age[0])
                            self.final_feat_age = tf.nn.relu(self.final_feat_age)
                            self.final_feat_age = tf.nn.dropout(self.final_feat_age, dropout)
                            for i in range(nb_layers-1):
                                self.final_feat_age = self.mono_conv_cheby(self.list_row_cheb_pol_age, self.ord_row, self.final_feat_age, weights_age[i+1], bias_age[i+1])
                                self.final_feat_age = tf.nn.relu(self.final_feat_age)
                                self.final_feat_age = tf.nn.dropout(self.final_feat_age, dropout)
                            X3= tf.multiply(self.mask_as , self.M)
                            self.final_feat_agesex = self.mono_conv_cheby(self.list_row_cheb_pol_agesex, self.ord_row, X3, weights_as[0], bias_as[0])
                            self.final_feat_agesex = tf.nn.relu(self.final_feat_agesex)
                            self.final_feat_agesex = tf.nn.dropout(self.final_feat_agesex, dropout)
                            for i in range(nb_layers-1):
                                self.final_feat_agesex = self.mono_conv_cheby(self.list_row_cheb_pol_agesex, self.ord_row, self.final_feat_agesex, weights_as[i+1], bias_as[i+1])
                                self.final_feat_agesex = tf.nn.relu(self.final_feat_agesex)
                                self.final_feat_agesex = tf.nn.dropout(self.final_feat_agesex, dropout)

                            X4= tf.multiply(self.mask_ns , self.M)
                            self.final_feat_ns = tf.matmul(X4, weights_ns[0]) + bias_ns[0]
                            self.final_feat_ns = tf.nn.relu(self.final_feat_ns)
                            self.final_feat_ns = tf.nn.dropout(self.final_feat_ns, dropout)
                            for i in range(nb_layers-1):
                                self.final_feat_ns = tf.matmul(self.final_feat_ns, weights_ns[i+1]) + bias_ns[i+1]
                                self.final_feat_ns = tf.nn.relu(self.final_feat_ns)
                                self.final_feat_ns = tf.nn.dropout(self.final_feat_ns, dropout)

                            self.final_feat=tf.concat([self.final_feat_age, self.final_feat_sex, self.final_feat_agesex, self.final_feat_ns], 1)
                            self.final_feat = tf.matmul(self.final_feat, weights_age[-1]) + bias_age[-1]

                        self.classification=tf.sigmoid(self.final_feat)

                        #########loss definition

                        #computation of the accuracy term
                        self.classification_train = tf.multiply(self.training_mask, self.classification)
                        self.classification_test= tf.multiply(self.testing_mask, self.classification)
                        self.classification_val= tf.multiply(self.validation_mask, self.classification)


                        self.binary_entropy = tf.losses.sigmoid_cross_entropy(multi_class_labels = self.labels_train, logits = self.classification_train)
                        self.l2=0
                        for i in range(len(weights_ns)):
                            self.l2+= tf.nn.l2_loss(weights_ns[i])+tf.nn.l2_loss(weights_as[i])+tf.nn.l2_loss(weights_sex[i])+tf.nn.l2_loss(weights_age[i])
                        self.l2+= tf.nn.l2_loss(weights_age[-1])
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
       ord_row: number of Chebyshev polynomials for the GCNN layers
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

    learning_obj = Train_test_matrix_completion(M, Lrow_age, Lrow_sex, Lrow_agesex, A_age, A_sex, A_sexage, mask_age, mask_sex, mask_agesex, mask_nosignificance,
                                                        labels_train, labels_test, labels_val, testing_set_mask, training_set_mask, validation_set_mask,
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
        # run of the algorithm on the training set
        list_of_outputs = learning_obj.session.run([learning_obj.optimizer, learning_obj.loss, learning_obj.norm_grad, learning_obj.classification_train, learning_obj.binary_entropy] + learning_obj.var_grad)#learning_obj.loss_frob, learning_obj.loss_trace_row, learning_obj.frob_norm_H, learning_obj.frob_norm_W, learning_obj.binary_entropy
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
            msg = "[TRN] iter = %03i, cost = %3.2e, |grad| = %.2e (%3.2es), AUC = %3.2e" \
                                        % (num_iter, list_training_loss[-1], list_training_norm_grad[-1], training_time, roc_auc_train)
            print msg

            auc_train_list.append(roc_auc_train)
            pred_train_list.append(pred_train)

            tic = time.time()
            # run of the algorithm on the validation set
            pred_val = learning_obj.session.run([learning_obj.classification_val])#
            test_time = time.time() - tic
            pred_val = np.delete(pred_val[0], indexes_validation,0)
            fpr, tpr, thresholds=roc_curve(labels_val_reduce, pred_val)
            roc_auc_val = auc(fpr, tpr)
            auc_val_list.append(roc_auc_val)
            pred_val_list.append(pred_val)

            msg =  "[VAL] iter = %03i, AUC = %3.2e" % (num_iter, roc_auc_val)
            print msg

            tic = time.time()
            # run of the algorithm on the test set
            pred_error, pred = learning_obj.session.run([learning_obj.predictions_error, learning_obj.classification_test])#
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

### MAIN CODE


def optimize(x):
    """
    Function to optimize the hyperparameters of the loss function with RBFOpt. The other hyperparameters are fixed
    Input:
        x: value given by RBFOpt for the hyperparameters  by running alg.optimize()
    Output: opposite of the max validation AUC (function to minimize)
    """
    seed=0
    nb_itera =500
    print(x)
    dropout, n_conv_feat, lr, l2_regu,nb_layers,ord_row=x
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

"""bb=rbfopt.RbfoptUserBlackBox(6, np.array([0.5, 1, 0.0005, 0.000001, 1, 1]), np.array([1.0, 100, 0.05, 0.0001, 10, 50]), np.array(['R', 'I', 'R', 'R', 'I', 'I']), optimize)
settings=rbfopt.RbfoptSettings(max_iterations=500,max_noisy_evaluations=500,max_evaluations=500, minlp_solver_path='~/TADPOLE/bonmin', nlp_solver_path='~/TADPOLE/ipopt', target_objval=-1)
alg=rbfopt.RbfoptAlgorithm(settings, bb)
objval, x, itercount, evalcount, fast_evalcount=alg.optimize()
"""

#run of the architecture for a number nb_seed of different training/validation/testing initializations
nb_seeds=3
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
    print('seed', seed)
    nb_itera =500
    dropout, n_conv_feat, lr, l2_regu,nb_layers,ord_row=0.519915912, 16, 0.000545564811, 0.0000983892114, 1, 2
    n_conv_feat=int(n_conv_feat)
    nb_layers=int(nb_layers)

    auc_test_list, auc_train_list, auc_val_list, pred_train_list, pred_test_list, pred_val_list, labels_test_reduce, labels_train_reduce, labels_val_reduce= running_one_time(nb_itera, dropout, seed, n_conv_feat, lr, l2_regu, nb_layers, ord_row)

    iteration_max_val = np.argmax(auc_val_list)
    auc_test_value = auc_test_list[iteration_max_val]

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


print(auc_test)
"""np.savetxt('mgcnn_auc_values_test_500.csv', auc_test, fmt='%f', delimiter=',')
np.savetxt('mgcnn_auc_values_train_500.csv', auc_train, fmt='%f', delimiter=',')
np.savetxt('mgcnn_auc_values_val_500.csv', auc_val, fmt='%f', delimiter=',')
np.savetxt('mgcnn_pred_test_500.csv', pred_test, fmt='%f', delimiter=',')
np.savetxt('mgcnn_pred_train_500.csv', pred_train, fmt='%f', delimiter=',')
np.savetxt('mgcnn_pred_val_500.csv', pred_val, fmt='%f', delimiter=',')
np.savetxt('mgcnn_labels_train_500.csv', labels_train, fmt='%f', delimiter=',')
np.savetxt('mgcnn_labels_test_500.csv', labels_test,fmt='%s',delimiter=',')
np.savetxt('mgcnn_labels_val_500.csv', labels_val, fmt='%s',delimiter=',')
"""

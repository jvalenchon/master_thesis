# author : Juliette Valenchon

#import of libraries
import csv
import numpy as np
from sklearn.metrics import roc_curve, auc


#functions for the preprocessing of the dataset


def load_csv_no_header(filePath):
    """
    function to load a .csv file into a matrix
    input:
        filePath: filePath to the csv file
    outputs:
        mergeAll : Matrix of str with the data in the csv file
        nrRows: Number of rows of the matrix
        nrCols: Number of columns of the matrix
    """
    with open(filePath, 'r') as f:
        reader = csv.reader(f, delimiter = ',', quotechar = "'")
        rows = [row for row in reader]
        nrRows = len(rows)
        nrCols = len(rows[0])
        mergeAll = np.ndarray((nrRows, nrCols), dtype=str)
        mergeAll[:] = ' '
        for r in range(nrRows):
            mergeAll[r,:] = [str.encode(word) for word in rows[r]]
    return mergeAll, nrRows, nrCols

def load_csv_no_header_float(filePath):
    """
    Load data of csv file into a matrix. The csv file is only composed of data, no header. The data is composed of floats only
    Input:
        filePath : file path to the .csv file
    Outputs:
        mergeAll: matrix of float with the data in the file path.
        nrRows: int, number of rows of the matrix
        nrCols: int, number of columns of the matrix
    """
    with open(filePath, 'r') as f:
        reader = csv.reader(f, delimiter = ',', quotechar = "'")
        rows = [row for row in reader]
        nrRows = len(rows)
        nrCols = len(rows[0])
        mergeAll = np.ndarray((nrRows, nrCols), dtype=np.float32)
        for r in range(nrRows):
            mergeAll[r,:] = [word for word in rows[r]]
    return mergeAll, nrRows, nrCols

#functions for the initialization and the mask creation
def normalization(M):
    """
    Normalize a matrix between -1 and 1. Put 0 values to missing values of the matrix.
    Input:
        M: matrix that we want to normalize.
    Output:
        M_norm: matrix that is normalized and that has 0 values for the missing values.
    """
    M_norm = np.ndarray((M.shape))
    for j in range(M.shape[1]):
        max = 0
        min = 0
        M_new = []
        for i in range(M.shape[0]):
            if M[i][j] != '':
                if M[i][j][0] == '>':
                    M[i][j] = M[i][j][1:]
                M_new.append(float(M[i][j]))
        max = np.amax(M_new)
        min = np.amin(M_new)
        denom = max - min
        for i in range(M.shape[0]):
            if M[i][j] != '':
                M_norm[i][j] = 2*(float(M[i][j])- min)/denom -1
            else:
                M_norm[i][j] = 0.0
    return M_norm

def preprocessing_nan(M):
    """
    Put nan values for the missing values and float of the str for the other values.
    Input:
        M: matrix that we want to preprocess.
    Output:
        M_norm: matrix that has the initial values for the non missing values and that has NAN values for the missing values.
    """
    M_norm = np.ndarray((M.shape))
    for j in range(M.shape[1]):
        for i in range(M.shape[0]):
            if M[i][j] != '':
                if M[i][j][0] == '>':
                    M[i][j] = M[i][j][1:]
                M_norm[i][j]=float(M[i][j])
            else:
                M_norm[i][j]= np.nan
    return M_norm

def preprocessing_nan_normalization(M):
    """
    Put nan values for the missing values and float of the str for the other values.
    Input:
        M: matrix that we want to normalize.
    Output:
        M_norm: matrix that is normalized and that has 0 values for the missing values.
    """
    M_norm = np.ndarray((M.shape))
    for j in range(M.shape[1]):
        max = 0
        min = 0
        M_new = []
        for i in range(M.shape[0]):
            if M[i][j] != '':
                if M[i][j][0] == '>':
                    M[i][j] = M[i][j][1:]
                M_new.append(float(M[i][j]))
        max = np.amax(M_new)
        min = np.amin(M_new)
        denom = max - min
        for i in range(M.shape[0]):
            if M[i][j] != '':
                M_norm[i][j] = 2*(float(M[i][j])- min)/denom -1
            else:
                M_norm[i][j] = np.nan
    return M_norm

def normalization_99(M):
    """
    normalization for TADPOLE dataset with removal of feature columns where there is more than 99% of data missing.
    Normalize a matrix between -1 and 1. Put 0 values to missing values of the matrix.
    Input:
        M: matrix that we want to normalize.
    Output:
        M_norm: matrix that is normalized and that has 0 values for the missing values.
    """
    M_norm = np.ndarray((M.shape[0], M.shape[1]-1))
    ind_to_delete=[]
    for j in range(M.shape[1]-1):
        max = 0
        min = 0
        M_new = []
        for i in range(M.shape[0]):
            if M[i][j] != '':
                if M[i][j][0] == '>':
                    M[i][j] = M[i][j][1:]
                M_new.append(float(M[i][j]))
        max = np.amax(M_new)
        min = np.amin(M_new)
        denom = max - min
        if denom==0:
            ind_to_delete.append(j)
        for i in range(M.shape[0]):
            if M[i][j] != '':
                M_norm[i][j] = 2*(float(M[i][j])- min)/denom -1
            else:
                M_norm[i][j] = 0.0

    ind_to_delete.sort(reverse = True)
    for i in range(len(ind_to_delete)):
        M_norm = np.hstack((M_norm[:,:ind_to_delete[i]], M_norm[:,ind_to_delete[i]+1:]))
    print(M_norm.shape)
    return M_norm

### the two next normalization function are done for the architectures based on GAE
def normalization_gae(M):
    """
    Normalize a matrix between -1 and 1. Put -10 values to missing values of the matrix.
    Input:
        M: matrix that we want to normalize.
    Output:
        M_norm: matrix that is normalized and that has -10 values for the missing values.
    """
    M_norm = np.ndarray((M.shape))
    for j in range(M.shape[1]):
        max = 0
        min = 0
        M_new = []
        for i in range(M.shape[0]):
            if M[i][j] != '':
                if M[i][j][0] == '>':
                    M[i][j] = M[i][j][1:]
                M_new.append(float(M[i][j]))
        max = np.amax(M_new)
        min = np.amin(M_new)
        denom = max - min
        for i in range(M.shape[0]):
            if M[i][j] != '':
                M_norm[i][j] = 2*(float(M[i][j])- min)/denom -1
            else:
                M_norm[i][j] = -10
    return M_norm

def normalization_for_supportreal(M):
    """
    Normalize a matrix between -1 and 1. Put 0 values to missing values of the matrix.
    Input:
        M: matrix that we want to normalize.
    Output:
        M_norm: matrix that is normalized and that has 0 values for the missing values.
    """
    M_norm = np.ndarray((M.shape))
    for j in range(M.shape[1]):
        max = 0
        min = 0
        M_new = []
        for i in range(M.shape[0]):
            if M[i][j] != '':
                if M[i][j][0] == '>':
                    M[i][j] = M[i][j][1:]
                M_new.append(float(M[i][j]))
        max = np.amax(M_new)
        min = np.amin(M_new)
        mean=np.mean(M_new)
        denom = max - min
        for i in range(M.shape[0]):
            if M[i][j] != '':
                M_norm[i][j] =  2*(float(M[i][j])- min)/denom -1
            else:
                M_norm[i][j] =0
    return M_norm

#functions for the initialization and the mask creation
def normalization_vect(M):
    """
    Normalize a matrix between 0 and 1.
    Input:
        M: matrix that we want to normalize.
    Output:
        M_norm: matrix that is normalized
    """
    M_norm = np.ndarray((M.shape))
    for j in range(M.shape[1]):
        max = 0
        min = 0
        M_new = []
        for i in range(M.shape[0]):
            M_new.append(float(M[i][j]))
        max = np.amax(M_new)
        min = np.amin(M_new)
        denom = max - min
        for i in range(M.shape[0]):
            M_norm[i][j] = (float(M[i][j])- min)/denom

    return M_norm

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix.
    Input:
        adj : adjacency matrix to symmetrically normalize
    Output:
        new adjacency matrix that is normalized
    """
    adj= np.array(adj)
    rowsum = np.sum(adj, axis=1)
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return np.dot(np.dot(d_mat_inv_sqrt, adj),d_mat_inv_sqrt)

def load_mask(mask, M, nrRows, nrCols):
    """
    function to create a mask on the data from the repartition of the dataset mask
    Put zero values when missing values
    input:
        mask : column (size m*1) with the number of examples, 1 if the example is taken in the mask and 0 if not
        M : matrix (size m*n) to do the mask on
    output:
        mask_final : mask on the matrix M following mask
    """
    mask_final = np.ndarray((nrRows, nrCols))
    for i in range(mask.shape[0]):
        if mask[i] == 1.0:
            for j in range(M.shape[1]):
                if M[i][j] != '':
                    mask_final[i][j] = 1.0
                else:
                    mask_final[i][j] = 0.0
        else:
            mask_final[i] = [0]*M.shape[1]
    return mask_final

def str_to_float(matrix):
    """
    function to put the element of matrix from str to float
    input:
        matrix: matrix of str that we want to put elements from str to float
    output:
        matrix: same matrix with float elements
    """
    matrix = matrix.astype(np.float32)
    return matrix

# split between train and test set. Can be changed by seed.
def split_train_test(percentage_train, matrix, seed, labels):
    """
    Random splitting of the matrix between the train and the test set.
    Put as many 0 as 1 in each set.
    inputs:
        percentage_train: percentage of the number of rows (subjects) used in the train set
        matrix: matrix to split
        seed: random seed for a random split
        labels: columns with the labels for the classification task
    outputs:
        training_set_mask, testing_set_mask: masks with one values if the rows are in the set, 0 if not
        idx_training, idx_testing: row indices of training and testing examples in the matrix
    """
    split_train_test = percentage_train
    training_set_mask = np.zeros((matrix.shape[0],1))
    testing_set_mask = np.zeros((matrix.shape[0],1))
    size_train_set = int(split_train_test*matrix.shape[0])
    size_test_set = matrix.shape[0]-size_train_set

    # random splitting of the subjects between train and test
    np.random.seed(seed)
    list_idx_ones = np.where(labels)[0]
    np.random.shuffle(list_idx_ones)
    idx_training_ones = list_idx_ones[:int(split_train_test*len(list_idx_ones))]
    idx_testing_ones = list_idx_ones[int(split_train_test*len(list_idx_ones)):]

    list_idx_zeros = np.where(labels==0.0)[0]
    np.random.shuffle(list_idx_zeros)
    idx_training_zeros = list_idx_zeros[:int(split_train_test*len(list_idx_zeros))]
    idx_testing_zeros = list_idx_zeros[int(split_train_test*len(list_idx_zeros)):]

    idx_training=np.concatenate((idx_training_ones,idx_training_zeros), axis=0)
    idx_testing=np.concatenate((idx_testing_ones,idx_testing_zeros), axis=0)
    training_set_mask[idx_training] = 1
    testing_set_mask[idx_testing] = 1
    return training_set_mask, testing_set_mask, idx_training, idx_testing

def split_train_validation_4(nb_val, matrix, seed, labels):
    """
    Random splitting of the matrix between the train and the validation set.
    We divide the training set into 4 equal parts and take one, indexed as nb_val for the validation set
    Put as many 0 as 1 in each set.
    inputs:
        nb_val: index of the validation set (0, 1,2 or 3)
        matrix: matrix to split
        seed: random seed for a random split
        labels: columns with the labels for the classification task
    outputs:
        training_set_mask, validation_set_mask: masks with one values if the rows are in the set, 0 if not
        idx_training, idx_validation: row indices of training and validation examples in the matrix
    """
    split_train_test = 0.25
    training_set_mask = np.zeros((matrix.shape[0],1))
    validation_set_mask = np.zeros((matrix.shape[0],1))

    # random splitting of the subjects between train and validation
    np.random.seed(seed)
    list_idx_ones = np.where(labels==1.0)[0]
    np.random.shuffle(list_idx_ones)
    idx_1_ones = list_idx_ones[:int(split_train_test*len(list_idx_ones))]
    idx_2_ones = list_idx_ones[int(split_train_test*len(list_idx_ones)):2*int(split_train_test*len(list_idx_ones))]
    idx_3_ones = list_idx_ones[2*int(split_train_test*len(list_idx_ones)):3*int(split_train_test*len(list_idx_ones))]
    idx_4_ones = list_idx_ones[3*int(split_train_test*len(list_idx_ones)):]

    list_idx_zeros = np.where(labels==0.0)[0]
    np.random.shuffle(list_idx_zeros)
    idx_1_zeros = list_idx_zeros[:int(split_train_test*len(list_idx_zeros))]
    idx_2_zeros = list_idx_zeros[int(split_train_test*len(list_idx_zeros)):2*int(split_train_test*len(list_idx_zeros))]
    idx_3_zeros = list_idx_zeros[2*int(split_train_test*len(list_idx_zeros)):3*int(split_train_test*len(list_idx_zeros))]
    idx_4_zeros = list_idx_zeros[3*int(split_train_test*len(list_idx_zeros)):]

    list_idx_zeros_c=[idx_1_zeros, idx_2_zeros, idx_3_zeros, idx_4_zeros]
    list_idx_ones_c= [idx_1_ones, idx_2_ones, idx_3_ones, idx_4_ones ]

    idx_validation=np.concatenate((list_idx_zeros_c[nb_val],list_idx_ones_c[nb_val]), axis=0)
    del list_idx_zeros_c[nb_val]
    del list_idx_ones_c[nb_val]
    idx_training= np.concatenate((list_idx_zeros_c[0],list_idx_ones_c[0]), axis=0)
    for i in range(1, len(list_idx_ones_c)):
        idx_training=np.concatenate((idx_training, list_idx_zeros_c[i],list_idx_ones_c[i]), axis=0)


    training_set_mask[idx_training] = 1
    validation_set_mask[idx_validation] = 1
    return training_set_mask, validation_set_mask, idx_training, idx_validation

def split_train_val_test(M, seed, labels, nb_val = 3, split=0.8):
    """
    Random splitting of the matrix M between the train, validation and testing sets.
    Put as many 0 as 1 in each set.
    Use function split_train_test and split_train_validation_4 to create the sets
    inputs:
        M: matrix to split
        nb_val: index of the validation set (0, 1,2 or 3)
        seed: random seed for a random split
        labels: columns with the labels for the classification task
    outputs:
        M_train_val, M_val, M_test: matrix for each set
        labels_train_val, labels_val, labels_test: label for each set
    """
    training_set_mask, testing_set_mask, idx_train, idx_test=split_train_test(split, M, seed, labels)
    M_train = np.delete(M, idx_test,0)
    M_test = np.delete(M, idx_train,0)
    labels_train = np.delete(labels, idx_test,0)
    labels_test = np.delete(labels, idx_train,0)

    new_labels_train=np.copy(labels)
    new_labels_train[idx_test]=-1
    #split train set into 4 parts to create a validation set
    training_set_mask, validation_set_mask, idx_training, idx_validation=split_train_validation_4(3, M, seed, new_labels_train)
    M_train_val = np.delete(M_train, idx_validation,0)
    M_val = np.delete(M_train, idx_training,0)
    labels_train_val = np.delete(labels_train, idx_validation,0)
    labels_val= np.delete(labels_train, idx_training,0)
    return M_train_val, M_val, M_test, labels_train_val, labels_val, labels_test

def split_train_into_2(mask, seed):
    """
    random splitting of the 1 of the mask into two masks with the same number of 1

    inputs:
        mask : mask with 1 and 0
        seed: random seed
    outputs:
        Odata, Otraining: two masks with the same number of 1, Odata+Otraining=mask
    """

    np.random.seed(seed)
    pos_tr_samples = np.where(mask)
    num_tr_samples = len(pos_tr_samples[0])
    list_idx = range(num_tr_samples)
    np.random.shuffle(list_idx)
    idx_data = list_idx[:num_tr_samples//2]
    idx_train = list_idx[num_tr_samples//2:]

    pos_data_samples = (pos_tr_samples[0][idx_data], pos_tr_samples[1][idx_data])
    pos_tr_samples = (pos_tr_samples[0][idx_train], pos_tr_samples[1][idx_train])

    Odata = np.zeros(mask.shape)
    Otraining = np.zeros(mask.shape)

    for k in range(len(pos_data_samples[0])):
        Odata[pos_data_samples[0][k], pos_data_samples[1][k]] = 1

    for k in range(len(pos_tr_samples[0])):
        Otraining[pos_tr_samples[0][k], pos_tr_samples[1][k]] = 1

    return Odata, Otraining

def init_mask(mask, M, labels):
    """
    mask to initialize the matrix with 0.5 values for the labels that are not in
    the mask for initialization.
    Keep all the other values for the features to the values of the mask.
    inputs:
        mask : mask. We want to only change the labels column of this mask.
        Size nb subj * (nb features+1)
        M: feature matrix, size nb subjects * nb features
        Labels: column with the labels
    output:
        M_init_final : application of the mask to the M matrix and concatenation
        with the new labels column
    """
    Odata_nolab = np.zeros(M.shape)
    Odata_lab = np.zeros(M.shape[0])
    for j in range(mask.shape[0]):
        if mask[j][-1] == 0.0:
            Odata_lab[j] = 0.5
        else:
            Odata_lab[j] = labels[j]
        Odata_nolab[j]  = mask[j][:-1]

    M_init_final =np.concatenate((Odata_nolab*M, Odata_lab[:, np.newaxis]), axis=1)
    return M_init_final


def svd_W_H(M, rank=156):
    """
    Apply SVD to matrix M. New matrices W and H have a rank rank
    inputs:
        M: matrix to apply SVD to
        rank: rank chosen for the decomposition
    outputs:
        W, H: two matrices created by SVD
    """
    U, s, V = np.linalg.svd(M, full_matrices=0)

    rank_W_H = rank
    partial_s = s[:rank_W_H]
    partial_S_sqrt = np.diag(np.sqrt(partial_s))
    W = np.dot(U[:, :rank_W_H], partial_S_sqrt)
    H = np.dot(partial_S_sqrt, V[:rank_W_H, :]).T
    return W, H

def accuracy_computation(pred, mask, labels):
    """
    compute accuracy, tn, tp, fp, fn on a dataset that has a mask
    inputs:
        pred: predictions
        mask: mask on the predictions
        labels: true labels to compare pred to
    outputs:
        acc : accuracy
        tn : true negative
        fn : false negative
        tp : true Positive
        fp : false positive
        iter : number of examples in the mask
    """
    acc = 0
    iter = 0
    tn =0
    tp = 0
    fp=0
    fn=0
    for i in range(len(pred)):
        if mask[i] == 1.0:
            iter +=1
            if labels[i] == 0.0:
                if pred[i] < 0.5:
                    acc +=1
                    tn +=1
                else:
                    fn +=1
            else:
                if pred[i] >= 0.5:
                    acc +=1
                    tp +=1
                else:
                    fp +=1
    if tp+fn !=0:
        spe = float(tp)/(tp+fn)
    else:
        spe = 0
    if tn +fp != 0:
        sen=float(tn)/(tn+fp)
    else:
        sen = 0
    accuracy=  float(acc)/iter
    return accuracy, tn, fn, tp, fp, spe, sen

def accuracy_computation_nomask(pred, labels):
    """
    compute accuracy, tn, tp, fp, fn on a dataset that has a mask
    inputs:
        pred: predictions
        mask: mask on the predictions
        labels: true labels to compare pred to
    outputs:
        acc : accuracy
        tn : true negative
        fn : false negative
        tp : true Positive
        fp : false positive
        iter : number of examples in the mask
    """
    acc = 0
    iter = 0
    tn =0
    tp = 0
    fp=0
    fn=0
    for i in range(len(pred)):
        iter +=1
        if labels[i] == 0.0:
            if pred[i] < 0.5:
                acc +=1
                tn +=1
            else:
                fn +=1
        else:
            if pred[i] >= 0.5:
                acc +=1
                tp +=1
            else:
                fp +=1
    if tp+fn !=0:
        spe = float(tp)/(tp+fn)
    else:
        spe = 0
    if tn +fp != 0:
        sen=float(tn)/(tn+fp)
    else:
        sen = 0
    accuracy=  float(acc)/iter
    return accuracy, tn, fn, tp, fp, spe, sen

def mean_over_runs_seed(listrun):
    """
    mean of the values for one seed for each iteration
    One value per iteration. Works for the AUC for example
    input:
        - listrun : np array of size nb of seed * nb times repeat one process * nb
        of iterations of one process
    output:
        - final : np array of size nb of seed * nb of iterations of one process
        mean over the process for one seed.
        - std_array: np array of size nb of seed * nb of iterations of one process
        std over the process for one seed.

    """
    final = []
    std_array=[]
    for i in range(0,len(listrun)):
        mean = np.mean(listrun[i], axis=0)
        std=np.std(listrun[i], axis=0)
        final.append(mean)
        std_array.append(std)

    return final, std_array

def pred_to_auc_roc(labels_train, labels_test, pred_train, pred_test, nb_seed):
    """
    for each seed, for each iteration of the algorithm, compute the AUC value, the fpr and tpr on the train and test set
    Inputs:
        - labels_train: train labels for each seed, shape nb examples * nb seeds
        - labels_test: test labels for each seed, shape nb examples * nb seeds
        - pred_train: output of the algorithm for each iteration and each seed on the train set, shape nb examples * nb iterations * nb seeds
        - pred_test: output of the algorithm for each iteration and each seed on the test set, shape nb examples * nb iterations * nb seeds
        - nb_seed: number of different initializations
    """
    fpr_train_pred_iter=[]
    tpr_train_pred_iter=[]
    fpr_test_pred_iter=[]
    tpr_test_pred_iter=[]
    auc_train_pred_iter=[]
    auc_test_pred_iter=[]
    for i in range(nb_seed):
        fpr_train_pred=[]
        tpr_train_pred=[]
        fpr_test_pred=[]
        tpr_test_pred=[]
        auc_train_pred=[]
        auc_test_pred=[]
        for j in range(len(pred_train[0])):
            fpr_train_1, tpr_train_1, thresholds=roc_curve(labels_train[i], pred_train[i][j])
            roc_auc_train = auc(fpr_train_1, tpr_train_1)
            fpr, tpr, thresholds=roc_curve(labels_test[i], pred_test[i][j])
            roc_auc = auc(fpr, tpr)
            fpr_train_pred.append(fpr_train_1)
            tpr_train_pred.append(tpr_train_1)
            fpr_test_pred.append(fpr)
            tpr_test_pred.append(tpr)
            auc_train_pred.append(roc_auc_train)
            auc_test_pred.append(roc_auc)
        auc_test_pred_iter.append(auc_test_pred)
        auc_train_pred_iter.append(auc_train_pred)
        fpr_train_pred_iter.append(fpr_train_pred)
        tpr_train_pred_iter.append(tpr_train_pred)
        fpr_test_pred_iter.append(fpr_test_pred)
        tpr_test_pred_iter.append(tpr_test_pred)
    return auc_test_pred_iter,  auc_train_pred_iter, fpr_train_pred_iter, tpr_train_pred_iter, fpr_test_pred_iter, tpr_test_pred_iter

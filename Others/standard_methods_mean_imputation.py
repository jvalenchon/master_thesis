###############################
##
## code for random forest , linear SVM, linear ridge and MLP with mean imputation
## dataset with missing values. Can also work for dataset with no missing value
##
##############################
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import RidgeClassifier

import preprocessing_dataset
import read_tadpole

###############################
##
## files required to run the program. Can do it on the tadpole dataset and on the synthetic datasets
##
##############################
path_dataset_matrix ='bl_data_MCI_v1.csv'
labels_path = 'labels_comma.csv'
"""
path_dataset_matrix ="synthetic_data_noteasy.csv"
labels_path = "labels_synthetic_data_noteasy.csv"
"""
#load the .csv files
M_str, nrRows, nrCols = read_tadpole.load_csv_no_header(path_dataset_matrix)
labels, _, _ = read_tadpole.load_csv_no_header(labels_path)
labels = preprocessing_dataset.str_to_float(labels)


###############################
##
## RANDOM FOREST
##
##############################

def random_forest(M, labels, seed, split=0.8, n_estimators=80):
    """
    Random forest algorithm for input M and output labels
    Inputs:
        M : matrix m*n where each row is a different example and the columns are composed of the features
        labels : vector m*1 where each row is the correponding class of the row of M
        seed : random seed to do the split between test/validation/training
        split: number between 0 and 1. Split between training and testing set. Default : 0.8
        n_estimators: number of estimators to use for random forest. Default: 80
    Ouputs:
        roc_auc_rf_train: AUC score on the train set
        roc_auc_rf_val: AUC score on the validation set
        roc_auc_rf: AUC score on the test set
    """
    M_float = preprocessing_dataset.preprocessing_nan(M)

    ### Initialization of the training/validation/testing sets.
    M_train_val, M_val, M_test, labels_train_val, labels_val, labels_test=preprocessing_dataset.split_train_val_test(M_float, seed, labels, nb_val = 3, split=0.8)
    X_train = M_train_val
    Y_train = labels_train_val
    Y_train=np.reshape(Y_train, (Y_train.shape[0],))

    X_test = M_test
    # Create our imputer to replace missing values with the mean e.g.
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp = imp.fit(X_train)

    # Impute our data, then train
    X_train_imp = imp.transform(X_train)
    clf = RandomForestClassifier(n_estimators=n_estimators)
    clf = clf.fit(X_train_imp, Y_train)


    # Impute each test item, then predict
    X_val_imp = imp.transform(M_val)
    X_test_imp = imp.transform(X_test)
    y_pred_train= clf.predict_proba(X_train_imp)[:,1]
    y_pred_proba = clf.predict_proba(X_test_imp)[:,1]
    y_pred_val = clf.predict_proba(X_val_imp)[:,1]

    fpr_rf, tpr_rf, thresholds_train=roc_curve(Y_train, y_pred_train)
    roc_auc_rf_train= auc(fpr_rf, tpr_rf)

    fpr_rf, tpr_rf, thresholds_train=roc_curve(labels_val, y_pred_val)
    roc_auc_rf_val= auc(fpr_rf, tpr_rf)

    fpr_rf, tpr_rf, thresholds_train=roc_curve(labels_test, y_pred_proba)
    roc_auc_rf= auc(fpr_rf, tpr_rf)
    print('random forest: train set: %0.5f, val set: %0.5f, test set: %0.5f'%(roc_auc_rf_train, roc_auc_rf_val, roc_auc_rf))
    return  roc_auc_rf_train, roc_auc_rf_val, roc_auc_rf





###############################
##
## linear SVM
##
##############################



def svm(M, labels, seed, split=0.8):
    """
    linear SVM algorithm for input M and output labels
    Inputs:
        M : matrix m*n where each row is a different example and the columns are composed of the features
        labels : vector m*1 where each row is the correponding class of the row of M
        seed : random seed to do the split between test/validation/training
        split: number between 0 and 1. Split between training and testing set. Default : 0.8
    Ouputs:
        roc_auc_svm_train: AUC score on the train set
        roc_auc_svm_val: AUC score on the validation set
        roc_auc_svm: AUC score on the test set
    """
    M_float = preprocessing_dataset.preprocessing_nan_normalization(M_str)
    M_train_val, M_val, M_test, labels_train_val, labels_val, labels_test=preprocessing_dataset.split_train_val_test(M_float, seed, labels, nb_val = 3, split=0.8)
    X_train = M_train_val
    Y_train = labels_train_val
    X_test = M_test
    Y_train=np.reshape(Y_train, (Y_train.shape[0],))
    # Create our imputer to replace missing values with the mean e.g.
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp = imp.fit(X_train)

    # Impute our data, then train
    X_train_imp = imp.transform(X_train)
    clf = SVC(kernel='linear',probability=True)
    clf = clf.fit(X_train_imp, Y_train)

    # Impute each test item, then predict
    X_test_imp = imp.transform(X_test)
    X_val_imp = imp.transform(M_val)

    y_pred = clf.predict_proba(X_train_imp)[:,1]
    y_pred = np.reshape(y_pred, (y_pred.shape[0],1))
    fpr_svm, tpr_svm, thresholds_train=roc_curve(Y_train, y_pred)
    roc_auc_svm_train = auc(fpr_svm, tpr_svm)

    y_pred_val = clf.predict_proba(X_val_imp)[:,1]
    y_pred_val = np.reshape(y_pred_val, (y_pred_val.shape[0],1))
    fpr_svm, tpr_svm, thresholds_train=roc_curve(labels_val, y_pred_val)
    roc_auc_svm_val = auc(fpr_svm, tpr_svm)

    y_pred_proba = clf.predict_proba(X_test_imp)[:,1]
    y_pred_proba = np.reshape(y_pred_proba, (y_pred_proba.shape[0],1))

    fpr_svm, tpr_svm, thresholds_train=roc_curve(labels_test, y_pred_proba)
    roc_auc_svm = auc(fpr_svm, tpr_svm)
    print('linear SVM: train set: %0.5f, val set: %0.5f, test set: %0.5f'%(roc_auc_svm_train, roc_auc_svm_val, roc_auc_svm))
    return roc_auc_svm_train, roc_auc_svm_val, roc_auc_svm


###############################
##
## linear_ridge
##
##############################

def linear_ridge(M, labels, seed, split=0.8):
    """
    linear ridge algorithm for input M and output labels
    Inputs:
        M : matrix m*n where each row is a different example and the columns are composed of the features
        labels : vector m*1 where each row is the correponding class of the row of M
        seed : random seed to do the split between test/validation/training
        split: number between 0 and 1. Split between training and testing set. Default : 0.8
    Ouputs:
        roc_auc_train: AUC score on the train set
        roc_auc_val: AUC score on the validation set
        roc_auc_test: AUC score on the test set
    """
    M_float = preprocessing_dataset.preprocessing_nan_normalization(M_str)
    M_train_val, M_val, M_test, labels_train_val, labels_val, labels_test=preprocessing_dataset.split_train_val_test(M_float, seed, labels, nb_val = 3, split=0.8)
    X_train = M_train_val
    Y_train = labels_train_val
    X_test = M_test
    Y_train=np.reshape(Y_train, (Y_train.shape[0],))
    # Create our imputer to replace missing values with the mean e.g.
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp = imp.fit(X_train)

    # Impute our data, then train
    X_train_imp = imp.transform(X_train)
    clf = RidgeClassifier()
    clf=clf.fit(X_train_imp, Y_train)

    # Impute each test item, then predict
    X_test_imp = imp.transform(X_test)
    X_val_imp = imp.transform(M_val)

    # Compute the accuracy
    lin_acc = clf.score(X_test_imp, labels_test)
    # Compute the AUC
    pred_train = clf.decision_function(X_train_imp)
    pred = clf.decision_function(X_test_imp)
    pred_val = clf.decision_function(X_val_imp)

    fpr_svm, tpr_svm, thresholds_train=roc_curve(Y_train, pred_train)
    roc_auc_train = auc(fpr_svm, tpr_svm)
    fpr_svm, tpr_svm, thresholds_train=roc_curve(labels_val, pred_val)
    roc_auc_val = auc(fpr_svm, tpr_svm)
    fpr_svm, tpr_svm, thresholds_train=roc_curve(labels_test, pred)
    roc_auc_test = auc(fpr_svm, tpr_svm)
    print('linear ridge: train set: %0.5f, validation: %0.5f, test set: %0.5f'%(roc_auc_train, roc_auc_val, roc_auc_test))
    return roc_auc_train, roc_auc_val, roc_auc_test


###############################
##
## Multi-layer perceptron
##
##############################

def MLP(M, labels, seed, split=0.8, hidden_layer_size_nb=50):
    """
    linear ridge algorithm for input M and output labels
    Inputs:
        M : matrix m*n where each row is a different example and the columns are composed of the features
        labels : vector m*1 where each row is the correponding class of the row of M
        seed : random seed to do the split between test/validation/training
        split: number between 0 and 1. Split between training and testing set. Default : 0.8
    Ouputs:
        roc_auc_rf_train: AUC score on the train set
        roc_auc_rf_val: AUC score on the validation set
        roc_auc_rf: AUC score on the test set
    """
    M_float = preprocessing_dataset.preprocessing_nan_normalization(M_str)
    M_train_val, M_val, M_test, labels_train_val, labels_val, labels_test=preprocessing_dataset.split_train_val_test(M_float, seed, labels, nb_val = 3, split=0.8)

    X_train = M_train_val
    Y_train = labels_train_val
    X_test = M_test
    Y_train=np.reshape(Y_train, (Y_train.shape[0],))
    # Create our imputer to replace missing values with the mean e.g.
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp = imp.fit(X_train)

    # Impute our data, then train
    X_train_imp = imp.transform(X_train)
    # Impute each test item, then predict
    X_test_imp = imp.transform(X_test)
    X_val_imp=imp.transform(M_val)

    clf = MLPClassifier(solver='adam', alpha=1e-5,
                 hidden_layer_sizes=(hidden_layer_size_nb), random_state=1)
    clf = clf.fit(X_train_imp, Y_train)

    y_pred_proba_train = clf.predict_proba(X_train_imp)[:,1]
    y_pred_proba_train = np.reshape(y_pred_proba_train, (y_pred_proba_train.shape[0],1))

    fpr_svm, tpr_svm, thresholds_train=roc_curve(Y_train, y_pred_proba_train)
    roc_auc_train = auc(fpr_svm, tpr_svm)

    y_pred_proba_val = clf.predict_proba(X_val_imp)[:,1]
    y_pred_proba_val = np.reshape(y_pred_proba_val, (y_pred_proba_val.shape[0],1))

    fpr_svm, tpr_svm, thresholds_train=roc_curve(labels_val, y_pred_proba_val)
    roc_auc_val = auc(fpr_svm, tpr_svm)
    y_pred_proba = clf.predict_proba(X_test_imp)[:,1]
    y_pred_proba = np.reshape(y_pred_proba, (y_pred_proba.shape[0],1))

    fpr_svm, tpr_svm, thresholds_train=roc_curve(labels_test, y_pred_proba)
    roc_auc_test = auc(fpr_svm, tpr_svm)
    print('MLP: train set: %0.5f, validation: %0.5f, test set: %0.5f'%(roc_auc_train, roc_auc_val, roc_auc_test))
    return roc_auc_train, roc_auc_val, roc_auc_test


if __name__=="__main__":
    res = []
    svm_res = []
    ridge= []
    mlp_res=[]
    for i in range(5):#choose number of different initialization you want
        res.append(random_forest(M_str, labels, i, 0.8, 80)[-1])
        svm_res.append(svm(M_str, labels, i, 0.8)[-1])
        ridge.append(linear_ridge(M_str, labels, i, 0.8)[-1])
        mlp_res.append(MLP(M_str, labels, i, 0.8, 50)[-1])

    mean = np.mean(res)
    std=np.std(res)
    print('mean result for Random Forest: %0.5f +- %0.5f' %(mean, std))
    #np.savetxt('auc_test_randomforest_tadpole.csv', res, fmt='%s', delimiter=',')
    mean = np.mean(svm_res)
    std=np.std(svm_res)
    print('mean result for linear SVM: %0.5f +- %0.5f' %(mean, std))
    #np.savetxt('auc_test_svm_synthnoteasy.csv', svm_res, fmt='%s', delimiter=',')
    mean = np.mean(ridge)
    std=np.std(ridge)
    print('mean result for linear ridge: %0.5f +- %0.5f' %(mean, std))
    mean = np.mean(mlp_res)
    std=np.std(mlp_res)
    print('mean result for mlp: %0.5f +- %0.5f' %(mean, std))

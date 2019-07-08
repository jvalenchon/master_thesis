# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import seaborn
import csv
import matplotlib
matplotlib.use('agg')
### size and font of the writtings in the plots
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 15}

matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def load_csv_no_header(filePath):
    """
    Load data of csv file into a matrix. The csv file is only composed of data, no header.
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

path_dataset_matrix ="gcn_u_train_emb128_synthnoteasy_nonorm.csv"#"gcn_u_train_emb217_tadpole_nonorm.csv"
M_str, nrRows, nrCols = load_csv_no_header(path_dataset_matrix)
gender_file_path = "gender_column_train_synthnoteasy.csv"#"gender_column_train_tadpole217.csv"
gender_column, nrRowsg, nrColsg = load_csv_no_header(gender_file_path)
gender_column=gender_column.reshape(-1)

age_file_path ="age_column_train_synthnoteasy.csv"#"age_column_train_tadpole217.csv"
age_column, nrRowsa, nrColsa = load_csv_no_header(age_file_path)
age_column=age_column.reshape(-1)

path_labels="labels_column_train_synthnoteasy.csv"#"labels_column_train_tadpole217.csv"
labels, _, _ = load_csv_no_header(path_labels)
labels= labels.reshape(-1)

hidden_a=61 #99
X = np.concatenate((age_column[:, np.newaxis], gender_column[:, np.newaxis]), axis=1) #covariates

age_max_int = 92
age_min_int = 84

for i in range(M_str.shape[1]//hidden_a):
    # i different support matrix

    #first step : construct the PCA decomposition
    y=[]
    print(hidden_a*i)
    print(hidden_a*i + hidden_a)
    for j in range(M_str.shape[0]):
        y.append(M_str[j][i*hidden_a:hidden_a*i+hidden_a])
    y =np.array(y)
    print(y.shape)
    pca = PCA(n_components=2)
    X_new = pca.fit_transform(y)

    #split between subjects that are MCIc and MCInc
    labels_zeros= np.where(labels==0)
    labels_ones= np.where(labels==1)

    plt.figure()
    plt.title('PCA support %i' %i)

    if (i-2)%3==0 and i!=2:
        #After 3 support, except for the 2 first one, the age range changes
        age_max_int = age_min_int
        age_min_int = age_min_int-5

    if i == 0:
        #Women support
        zero_class = np.where(gender_column==1)
        zero_class_label0=np.intersect1d(zero_class, labels_zeros)
        zero_class_label1=np.intersect1d(zero_class, labels_ones)
        one_class = np.where(gender_column==0)
    elif i==1:
        #Men support
        zero_class = np.where(gender_column==0)
        zero_class_label0=np.intersect1d(zero_class, labels_zeros)
        zero_class_label1=np.intersect1d(zero_class, labels_ones)
        one_class = np.where(gender_column==1)
    elif (i-2)%3==0:
        #Women and age
        gender = np.where(gender_column==1)
        age_min = np.where(age_column>=age_min_int)
        age_max = np.where(age_column<=age_max_int)
        age_f = np.intersect1d(age_min, age_max)
        zero_class = np.intersect1d(gender, age_f)
        zero_class_label0=np.intersect1d(zero_class, labels_zeros)
        zero_class_label1=np.intersect1d(zero_class, labels_ones)
        new_ind = np.arange(len(gender_column))
        one_class=np.delete(new_ind, zero_class)
        print(len(zero_class))
    elif (i%3)==0:
        #Men and age
        gender = np.where(gender_column==0)
        age_min = np.where(age_column>=age_min_int)
        age_max = np.where(age_column<=age_max_int)
        age_f = np.intersect1d(age_min, age_max)
        zero_class = np.intersect1d(gender, age_f)
        zero_class_label0=np.intersect1d(zero_class, labels_zeros)
        zero_class_label1=np.intersect1d(zero_class, labels_ones)
        new_ind = np.arange(len(gender_column))
        one_class=np.delete(new_ind, zero_class)
    elif (i-4)%3==0:
        # age
        age_min = np.where(age_column>=age_min_int)
        age_max = np.where(age_column<=age_max_int)
        zero_class = np.intersect1d(age_min, age_max)
        zero_class_label0=np.intersect1d(zero_class, labels_zeros)
        zero_class_label1=np.intersect1d(zero_class, labels_ones)
        new_ind = np.arange(len(gender_column))
        one_class=np.delete(new_ind, zero_class)

    plt.scatter(X_new[zero_class_label0, 0], X_new[zero_class_label0, 1],s=160, edgecolors='b',
               facecolors='none', linewidths=2, label='Support 1 label 0')
    plt.scatter(X_new[zero_class_label1, 0], X_new[zero_class_label1, 1],s=160, edgecolors='r',
               facecolors='none', linewidths=2, label='Support 1 label 1')
    plt.scatter(X_new[one_class, 0], X_new[one_class, 1], s=80, edgecolors='orange',
               facecolors='none', linewidths=2, label='Support 0')
    plt.legend()
    plt.savefig('embeddings_tadpole_test/pca_embedding%i_features_labels.pdf' %i )
    plt.show()

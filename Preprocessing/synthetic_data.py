###########################################"
##
## CREATION OF SYNTHETIC DATASETS
##
#########################################

import numpy as np
import preprocessing_dataset
from sklearn.metrics import roc_curve, auc
import read_tadpole

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


def creation_dataset(m, n, nb_features_generated):
    """
    function to create a synthetic dataset of shape m * n, n features being chosen in the nb_features_generated ones created
    The parameters to create the dataset can be changed, these are the one used to create our synthetic dataset
    Save the dataset created in csv files
    Inputs:
        m: number of subjects
        n : number of features
        nb_features_generated : number of features generated
    """
    M = np.zeros((m,nb_features_generated))
    sex = []
    label = []
    age = []

    for i in range(m):
        sex.append(np.random.randint(2))
        label.append(np.random.randint(2))
        age.append(np.random.uniform(64, 92))

    sex = np.array(sex)
    age = np.array(age)
    agesex = sex * age

    attributes = np.concatenate((age[:, np.newaxis], sex[:, np.newaxis], agesex[:, np.newaxis]), axis = 1)

    feat_dependence=[]
    for j in range(nb_features_generated):
        feat_type = np.random.randint(3)
        feat_dependence.append(feat_type)
        slope = np.random.uniform(-0.1,0.1)
        intercept = np.random.uniform(0,1)
        for i in range(m):
            error = np.random.normal(0,5)
            M[i][j] = slope * attributes[i][feat_type] + intercept + error

    np.random.seed(0)
    list_idx = range(nb_features_generated)
    np.random.shuffle(list_idx)
    idx_delete = list_idx[n:]

    M= np.delete(M, idx_delete,1)
    feat_dependence= np.delete(feat_dependence, idx_delete,0)

    for j in range(n):
        v=np.random.uniform(0.25)
        for i in range(m):
            M[i][j] += v  * label[i]

    np.savetxt('synthetic_data.csv', M, fmt='%f', delimiter=',')
    np.savetxt('labels_synthetic_data.csv', label, fmt='%f', delimiter=',')
    np.savetxt('ages_synthetic_data.csv', age, fmt='%f', delimiter=',')
    np.savetxt('sexs_synthetic_data.csv', sex, fmt='%f', delimiter=',')
    np.savetxt('feat_dependence_synthetic_data.csv', feat_dependence, fmt='%f', delimiter=',')


m = 779
n = 563
nb_features_generated = 1000
creation_dataset(m, n, nb_features_generated)

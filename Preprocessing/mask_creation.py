# Juliette Valenchon 04/2019

import read_tadpole
import numpy as np
import preprocessing_dataset


path_dataset_matrix = 'stat_results_v1_notext.csv' #matrix to complete
stat, nrRows, nrCols = read_tadpole.load_csv_no_header(path_dataset_matrix)
res_stat = stat.astype(np.float64)

path_dataset_matrix = 'bl_data_MCI_v1.csv'
M_str, nrRows, nrCols = read_tadpole.load_csv_no_header(path_dataset_matrix)


M_init = preprocessing_dataset.normalization(M_str)

def remove_column(column_remove, M):
    """
    Function to remove the column of M which are indexed by an indice in column_remove
    Input:
       column_remove: array with indices of the column to remove
       M: matrix with we want to remove the columns that are indexed by an indice in column_remove
    Output:
       M: matrix where we removed the columns
    """
    column_remove.sort(reverse = True)
    for i in range(len(column_remove)):
        M = np.hstack((M[:,:column_remove[i]], M[:,column_remove[i]+1:]))
    return M

def feat_dep(M):
    """
    If the p-value corresponding to one feature for an attribute (age, sex, age&sex) is inferior to 0.05, the feature is dependent on the attribute and we keep it
    Input:
       M : matrix with the p-values. Each column corresponds to an attribute : age, sex, age&sex. Each row corresponds to a feature
    Output:
       to_keep : matrix with the indices (corresponding to features) to keep for each attribute
    """
    to_keep=[] #contains indices to keep
    for j in range(len(M[0])):
        indices_to_keep = []
        for i in range(len(M)):
            if M[i][j] <= 0.05:
                indices_to_keep.append(i)
        to_keep.append(indices_to_keep)
    return to_keep

def missing_values_array(array):
    """
    Return the indices that are missing in array
    Input :
        array: array of indices
    Output:
        missing_element: array with the indices that were missing in array
    """
    missing_element = []
    if min(array)!=0:
        for k in range(min(array)):
            missing_element.append(k)
    for i in range(array[0], array[-1]+1):
        if i not in array:
            missing_element.append(i)
    if max(array) != 562:
        for j in range(max(array)+1, 563):
            missing_element.append(j)
    return missing_element

def load_mask_col(indices, M):
    """
    function to create a mask on the data from the repartition of the dataset mask
    Put zero values when missing value
    Inputs:
        indices : array with indices of columns to remove. We put 1 in mask if the column is taken and 0 if not
        M : matrix to do the mask on
    Output:
        mask_final : mask where the columns are filled with one if the indice of the column is not in indices
    """
    mask_final = np.ones((M.shape[0],M.shape[1]+1))
    for i in range(len(indices)):
        for j in range(M.shape[0]):
            mask_final[j][indices[i]] = 0.0
    return mask_final


to_keep=feat_dep(res_stat)
no_significance=np.concatenate((to_keep[0], to_keep[1], to_keep[2]), axis=0)
no_significance = list(set(no_significance))

missing_element_age=missing_values_array(to_keep[0])
missing_element_sex=missing_values_array(to_keep[1])
missing_element_agesex=missing_values_array(to_keep[2])

mask_age = load_mask_col(missing_element_age, M_init)
mask_sex = load_mask_col(missing_element_sex, M_init)
mask_agesex = load_mask_col(missing_element_agesex, M_init)
mask_nosignificance= load_mask_col(no_significance, M_init)

np.savetxt('mask_age.csv', mask_age, fmt='%f', delimiter=',')
np.savetxt('mask_sex.csv', mask_sex, fmt='%f', delimiter=',')
np.savetxt('mask_agesex.csv', mask_agesex, fmt='%f', delimiter=',')
np.savetxt('mask_nosignificance.csv', mask_nosignificance, fmt='%f', delimiter=',')

print('number of age-related features : %i, sex-related: %i and age&sex related: %i' %(len(to_keep[0]), len(to_keep[1]), len(to_keep[2])))
print('number of non significant features', len(no_significance))

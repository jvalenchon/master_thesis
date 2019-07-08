### creation of the support for the GC-MC architecture for the TADPOLE dataset

import numpy as np
import os
import csv
import read_tadpole
import preprocessing_dataset

def support_binary(mask, attribute, value):
    """
        mask       : mask on the column of the feature matrix that only keep the features that are related to value
        attribute : binary attribute (sex)
        value : which value to keep in attribute

    return:
        graph        : mask taking into account value
    """

    graph = np.zeros(mask.shape)
    attribute_mat = np.zeros(mask.shape)

    if value == 1:
        for i in range(mask.shape[1]):
            for j in range(mask.shape[0]):
                attribute_mat[j][i]=attribute[j]
    else:
        attribute = 1- attribute
        for i in range(mask.shape[1]):
            for j in range(mask.shape[0]):
                attribute_mat[j][i]=attribute[j]
    graph = attribute_mat*mask
    return graph


def support_continuous(mask, attribute, min, max):
    """
        mask       : mask on the column of the feature matrix that only keep the features that are related to value
        attribute : continuous attribute (age)
        min : value min taken by the attribute that is included
        max : value max taken by the attribute that is included
    return:
        graph        : mask taking into account value
    """

    graph = np.zeros(mask.shape)
    attribute_mat = np.zeros(mask.shape)
    age_in=[]
    for i in range(len(attribute)):
        if attribute[i] >=min and attribute[i]<=max:
            age_in.append(1)
        else:
            age_in.append(0)

    for i in range(mask.shape[1]):
        for j in range(mask.shape[0]):
            attribute_mat[j][i]=age_in[j]
    graph = attribute_mat*mask
    return graph

def support_continuous_binary(mask, attribute_bin, attribute_cont, val, min, max):
    """
        mask       : mask on the column of the feature matrix that only keep the features that are related to value
        attribute_bin : binary attribute
        attribute_cont : continuous attribute
        val : value taken by the binary attribute
        min : value min taken by the continuous attribute that is included
        max : value max taken by the continuous attribute that is included
    return:
        graph        : mask taking into account value
    """

    graph = np.zeros(mask.shape)
    attribute_mat = np.zeros(mask.shape)
    age_sex=[]
    for i in range(len(attribute_cont)):
        if attribute_cont[i] >=min and attribute_cont[i]<=max:
            if attribute_bin[i] == val:
                age_sex.append(1)
            else:
                age_sex.append(0)
        else:
            age_sex.append(0)

    for i in range(mask.shape[1]):
        for j in range(mask.shape[0]):
            attribute_mat[j][i]=age_sex[j]
    graph = attribute_mat*mask
    return graph


def support_binary_synth(feat_dep, attribute, value):
    """
        feat_dep       : array with feature dependencies (0: age related , 1: sex related ,2: age and sex related)
        attribute : array with values of binary attribute (sex)
        value : which value to keep in attribute

    return:
        attribute_mat        : mask taking into account value
    """
    print(len(feat_dep)+1, len(attribute))
    attribute_mat = np.zeros((len(attribute), len(feat_dep)+1))

    if value == 1:
        for i in range(len(feat_dep)):
            if feat_dep[i] == 1:
                for j in range(len(attribute)):
                    attribute_mat[j][i]=attribute[j]
            else:
                for j in range(len(attribute)):
                    attribute_mat[j][i]=0
    else:
        attribute = 1- attribute
        for i in range(len(feat_dep)):
            if feat_dep[i] == 1:
                for j in range(len(attribute)):
                    attribute_mat[j][i]=attribute[j]
            else:
                for j in range(len(attribute)):
                    attribute_mat[j][i]=0
    for j in range(len(attribute)):
        attribute_mat[j][-1]=attribute[j]

    return attribute_mat


def support_continuous_synth(feat_dep, attribute, min, max):
    """
        feat_dep       : array with feature dependencies (0: age related , 1: sex related ,2: age and sex related)
        attribute : array with values taken by the attribute
        min : value min taken by the continuous attribute that is included
        max : value max taken by the continuous attribute that is included

    return:
        attribute_mat        : mask taking into account value
    """

    print(len(feat_dep)+1, len(attribute))
    attribute_mat = np.zeros((len(attribute), len(feat_dep)+1))

    age_in=[]
    for i in range(len(attribute)):
        if attribute[i] >=min and attribute[i]<=max:
            age_in.append(1)
        else:
            age_in.append(0)

    for i in range(len(feat_dep)):
        if feat_dep[i] == 0:
            for j in range(len(attribute)):
                attribute_mat[j][i]=age_in[j]
        else:
            for j in range(len(attribute)):
                attribute_mat[j][i]=0

    for j in range(len(attribute)):
        attribute_mat[j][-1]=age_in[j]

    return attribute_mat

def support_continuous_binary_synth(feat_dep, attribute_bin, attribute_cont, val, min, max):
    """
        feat_dep       : array with feature dependencies (0: age related , 1: sex related ,2: age and sex related)
        attribute_bin : binary attribute
        attribute_cont : continuous attribute
        val : value taken by the binary attribute
        min : value min taken by the continuous attribute that is included
        max : value max taken by the continuous attribute that is included

    return:
        attribute_mat        : mask taking into account value
    """
    print((len(feat_dep)+1, len(attribute_bin)))
    attribute_mat = np.zeros((len(attribute_bin), len(feat_dep)+1))
    age_sex=[]
    for i in range(len(attribute_cont)):
        if attribute_cont[i] >=min and attribute_cont[i]<=max:
            if attribute_bin[i] == val:
                age_sex.append(1)
            else:
                age_sex.append(0)
        else:
            age_sex.append(0)

    for i in range(len(feat_dep)):
        if feat_dep[i] == 2:
            for j in range(len(attribute_bin)):
                attribute_mat[j][i]=age_sex[j]
        else:
            for j in range(len(attribute_bin)):
                attribute_mat[j][i]=0
    for j in range(len(attribute_bin)):
        attribute_mat[j][-1]=age_sex[j]

    return attribute_mat


def mask_synth(feat_dep, attribute, value):
    """
        mask       : mask (0 and 1) on the feature matrix that only keeps the features that are related to value
        attribute
        value : which value to keep in attribute

    return:
        attribute_mat        : mask taking into account value
    """
    print(len(feat_dep)+1, len(attribute))
    attribute_mat = np.zeros((len(attribute), len(feat_dep)+1))

    for i in range(len(feat_dep)):
        if feat_dep[i] == value:
            for j in range(len(attribute)):
                attribute_mat[j][i]=1
        else:
            for j in range(len(attribute)):
                attribute_mat[j][i]=0

    for j in range(len(attribute)):
        attribute_mat[j][-1]=1

    return attribute_mat


sex=[]
SYNTH=False

if __name__ == '__main__':
    if SYNTH:

        path_feat_dep = 'feat_dependence_synthetic_data_noteasy.csv'
        feat_dep, _, _ = read_tadpole.load_csv_no_header(path_feat_dep)
        feat_dep=preprocessing_dataset.str_to_float(feat_dep)

        age_file_path = 'ages_synthetic_data_noteasy.csv'
        gender_file_path = 'sexs_synthetic_data_noteasy.csv'

        age_column, nrRowsa, nrColsa = read_tadpole.load_csv_no_header(age_file_path)
        gender_column, nrRowsg, nrColsg = read_tadpole.load_csv_no_header(gender_file_path)
        age_column=preprocessing_dataset.str_to_float(age_column)
        gender_column=preprocessing_dataset.str_to_float(gender_column)

        mask_age = mask_synth(feat_dep, age_column, 0)
        mask_sex = mask_synth(feat_dep, gender_column, 1)
        mask_agesex = mask_synth(feat_dep, age_column, 2)
        np.savetxt('mask_age_synthnoteasy.csv', mask_age, fmt='%f', delimiter=',')
        np.savetxt('mask_sex_synthnoteasy.csv', mask_sex, fmt='%f', delimiter=',')
        np.savetxt('mask_agesex_synthnoteasy.csv', mask_agesex, fmt='%f', delimiter=',')

        age_min=54
        for age in [59, 64,69,74,79,84,92]:
            age_m = support_continuous_binary_synth(feat_dep, gender_column, age_column, 0, age_min, age)
            np.savetxt('age_%i_%i_men_synth.csv' %(age_min, age), age_m, fmt='%f', delimiter=',')
            age_w = support_continuous_binary_synth(feat_dep, gender_column, age_column, 1, age_min, age)
            np.savetxt('age_%i_%i_women_synth.csv' %(age_min, age), age_w, fmt='%f', delimiter=',')
            age_array = support_continuous_synth(feat_dep, age_column, age_min, age)
            np.savetxt('age_%i_%i_synth.csv' %(age_min, age), age_array, fmt='%f', delimiter=',')
            age_min = age

        women = support_binary_synth(feat_dep, gender_column, 1)
        np.savetxt('women_synth.csv', women, fmt='%f', delimiter=',')
        men = support_binary_synth(feat_dep, gender_column, 0)
        np.savetxt('men_synth.csv', men, fmt='%f', delimiter=',')

    else:
        age_file_path = 'age.csv'
        gender_file_path = 'gender.csv'

        age_column, nrRowsa, nrColsa = read_tadpole.load_csv_no_header(age_file_path)
        gender_column, nrRowsg, nrColsg = read_tadpole.load_csv_no_header(gender_file_path)
        for i in range(len(gender_column)):
            if gender_column[i] == 'F':
                sex.append(1)
            else:
                sex.append(0)

        sex=np.array(sex)
        age_column=preprocessing_dataset.str_to_float(age_column)
        gender_column=preprocessing_dataset.str_to_float(sex)

        path_mask_X1 = 'mask_age.csv'
        path_mask_X2 = 'mask_sex.csv'
        path_mask_X3 = 'mask_agesex.csv'
        path_mask_X4 = 'mask_nosignificance.csv'

        mask_age, _, _ = read_tadpole.load_csv_no_header(path_mask_X1)
        mask_sex, _, _ = read_tadpole.load_csv_no_header(path_mask_X2)
        mask_agesex, _, _ = read_tadpole.load_csv_no_header(path_mask_X3)
        mask_nosignificance, _, _ = read_tadpole.load_csv_no_header(path_mask_X4)

        mask_age=preprocessing_dataset.str_to_float(mask_age)
        mask_sex= preprocessing_dataset.str_to_float(mask_sex)
        mask_agesex=preprocessing_dataset.str_to_float(mask_agesex)
        mask_nosignificance=preprocessing_dataset.str_to_float(mask_nosignificance)

        men_sex = support_binary(mask_sex, sex, 0)
        women_sex = support_binary(mask_sex, sex, 1)
        np.savetxt('sex_men.csv', men_sex, fmt='%f', delimiter=',')
        np.savetxt('sex_women.csv', women_sex, fmt='%f', delimiter=',')
        age_min=54
        for age in [59, 64,69,74,79,84,92]:
            age_m = support_continuous_binary(mask_agesex, sex, age_column, 0, age_min, age)
            np.savetxt('age_%i_%i_men.csv' %(age_min, age), age_m, fmt='%f', delimiter=',')
            age_w = support_continuous_binary(mask_agesex, sex, age_column, 1, age_min, age)
            np.savetxt('age_%i_%i_women.csv' %(age_min, age), age_w, fmt='%f', delimiter=',')
            age_array = support_continuous(mask_age, age_column, age_min, age)
            np.savetxt('age%i_%i.csv' %(age_min, age), age_array, fmt='%f', delimiter=',')
            age_min = age

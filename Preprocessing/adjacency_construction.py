#Created by Juliette Valenchon 04/2019, based on Sarah Parisot for the first function
############################################################
##
##  CREATION OF THE 4 ADJACENCY MATRICES FOR THE
##
###########################################################

import numpy as np
import os
import csv
import read_tadpole

# Construct the adjacency matrix from Parisot et al
def adjacency_age_sex_agesex(attributes):
    """
    Function to built the adjacency (A+2I) matrix based on age, sex and age & sex. Adjancency matrix used by Parisot et al
    Value in adjacency is 2 if sujects have the same sex and an age gap inferior to 2,
                          1 if subjects have an age gap inferior to 2 or same sex and
                          0 otherwise
    Inputs:
        attributes: list of phenotypic information to be used to construct the adjacency matrix.
        Vector composed of 2 columns, the first one being composed of the age of subjects and the second one their sex
    return:
        adj: adjacency matrix of the graph (num_subjects x num_subjects)
    """

    num_nodes = attributes.shape[0]
    adj = np.zeros((num_nodes, num_nodes))

    for l in range(attributes.shape[1]):
        if l == 0:
            for k in range(num_nodes):
                adj[k, k] = 2
                for j in range(k + 1, num_nodes):
                    val = abs(float(attributes[k][l]) - float(attributes[j][l]))
                    if val < 2:
                        adj[k, j] += 1
                        adj[j, k] += 1
        elif l==1:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    if attributes[k][l] == attributes[j][l]:
                        adj[k, j] += 1
                        adj[j, k] += 1
        else:
            print("we only use two attributes here to built the adjacency matrix ")

    return adj

def adjacency_age(attribute):
    """
    Function to built the adjacency (A+I) matrix based on age. Value in adjacency is 1 if subjects have an age gap inferior to 2 and 0 otherwise
    Inputs:
        attribute: list of phenotypic information to be used to construct the adjacency matrix.
        Vector composed of 1 column composed of the sex of the subjects
    return:
        adj: adjacency matrix of the graph (num_subjects x num_subjects)
    """
    num_nodes = attribute.shape[0]
    adj = np.zeros((num_nodes, num_nodes))
    for k in range(num_nodes):
        adj[k, k] = 1
        for j in range(k + 1, num_nodes):
            val = abs(float(attribute[k]) - float(attribute[j]))
            if val < 2:
                adj[k, j] += 1
                adj[j, k] += 1
    return adj

def adjacency_sex(attribute):
    """
    Function to built the adjacency (A+I) matrix based on sex. Value in adjacency is 1 if subjects have the same sex and 0 otherwise
    Inputs:
        attribute: list of phenotypic information to be used to construct the adjacency matrix.
        Vector composed of 1 column composed of the sex of the subjects
    return:
        adj: adjacency matrix of the graph (num_subjects x num_subjects)
    """
    num_nodes = attribute.shape[0]
    adj = np.zeros((num_nodes, num_nodes))
    for k in range(num_nodes):
        adj[k, k] = 1
        for j in range(k + 1, num_nodes):
            if attribute[k] == attribute[j]:
                adj[k, j] += 1
                adj[j, k] += 1
    return adj

def adjacency_agesex(attributes):
    """
    Function to built the adjacency (A+I) matrix based on age & sex. Value in adjacency is 1 if subjects have the same age and same sex and 0 otherwise
    Inputs:
        attributes: list of phenotypic information to be used to construct the adjacency matrix.
        Vector composed of 2 columns, the first one being composed of the age of subjects and the second one their sex
    return:
        adj: adjacency matrix of the graph (num_subjects x num_subjects)
    """
    num_nodes = attributes.shape[0]
    adj = np.zeros((num_nodes, num_nodes))
    for k in range(num_nodes):
        adj[k, k] = 1
        for j in range(k + 1, num_nodes):
            val = abs(float(attributes[k][0]) - float(attributes[j][0]))
            if val < 2:
                if attributes[k][1] == attributes[j][1]:
                    adj[k, j] += 1
                    adj[j, k] += 1
    return adj

def adjacency_age_sex_agesex_apoe(attributes):
    """
    Function to built the adjacency (A+3I) matrix based on age, sex, age & sex and APOE. Adjancency matrix used by Parisot et al
    Value in adjacency is 3 if sujects have the same sex and an age gap inferior to 2 and the same APOE allele,
                          2 if sujects have the same sex and an age gap inferior to 2, the same sex and the same APOE allele,
                             an age gap inferior to 2 and the same APOE allele,
                          1 if subjects have an age gap inferior to 2 or same sex or the same APOE allele and
                          0 otherwise
    Inputs:
        attributes: list of phenotypic information to be used to construct the adjacency matrix.
        Vector composed of 2 columns, the first one being composed of the age of subjects and the second one their sex
    return:
        adj: adjacency matrix of the graph (num_subjects x num_subjects)
    """

    num_nodes = attributes.shape[0]
    adj = np.zeros((num_nodes, num_nodes))

    for l in range(attributes.shape[1]):
        if l == 0:
            for k in range(num_nodes):
                adj[k, k] = 3
                for j in range(k + 1, num_nodes):
                    val = abs(float(attributes[k][l]) - float(attributes[j][l]))
                    if val < 2:
                        adj[k, j] += 1
                        adj[j, k] += 1
        elif l==1:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    if attributes[k][l] == attributes[j][l]:
                        adj[k, j] += 1
                        adj[j, k] += 1
        elif l==2:
            for k in range(num_nodes):
                for j in range(k+1, num_nodes):
                    if attributes[k][l]!= '' and attributes[j][l]!='':
                        if attributes[k][l] == attributes[j][l]:
                            adj[k, j] += 1
                            adj[j, k] += 1
        else:
            print("we only use two attributes here to built the adjacency matrix ")

    return adj

if __name__ == '__main__':
    age_file_path = 'age.csv'
    gender_file_path = 'gender.csv'

    age_column, nrRowsa, nrColsa = read_tadpole.load_csv_no_header(age_file_path)
    gender_column, nrRowsg, nrColsg = read_tadpole.load_csv_no_header(gender_file_path)

    """meta_data_graph = np.concatenate((age_column, gender_column), axis = 1)
    a3=adjacency_agesex(meta_data_graph)
    a2=adjacency_sex(gender_column)
    a1=adjacency_age(age_column)
    affinity_matrix = adjacency_age_sex_agesex(meta_data_graph)
    np.savetxt('affinity_mat_parisot_v1.csv', affinity_matrix, fmt='%f', delimiter=',')
    np.savetxt('affinity_mat_1.csv', a1, fmt='%f', delimiter=',')
    np.savetxt('affinity_mat_2.csv', a2, fmt='%f', delimiter=',')
    np.savetxt('affinity_mat_3.csv', a3, fmt='%f', delimiter=',')"""

    meta_data_graph = np.concatenate((age_column, gender_column), axis = 1)

    adja=adjacency_age_sex_agesex(meta_data_graph)
    np.savetxt('adjacency.csv', adja, fmt='%f', delimiter=',')

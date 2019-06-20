# Juliette Valenchon 04/2019

#################################################"
##
## PROGRAM TO FIND FEATURE DEPENDENCIES ON AGE, SEX AND AGE& SEX
##
###################################################


import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
font = {'family' : 'Times New Roman',
        'weight' : 'medium',
        'size'   : 20}
import matplotlib

matplotlib.rc('font', **font)
import numpy as np
from pylab import plot, show, savefig, xlim, figure, \
                hold, ylim, legend, boxplot, setp, axes

import pandas as pd
import seaborn

import read_tadpole
import preprocessing_dataset

path_dataset_matrix = 'bl_data_MCI_v1.csv' #matrix to complete
M_str, nrRows, nrCols = read_tadpole.load_csv_no_header(path_dataset_matrix)

path_dataset_header = 'header.csv' #labels of the different columns of features of M_str
header_str_col, _, _ = read_tadpole.load_csv_no_header(path_dataset_header)

gender_file_path = 'gender.csv'
gender_column, nrRowsg, nrColsg = read_tadpole.load_csv_no_header(gender_file_path)

age_file_path = 'age.csv'
age_column, nrRowsa, nrColsa = read_tadpole.load_csv_no_header(age_file_path)



M_str= preprocessing_dataset.normalization_gae(M_str)

#create a matrix X with the three covariates AGE, SEX, AGE&SEX
age = age_column.astype(np.float64)
sex = []
agenew= []
for i in range(len(gender_column)):
    if gender_column[i] == 'F':
        sex.append(1)
    else:
        sex.append(0)
    agenew.append(float(age[i][0]))
sex = np.array(sex)
agenew=np.array(agenew)
agesex = agenew*sex # dot_product(agenew, sex)
X = np.concatenate((agenew[:, np.newaxis], sex[:, np.newaxis], agesex[:, np.newaxis]), axis=1) #covariates

stat_array=[]
def feat_dependency(M_str, X, agenew, sex, agesex, plot=False):
    """
    Function to compute the feature dependency for each feature. Put the results in a csv file 'stat_results.csv'
    Input: plot: to plot the feature values as a function of age and sex
    """
    for j in range(M_str.shape[1]):
        #initialize X with all the values
        #initialize y with the value of feature in column j and remove from X and y the missing values
        y=[]
        index_to_delecte = []
        for i in range(M_str.shape[0]):
            if M_str[i][j] != -10:
                y.append((M_str[i][j]))
            else:
                index_to_delecte.append(i)

        X = np.delete(X, index_to_delecte, 0)
        y = np.array(y)

        data_np = np.concatenate((y[:,np.newaxis],X), axis=1)
        data = pd.DataFrame(data=data_np, columns=['feature', 'age', 'sex', 'agesex'])

        if plot:
            var = seaborn.lmplot(y='feature', x='age', hue='sex', data=data)
            plt.savefig('norm_seaborn_age_sex_feature_%i_icassp.pdf' %j )
            plt.show()

        #GLM
        lm2 = smf.ols('feature ~ age * sex',data=data).fit()

        stat_array.append([header_str_col[j][0], lm2.pvalues[1], lm2.pvalues[2], lm2.pvalues[3],lm2.rsquared]  )
        #reinitialize X with all subjects
        X = np.concatenate((agenew[:, np.newaxis], sex[:, np.newaxis], agesex[:, np.newaxis]), axis=1) #covariates

    stat_res = pd.DataFrame(data=stat_array, columns=['feature_name', 'age', 'sex', 'age_sex','r-squared'])
    stat_res.to_csv('stat_results.csv')



def plots_data_age_sex(X):
    """
    Plot number of women and men per age group in an histogram
    """
    data_np =X
    bins = [54,59,64,69,74,79,84,92]
    data = pd.DataFrame(data=data_np, columns=['age', 'sex', 'agesex'])

    data['group_age']=np.digitize(data.age, bins, right=True)
    counts=data.groupby(['group_age', 'sex']).age.count().unstack()
    print(counts)

    counts.plot(kind='bar', stacked=True)
    plt.show()

if __name__=="__main__":
    feat_dependency(M_str, X, agenew, sex, agesex, plot=True)
    plots_data_age_sex(X)

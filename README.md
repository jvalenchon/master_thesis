RUN in python 2 (2.7.13)
library to download: 
- Tensorflow 1.1.0
- RBFOpt (for optimization) https://github.com/coin-or/rbfopt 

FILES FOR BOTH ARCHITECTURES

read_tadpole.py : functions to preprocess the tadpole dataset and matrices stored in .csv files

preprocessing_dataset.py : functions for normalization, splitting of data, initialization of data...

wilcoxon_tesT.py : Wilcoxon score between two arrays
___________________________________________________________________________________________________________________________________________________________________________
PREPROCESSING

In FOLDER preprocessing  

COMPUTATION OF THE ADJACENCY MATRIX 
adjacency_construction.py : program to built the four adjacency matrices used in my work. 
files required to run the program:
   - gender column : 'gender.csv'
   - age column : 'age.csv'
programs required to run the program:
   - read_tadpole.py
creation of 'affinity_mat_parisot_v1.csv', 'affinity_mat_1.csv' (age), 'affinity_mat_2.csv' (sex), 'affinity_mat_3.csv' (age & sex)

FEATURE DEPENDENCY 
feature_dep.py: Can plot feature dependency with age and sex and also run a GLM through the data to compute the p-values for a linear model to automatically know the feature dependency with age, sex and age & sex
files required to run the program:
   - matrix M : 'bl_data_MCI_v1.csv'
   - gender column : 'gender.csv'
   - age column : 'age.csv'
   - header column (with the names of the columns of M) : 'header.csv'
programs required to run the program:
   - read_tadpole.py
   - preprocessing_dataset.py
creation of 'stat_results_v1.csv' and 'stat_results_v1_notext.csv' (no headers, only the p-values) 

mask_creation.py : function to create the masks on M depending on the attributes. Creates 4 binary matrices, one for age, one for sex, one for age&sex and one for no significance. It uses the statistical results from feature_dep.py and put a threshold on the p-value corresponding to an attribute for a feature to know if the feature is dependent on the attribute or not.
files required to run the program:
   - matrix M : 'bl_data_MCI_v1.csv'
   - statistical results without the headers 'stat_results_v1_notext.csv'
programs required to run the program:
   - read_tadpole.py
   - preprocessing_dataset.py 
creation of 'mask_age.csv', 'mask_sex.csv', 'mask_agesex.csv', 'mask_nosignificance.csv'

CREATION SYNTHETIC DATASET 
synthetic_data.py: creation of the synthetic dataset. Details are in the appendix of the thesis for the creation 
creation of - matrix M : 'synthetic_data_noteasy.csv'
            - labels Y : 'labels_synthetic_data_noteasy.csv'  
            - age column : 'ages_synthetic_data_noteasy.csv'
            - sex column : 'sexs_synthetic_data_noteasy.csv'
            - feature dependency : 'feat_dependence_synthetic_data_noteasy.csv' (to create the mask on M)

___________________________________________________________________________________________________________________________________________________________________________
MG-RGCNN

In FOLDER MGRGCNN  

mgrgcnn_hyperparam.py : run the MG-RGCNN algorithm. Can also do the optimization of the hyperparameters with RBFOpt. 
files required to run the program:
   - matrix M : 'bl_data_MCI_v1.csv'
   - masks on M for the different graphs : 'mask_age.csv', 'mask_sex.csv', 'mask_agesex.csv', 'mask_nosignificance.csv'
   - adjacency matrices : 'affinity_mat_1.csv' (age), 'affinity_mat_2.csv' (sex), 'affinity_mat_3.csv' (age & sex)
   - labels Y : 'labels_comma.csv'
programs required to run the program:
   - read_tadpole.py
   - preprocessing_dataset.py

___________________________________________________________________________________________________________________________________________________________________________
MG-GAE

In FOLDER MGGAE  

utils.py : program to create a dictionnary with the placeholders to run the algorithm with tensorflow

layers.py : program to define the layers of the MG-GAE architecture. 
programs required to run the program: 
   - initializations.py 

model.py : definition of the MG-GAE architecture with Tensorflow. 
programs required to run the program:
   - layers.py

preprocessing.py: functions to load the dataset for the MG-GAE. Creation of the inputs of the architecture and loading of the dataset from the csv file. 

MG_GAE_main.py: main file to perform the optimization of the hyperparameters with RBFOpt or to run the architecture for different training/validation/test initializations. For the TADPOLE dataset
programs required to run the program:
   - read_tadpole.py
   - preprocessing_dataset.py
   - preprocessing.py
   - model.py
   - utils.py
files required to run the program:
   - matrix M : 'bl_data_MCI_v1.csv'
   - labels Y : 'labels_comma.csv' 
   - all support matrices: "sex_women.csv", "sex_men.csv", "age_84_92_women.csv", "age_84_92_men.csv", "age84_92.csv", "age_79_84_women.csv", "age_79_84_men.csv", "age79_84.csv", "age_74_79_women.csv", "age_74_79_men.csv", "age74_79.csv", "age_69_74_women.csv", "age_69_74_men.csv", "age69_74.csv", "age_64_69_women.csv", "age_64_69_men.csv", "age64_69.csv", "age_59_64_women.csv", "age_59_64_men.csv", "age59_64.csv",  "age_54_59_women.csv", "age_54_59_men.csv", "age54_59.csv"   

MG_GAE_main_synth.py: main file to perform the optimization of the hyperparameters with RBFOpt or to run the architecture for different training/validation/test initializations. For the synthetic dataset
programs required to run the program:
   - read_tadpole.py
   - preprocessing_dataset.py
   - preprocessing.py
   - model.py
   - utils.py
files required to run the program:
   - matrix M : 'synthetic_data_noteasy.csv'
   - labels Y : 'labels_synthetic_data_noteasy.csv'  
   - all support matrices: "women_synth_noteasy.csv", "men_synth_noteasy.csv", "age_84_92_women_synth_noteasy.csv", "age_84_92_men_synth_noteasy.csv", "age_84_92_synth_noteasy.csv", "age_79_84_women_synth_noteasy.csv", "age_79_84_men_synth_noteasy.csv", "age_79_84_synth_noteasy.csv", "age_74_79_women_synth_noteasy.csv", "age_74_79_men_synth_noteasy.csv", "age_74_79_synth_noteasy.csv", "age_69_74_women_synth_noteasy.csv", "age_69_74_men_synth_noteasy.csv", "age_69_74_synth_noteasy.csv", "age_64_69_women_synth_noteasy.csv", "age_64_69_men_synth_noteasy.csv", "age_64_69_synth_noteasy.csv", "age_59_64_women_synth_noteasy.csv", "age_59_64_men_synth_noteasy.csv", "age_59_64_synth_noteasy.csv", "age_54_59_women_synth_noteasy.csv", "age_54_59_men_synth_noteasy.csv", "age_54_59_synth_noteasy.csv"

MG_GAE_creation_embeddings.py: function to create all the files needed to do the scatter plots of the 2 first components of the PCA decomposition. For the TADPOLE dataset
programs required to run the program:
   - read_tadpole.py
   - preprocessing_dataset.py
   - preprocessing.py
   - model.py
   - utils.py
files required to run the program:
   - matrix M : 'bl_data_MCI_v1.csv'
   - labels Y : 'labels_comma.csv' 
   - all support matrices: "sex_women.csv", "sex_men.csv", "age_84_92_women.csv", "age_84_92_men.csv", "age84_92.csv", "age_79_84_women.csv", "age_79_84_men.csv", "age79_84.csv", "age_74_79_women.csv", "age_74_79_men.csv", "age74_79.csv", "age_69_74_women.csv", "age_69_74_men.csv", "age69_74.csv", "age_64_69_women.csv", "age_64_69_men.csv", "age64_69.csv", "age_59_64_women.csv", "age_59_64_men.csv", "age59_64.csv",  "age_54_59_women.csv", "age_54_59_men.csv", "age54_59.csv"   

MG_GAE_creation_embeddings_synth.py: function to create all the files needed to do the scatter plots of the 2 first components of the PCA decomposition. For the synthetic dataset
programs required to run the program:
   - read_tadpole.py
   - preprocessing_dataset.py
   - preprocessing.py
   - model.py
   - utils.py
files required to run the program: 
   - matrix M : 'synthetic_data_noteasy.csv'
   - labels Y : 'labels_synthetic_data_noteasy.csv'  
   - all support matrices: "women_synth_noteasy.csv", "men_synth_noteasy.csv", "age_84_92_women_synth_noteasy.csv", "age_84_92_men_synth_noteasy.csv", "age_84_92_synth_noteasy.csv", "age_79_84_women_synth_noteasy.csv", "age_79_84_men_synth_noteasy.csv", "age_79_84_synth_noteasy.csv", "age_74_79_women_synth_noteasy.csv", "age_74_79_men_synth_noteasy.csv", "age_74_79_synth_noteasy.csv", "age_69_74_women_synth_noteasy.csv", "age_69_74_men_synth_noteasy.csv", "age_69_74_synth_noteasy.csv", "age_64_69_women_synth_noteasy.csv", "age_64_69_men_synth_noteasy.csv", "age_64_69_synth_noteasy.csv", "age_59_64_women_synth_noteasy.csv", "age_59_64_men_synth_noteasy.csv", "age_59_64_synth_noteasy.csv", "age_54_59_women_synth_noteasy.csv", "age_54_59_men_synth_noteasy.csv", "age_54_59_synth_noteasy.csv"

embedding_plots.py: function to plot the scatter plots of the 2 first components of PCA of the outputs of the GCNN layer for the embeddings of the subjects. cf Section 5.4.3. of Thesis
files required to run the program: all these files are generated from the algorithm that runs the architecture (MG_GAE_creation_embeddings_synth.py)
    - matrix composed of the outputs of the GCNN layer for one set (training, test or validation). 
                                                        Example for the synthetic dataset: "gcn_u_train_emb128_synthnoteasy_nonorm.csv"
                                                        Example for the TADPOLE dataset: "gcn_u_train_emb217_tadpole_nonorm.csv"
    - array composed of the sex of the subjects of the set. 
                                                        Example for the synthetic dataset: "gender_column_train_synthnoteasy.csv"
                                                        Example for the TADPOLE dataset: "gender_column_train_tadpole217.csv"
    - array composed of the age of the subjects of the set. 
                                                        Example for the synthetic dataset: "age_column_train_synthnoteasy.csv"
                                                        Example for the TADPOLE dataset: "age_column_train_tadpole217.csv"
    - array composed of the labels of the subjects of the set. 
                                                        Example for the synthetic dataset: "labels_column_train_synthnoteasy.csv"
                                                        Example for the TADPOLE dataset: "labels_column_train_tadpole217.csv"
Parameters: 
    - number of outputs of the GCNN layer: hidden_a, defined line 52
                                                        Example for the synthetic dataset: 61
                                                        Example for the TADPOLE dataset: 99
For each one of the 23 support matrices, we do a PCA decomposition of the hidden_a outputs of the GCNN layer. 
In the plot, we differentiate the subjects that have the value of the attribute(s) that defines the support matrix and the one that do not have it. We split the subjects that have the value of the attribute(s) that defines the support matrix between the one that are MCIc and the one that are MCInc. 
The support appears in the following order: Women, Men, Women from 84 to 92, Men from 84 to 92, age between 84 and 92, Women from 79 to 84, Men from 79 and 84, age between 79 and 84, Women from 74 to 79, Men from 74 to 79, age between 74 and 79, Women from 69 to 74, Men from 69 and 74, age between 69 and 74, Women from 64 to 69, Men from 64 to 69, age between 64 and 69, Women from 59 to 64, Men from 59 and 64, age between 59 and 64, Women from 54 to 59, Men from 54 to 59, age between 54 and 59.
___________________________________________________________________________________________________________________________________________________________
OTHER METHODS
___________________________________________________________________________________________________________________________________________________________
Random Forest, SVM, linear ridge, MLP
standard_methods_mean_imputation.py : 
4 standards methods for classification of data with missing values. Missing values imputed by putting the mean value
   - random forest
   - linear SVM 
   - linear ridge
   - MLP one layer
___________________________________________________________________________________________________________________________________________________________
Replication of Vivar et al 
vivar.py : replication of the architecture used by Vivar et al. Can do the optimization of the hyperparameters with RBFOpt.
files required to run the program:
   - matrix M : 'bl_data_MCI_v1.csv'
   - adjacency matrix : 'affinity_mat_parisot_v1.csv' 
   - labels Y : 'labels_comma.csv'
programs required to run the program:
   - read_tadpole.py
   - preprocessing_dataset.py

___________________________________________________________________________________________________________________________________________________________________________
Replication of Parisot et al (ChebyNet architecture)
parisot.py: replication of the architecture used by Parisot et al. Can do the optimization of the hyperparameters with RBFOpt.
files required to run the program:
   - matrix M : 'bl_data_MCI_v1.csv'
   - adjacency matrix : 'affinity_mat_parisot_v1.csv' 
   - labels Y : 'labels_comma.csv'
programs required to run the program:
   - read_tadpole.py
   - preprocessing_dataset.py

GCNN-based architecture. 
parisot_gcn.py: main file to perform the optimization of the hyperparameters with RBFOpt or to run the architecture for different training/validation/test initializations. For the TADPOLE dataset
files required to run the program:
   - matrix M : 'bl_data_MCI_v1.csv'
   - adjacency matrix : 'affinity_mat_parisot_v1.csv' 
   - labels Y : 'labels_comma.csv'
programs required to run the program:
   - read_tadpole.py
   - preprocessing_dataset.py
___________________________________________________________________________________________________________________________________________________________________________
MG-GCNN

In FOLDER main  

mg_gcnn_main.py: run the MG-GCNN algorithm. Can also do the optimization of the hyperparameters with RBFOpt. 
files required to run the program:
   - matrix M : 'bl_data_MCI_v1.csv'
   - masks on M for the different graphs : 'mask_age.csv', 'mask_sex.csv', 'mask_agesex.csv', 'mask_nosignificance.csv'
   - adjacency matrices : 'affinity_mat_1.csv' (age), 'affinity_mat_2.csv' (sex), 'affinity_mat_3.csv' (age & sex)
   - labels Y : 'labels_comma.csv'
programs required to run the program:
   - read_tadpole.py
   - preprocessing_dataset.py

og_gcnn_main.py: run the OG-GCNN algorithm. Can also do the optimization of the hyperparameters with RBFOpt. 
files required to run the program:
   - matrix M : 'bl_data_MCI_v1.csv'
   - adjacency matrix : 'affinity_mat_parisot_v1.csv' 
   - labels Y : 'labels_comma.csv'
programs required to run the program:
   - read_tadpole.py
- preprocessing_dataset.py

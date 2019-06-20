# created by Juliette Valenchon 04/2019

#########################################################################
##
## PROGRAM TO PREPROCESS THE RAW TADPOLE DATA ('TADPOLE_D1_D2.csv'). Register on the ADNI website to have the dataset.
## REMOVE THE HEADER AND STORE THEM
## REMOVE THE SUBJECTS THAT ARE NEITHER MCIc OR MCInc (definition of MCIc and MCInc in the thesis)
## REMOVE THE FEATURES THAT ARE NOT MEDICAL DATA AND THAT HAVE MORE THAN 50% OF MISSING VALUES
##
########################################################################


import csv
import numpy as np
import argparse
from argparse import RawTextHelpFormatter


dataType = 'S%d' %100

def load_csv_header(filePath):
    """
    Load data of csv file into a matrix. The first row of the csv file is composed of the names for the columns
    Input:
        filePath : file path to the .csv file
    Outputs:
        mergeAll: matrix of str with the data in the file path.
        header: array with the names of the column of the .csv file
        nrRows: int, number of rows of the matrix
        nrCols: int, number of columns of the matrix
    """
    with open(filePath, 'r') as f:
        reader = csv.reader(f, delimiter = ',', quotechar = '"')
        rows = [row for row in reader]
        header = rows[0]
        header = np.array(header)
        rows = rows[1:]
        nrRows = len(rows)
        nrCols = len(rows[0])
        mergeAll = np.ndarray((nrRows, nrCols), dtype=dataType)
        mergeAll[:] = ' '
        for r in range(nrRows):
          mergeAll[r,:] = [str.encode(word) for word in rows[r]]
    return mergeAll, header, nrRows, nrCols

def load_csv_no_header(filePath):
    """
    Load data of csv file into a matrix. The csv file is only composed of data, no header.
    Input:
        filePath : file path to the .csv file
    Outputs:
        mergeAll: matrix of str with the data in the file path.
        nrRows: int, number of rows of the matrix
        nrCols: int, number of columns of the matrix
    """
    with open(filePath, 'r') as f:
        reader = csv.reader(f, delimiter = ',', quotechar = "'")
        rows = [row for row in reader]
        nrRows = len(rows)
        nrCols = len(rows[0])
        mergeAll = np.ndarray((nrRows, nrCols), dtype=dataType)
        mergeAll[:] = ' '
        for r in range(nrRows):
            mergeAll[r,:] = [str.encode(word) for word in rows[r]]
    return mergeAll, nrRows, nrCols

def load_csv_no_header_apoe(filePath):
    """
    Load apoe csv file into a matrix. The csv file is only composed of data, no header.
    Input:
        filePath : file path to the .csv file
    Outputs:
        mergeAll: matrix of str with the data in the file path.
        nrRows: int, number of rows of the matrix
        nrCols: int, number of columns of the matrix
    """
    with open(filePath, 'r') as f:
        reader = csv.reader(f, delimiter = ',', quotechar = "'")
        rows = [row for row in reader]
        print(rows)
        nrRows = len(rows)
        nrCols = len(rows[0])
        mergeAll = np.ndarray((nrRows, nrCols), dtype=dataType)
        mergeAll[:] = ' '
        for r in range(nrRows):
            if rows[r] != []:
                mergeAll[r,:] = [str.encode(word) for word in rows[r]]
    return mergeAll, nrRows, nrCols


def extract_cMCI_sMCI(mergeAll, header, nrRows):
    """
    Extraction of the labels from the dataset. Taking into account the baseline examination to remove subjects that are not MCI.
    Inputs:
        mergeAll: matrix with data
        header: header corresponding to the columns of the matrix
        nrRows: number of rows of mergeAll
    Outputs:
        bl_data_MCI: matrix with the subjects that are MCI at baseline
        labels: array with labels.
            -1: subjects that are neither MCIc or MCInc and that we need to remove
            0: subjects that are MCInc
            1: subjects that are MCIc
    """
    DX_bl_index = np.where(header == 'DX_bl' )
    DXChange_index = np.where(header == 'DXCHANGE' )
    RID_index = np.where(header == 'RID' )
    viscode_index = np.where(header == 'VISCODE' )
    viscode_index=int(viscode_index[0][0])
    DX_bl_index=int(DX_bl_index[0][0])
    DXChange_index=int(DXChange_index[0][0])
    RID_index=int(RID_index[0][0])
    data_MCI = []
    bl_data_MCI =[]
    labels = []
    for i in range(nrRows):
        if (mergeAll[i, DX_bl_index] == 'LMCI' or mergeAll[i, DX_bl_index] == 'EMCI'):
            data_MCI.append(mergeAll[i,:])
    RID_used = []
    data_MCI = np.array(data_MCI)
    Viscode_save = []
    RID_save = []
    DXChange_save = []
    for j in range(len(data_MCI)):
        RID_val = data_MCI[j, RID_index]
        if RID_val not in RID_used:
            RID_used.append(RID_val)
            DXChange_evo = []
            Viscode_evo = []
            if data_MCI[j, DXChange_index] != '':
                if data_MCI[j, viscode_index] == 'bl':
                    bl_data_MCI.append(data_MCI[j, :])
                    data_MCI[j, viscode_index] = 'm00'
                DXChange_evo.append(data_MCI[j, DXChange_index])
                Viscode_evo.append(data_MCI[j, viscode_index][1:])
            for k in range(j+1, len(data_MCI)):
                RID_val2 = data_MCI[k, RID_index]
                if RID_val == RID_val2:
                    if data_MCI[k, DXChange_index] != '':
                        if data_MCI[k, viscode_index] == 'bl':
                            bl_data_MCI.append(data_MCI[k, :])
                            data_MCI[k, viscode_index] = 'm00'

                        DXChange_evo.append(data_MCI[k, DXChange_index])
                        Viscode_evo.append(data_MCI[k, viscode_index][1:])
            Viscode_evo = np.array(Viscode_evo, dtype =int)
            DXChange_evo = np.array(DXChange_evo)
            permu = np.argsort(Viscode_evo)
            Viscode_evo= Viscode_evo[permu]
            DXChange_evo = DXChange_evo[permu]
            for l in range(len(DXChange_evo)):
                if 0 in Viscode_evo:
                    if (DXChange_evo[l] == '5' or DXChange_evo[l] == '3' ) and (Viscode_evo[l]) <= 48:
                        labels.append('1')
                        Viscode_save.append(Viscode_evo)
                        DXChange_save.append(DXChange_evo)
                        RID_save.append(RID_val)
                        break
                    elif (DXChange_evo[l] == '5' or DXChange_evo[l] == '3' ) and (Viscode_evo[l]) > 48:
                        labels.append('-1')
                        break
                    elif int(Viscode_evo[l]) == max(Viscode_evo) and DXChange_evo[l] == '2':
                        labels.append('0')
                        Viscode_save.append(Viscode_evo)
                        DXChange_save.append(DXChange_evo)
                        RID_save.append(RID_val)
                        break
                    elif (DXChange_evo[l] == '7' or DXChange_evo[l] == '1' or DXChange_evo[l] == '8' or DXChange_evo[l] == '4'):
                        labels.append('-1')
                        break

    bl_data_MCI =np.array(bl_data_MCI)
    labels =np.array(labels)
    #np.savetxt('Visocde_evo_comma.csv', Viscode_save, fmt='%s', delimiter=',')
    #np.savetxt('DXchange_evo_comma.csv', DXChange_save, fmt='%s', delimiter=',')
    #np.savetxt('RID_values_comma.csv', RID_save, fmt='%s', delimiter=',')
    return bl_data_MCI, labels

def remove_rows_wrong_label(bl_data_MCI, labels):
    """
    Remove subjects that are associated to a label of -1 (neither MCIc or MCInc)
    Inputs:
        bl_data_MCI: matrix with MCI subjects
        labels: labels associated to the subjects in bl_data_MCI
    Outputs:
        bl_data_MCI: matrix with only MCIc or MCInc subjects
        labels: labels associated to the subjects in bl_data_MCI
    """
    labels_indexes = np.where(labels=='-1')[0]
    bl_data_MCI = np.delete(bl_data_MCI, labels_indexes, 0)
    labels = np.delete(labels, labels_indexes, 0)
    return bl_data_MCI, labels

def extract_column_str(bl_data, header, str_to_remove):
    """
    Extract column associated to str_to_remove in header. Remove that column of str from bl_data
    Inputs:
        bl_data: matrix with data
        header: header corresponding to the columns of the matrix bl_data
        str_to_remove: str in header to remove from bl_data
    Outputs:
        bl_data: matrix with data where we remove the column corresponding to str_to_remove
        header: header corresponding to the columns of the matrix bl_data
        column: column extracted from bl_data corresponding to str_to_remove. Column composed of str values
    """
    index = np.where(header ==str_to_remove)
    index=int(index[0][0])
    column= np.chararray((bl_data.shape[0]))
    column[:] = bl_data[:,index]
    bl_data = np.hstack((bl_data[:,:index], bl_data[:,(index+1):]))
    header = np.hstack((header[:index], header[(index+1):]))
    return bl_data, header, column


def extract_column_float(bl_data, header, str_to_remove):
    """
    Extract column associated to str_to_remove in header. Remove that column of float from bl_data
    Inputs:
        bl_data: matrix with data
        header: header corresponding to the columns of the matrix bl_data
        str_to_remove: str in header to remove from bl_data
    Outputs:
        bl_data: matrix with data where we remove the column corresponding to str_to_remove
        header: header corresponding to the columns of the matrix bl_data
        column: column extracted from bl_data corresponding to str_to_remove. Columns composed of float values
    """
    index = np.where(header ==str_to_remove) #str
    index=int(index[0][0])
    column= np.ndarray((bl_data.shape[0]))
    column[:] = bl_data[:,index]
    bl_data = np.hstack((bl_data[:,:index], bl_data[:,(index+1):]))
    header = np.hstack((header[:index], header[(index+1):]))
    return bl_data, header, column

def remove_column(bl_data, header, str_to_remove):
    """
    Remove column associated to str_to_remove in header.
    Inputs:
        bl_data: matrix with data
        header: header corresponding to the columns of the matrix bl_data
        str_to_remove: str in header to remove from bl_data
    Outputs:
        bl_data: matrix with data where we remove the column corresponding to str_to_remove
        header: header corresponding to the columns of the matrix bl_data
    """
    index = np.where(header ==str_to_remove)
    index=int(index[0][0])
    bl_data = np.hstack((bl_data[:,:index], bl_data[:,(index+1):]))
    header = np.hstack((header[:index], header[(index+1):]))
    return bl_data, header

def remove_column_pattern(bl_data, header, piece_str_to_remove):
    """
    Remove columns where piece_str_to_remove is in the header.
    Inputs:
        bl_data: matrix with data
        header: header corresponding to the columns of the matrix bl_data
        piece_str_to_remove: str in header that can have multiple apparition to remove from bl_data
    Outputs:
        bl_data: matrix with data where we remove the columns corresponding to piece_str_to_remove
        header: header corresponding to the columns of the matrix bl_data
    """
    index = [i for i,item in enumerate(header) if piece_str_to_remove in item]
    index.sort(reverse = True)
    for i in range(len(index)):
        bl_data = np.hstack((bl_data[:,:index[i]], bl_data[:,index[i]+1:]))
        header = np.hstack((header[:index[i]], header[index[i]+1:]))
    return bl_data, header

def save_results() :
    """
    Function to save the different matrix and arrays to csv files.
    """
    np.savetxt("bl_data_MCI.csv", bl_data_MCI, fmt='%s', delimiter=',')
    np.savetxt("header.csv", header, fmt='%s', delimiter=',')
    np.savetxt("gender.csv", gender_column, fmt='%s', delimiter=',')
    np.savetxt("age.csv", age_column, delimiter=',')
    np.savetxt('labels.csv', labels, fmt='%s', delimiter=',')


def remove_column_too_many_missing_values(bl_data_MCI, header ):
    """
    Remove columns from bl_data_MCI when there are too many missing values in the column
    Inputs:
        bl_data: matrix with data
        header: header corresponding to the columns of the matrix bl_data
    Outputs:
        bl_data: matrix with data where we remove the columns that have too many missing values
        header: header corresponding to the columns of the matrix bl_data
    """
    column_remove = []
    for i in range(bl_data_MCI.shape[1]):
        it = 0
        for j in range(bl_data_MCI.shape[0]):
            if bl_data_MCI[j][i] == '' or bl_data_MCI[j][i] == ' ':
                bl_data_MCI[j][i] = ''
                it +=1
        if it >=  bl_data_MCI.shape[0] /2:
            column_remove.append(i)
    column_remove.sort(reverse = True)
    for i in range(len(column_remove)):
        bl_data_MCI = np.hstack((bl_data_MCI[:,:column_remove[i]], bl_data_MCI[:,column_remove[i]+1:]))
        header = np.hstack((header[:column_remove[i]], header[column_remove[i]+1:]))
    return bl_data_MCI, header

def preprocessing():
    """
    Function to preprocess the TADPOLE dataset.
    """
    filePath= 'TADPOLE_D1_D2.csv'
    mergeAll, header, nrRows, nrCols = load_csv_header(filePath)
    bl_data_MCI, labels =extract_cMCI_sMCI(mergeAll, header, nrRows)
    bl_data_MCI, labels = remove_rows_wrong_label(bl_data_MCI, labels)
    bl_data_MCI, header, gender_column = extract_column_str(bl_data_MCI, header, 'PTGENDER')
    bl_data_MCI, header, age_column =extract_column_float(bl_data_MCI, header, 'AGE')
    bl_data_MCI, header, D1_column = extract_column_float(bl_data_MCI, header, 'D1')
    bl_data_MCI, header, D2_column = extract_column_float(bl_data_MCI, header, 'D2')
    #print(bl_data_MCI.shape)
    #bl_data_MCI, header, apoe = extract_column_str(bl_data_MCI, header, 'APOE4')
    #np.savetxt('apoe_col.csv', apoe, fmt='%s', delimiter=',')

    """bl_data_MCI, header = remove_column(bl_data_MCI, header, 'RID')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'PTID')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'VISCODE')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'SITE')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'COLPROT')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'ORIGPROT')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'EXAMDATE')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'DX_bl')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'DXCHANGE')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'PTEDUCAT')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'PTETHCAT')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'PTRACCAT')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'PTMARRY')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'APOE4')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'PIB')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'CDRSB')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'ADAS11')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'ADAS13')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'MMSE')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'RAVLT_immediate')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'RAVLT_learning')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'RAVLT_forgetting')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'RAVLT_perc_forgetting')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'FAQ')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'MOCA')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'EcogPtMem')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'EcogPtLang')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'EcogPtVisspat')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'EcogPtPlan')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'EcogPtOrgan')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'EcogPtDivatt')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'EcogPtTotal')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'EcogSPMem')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'EcogSPLang')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'EcogSPVisspat')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'EcogSPPlan')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'EcogSPOrgan')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'EcogSPDivatt')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'EcogSPTotal')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'FLDSTRENG')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'FSVERSION')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'DX')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'EXAMDATE_bl')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'CDRSB_bl')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'ADAS11_bl')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'ADAS13_bl')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'MMSE_bl')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'RAVLT_immediate_bl')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'RAVLT_learning_bl')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'RAVLT_forgetting_bl')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'RAVLT_perc_forgetting_bl')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'FAQ_bl')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'FLDSTRENG_bl')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'FSVERSION_bl')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'Ventricles_bl')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'Hippocampus_bl')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'WholeBrain_bl')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'Entorhinal_bl')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'Fusiform_bl')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'MidTemp_bl')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'ICV_bl')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'MOCA_bl')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'EcogPtMem_bl')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'EcogPtLang_bl')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'EcogPtVisspat_bl')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'EcogPtPlan_bl')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'EcogPtOrgan_bl')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'EcogPtDivatt_bl')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'EcogPtTotal_bl')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'EcogSPMem_bl')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'EcogSPLang_bl')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'EcogSPVisspat_bl')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'EcogSPPlan_bl')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'EcogSPOrgan_bl')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'EcogSPDivatt_bl')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'EcogSPTotal_bl')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'FDG_bl')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'PIB_bl')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'AV45_bl')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'Years_bl')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'Month_bl')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'Month')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'M')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'update_stamp')

    bl_data_MCI, header = remove_column_pattern(bl_data_MCI, header, '_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16')
    #bl_data_MCI, header = remove_column(bl_data_MCI, header, 'RID_UCSFFSX_11_02_15_UCSFFSX51_08_01_16')
    #bl_data_MCI, header = remove_column(bl_data_MCI, header, 'VISCODE_UCSFFSX_11_02_15_UCSFFSX51_08_01_16')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'EXAMDATE_UCSFFSX_11_02_15_UCSFFSX51_08_01_16')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'VERSION_UCSFFSX_11_02_15_UCSFFSX51_08_01_16')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'LONISID_UCSFFSX_11_02_15_UCSFFSX51_08_01_16')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'FLDSTRENG_UCSFFSX_11_02_15_UCSFFSX51_08_01_16')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'LONIUID_UCSFFSX_11_02_15_UCSFFSX51_08_01_16')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'IMAGEUID_UCSFFSX_11_02_15_UCSFFSX51_08_01_16')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'RUNDATE_UCSFFSX_11_02_15_UCSFFSX51_08_01_16')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'STATUS_UCSFFSX_11_02_15_UCSFFSX51_08_01_16')

    bl_data_MCI, header = remove_column_pattern(bl_data_MCI, header, 'QC_UCSFFSX_11_02_15_UCSFFSX51_08_01_16')

    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'update_stamp_UCSFFSX_11_02_15_UCSFFSX51_08_01_16')
    #bl_data_MCI, header = remove_column(bl_data_MCI, header, 'RID_BAIPETNMRC_09_12_16')
    #bl_data_MCI, header = remove_column(bl_data_MCI, header, 'VISCODE_BAIPETNMRC_09_12_16')
    #bl_data_MCI, header = remove_column(bl_data_MCI, header, 'VISCODE2_BAIPETNMRC_09_12_16')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'EXAMDATE_BAIPETNMRC_09_12_16')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'VERSION_BAIPETNMRC_09_12_16')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'LONIUID_BAIPETNMRC_09_12_16')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'RUNDATE_BAIPETNMRC_09_12_16')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'STATUS_BAIPETNMRC_09_12_16')

    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'update_stamp_BAIPETNMRC_09_12_16')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'EXAMDATE_UCBERKELEYAV45_10_17_16')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'update_stamp_UCBERKELEYAV45_10_17_16')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'EXAMDATE_UCBERKELEYAV1451_10_17_16')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'update_stamp_UCBERKELEYAV1451_10_17_16')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'EXAMDATE_DTIROI_04_30_14')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'VERSION_DTIROI_04_30_14')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'VISNAME_DTIROI_04_30_14')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'VISITNO_DTIROI_04_30_14')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'LONIUID_1_DTIROI_04_30_14')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'LONIUID_2_DTIROI_04_30_14')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'LONIUID_3_DTIROI_04_30_14')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'LONIUID_4_DTIROI_04_30_14')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'RUNDATE_DTIROI_04_30_14')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'STATUS_DTIROI_04_30_14')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'update_stamp_DTIROI_04_30_14')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'EXAMDATE_UPENNBIOMK9_04_19_17')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'update_stamp_UPENNBIOMK9_04_19_17')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'RUNDATE_UPENNBIOMK9_04_19_17')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'STDS_UPENNBIOMK9_04_19_17')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'KIT_UPENNBIOMK9_04_19_17')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'BATCH_UPENNBIOMK9_04_19_17')
    bl_data_MCI, header = remove_column(bl_data_MCI, header, 'PHASE_UPENNBIOMK9_04_19_17')
    bl_data_MCI, header= remove_column(bl_data_MCI, header, 'SUMMARYSUVR_WHOLECEREBNORM_1.11CUTOFF_UCBERKELEYAV45_10_17_16')
    bl_data_MCI, header= remove_column(bl_data_MCI, header, 'SUMMARYSUVR_COMPOSITE_REFNORM_0.79CUTOFF_UCBERKELEYAV45_10_17_16')

    bl_data_MCI, header = remove_column_too_many_missing_values(bl_data_MCI, header)"""
    return bl_data_MCI, header, gender_column, age_column, D1_column, D2_column, labels

def matrix_missing(matrix):
    """
    see how empty the matrix is
    input :
        matrix : np array
    output :
        % emptyness of the matrix
    """
    empty = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i][j] == '':
                empty +=1
    return float(empty)/(matrix.shape[0]*matrix.shape[1])


if __name__ == '__main__':
    bl_data_MCI, header, gender_column, age_column, D1_column, D2_column, labels = preprocessing()
    print(bl_data_MCI.shape)
    #save_results()
    emptiness = matrix_missing(bl_data_MCI)
    print('emptiness of the matrix', emptiness)

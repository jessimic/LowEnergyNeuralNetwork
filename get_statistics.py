##########################################
# Functions meant to determine statistics from datasets
# Optimizes lookup but flattening data array
# Most sets assume last index is variable to find statistics on
# Should adapt to any size array
# Meant to call for scaler_transformations.py functions
###########################################

import numpy as np
import h5py

def GetAQuartile(a_list):
    """Masks zeros and finds the first and third quartiles
    Input:
        a_list = flattened 1D list
    Output
        q1 = 25% quartile
        q3 = 75% quartfile
    """
    mask_zeros = np.logical_or(a_list>0,a_list<0)
    a_list_nozero = a_list[mask_zeros]
    q1, q3 = np.percentile(a_list_nozero,[25,75])

    return q1, q3

def GetQuartilesList(full_data_set):
    """Finds the 25 and 75% quartiles for multiple variables in array
    Assumes last index is each independent variable
    Input:
        full_data_set = N-D array of data with stat variable as last index
    Outputs:
        q1_list = list of the 25% quartile values for all variables in dataset
        q3_list = list of the 75% quartile values for all variables in dataset
    """
    q1_list = []
    q3_list = []
    for data_index in range(0,full_data_set.shape[-1]):

        data_list = full_data_set[...,data_index].flatten()
        q1, q3 = GetAQuartile(data_list)
        q1_list.append(q1)
        q3_list.append(q3)
    return q1_list, q3_list

def FindQuartilesPerDataset(X_test_DC_raw, X_test_IC_raw, X_validate_DC_raw, X_validate_IC_raw, X_train_DC_raw, X_train_IC_raw):
    """
       Finds 25% and 75% quartiles for multiple datasets, if the test, train, and validation are already split
       Return dictionary with list of results (key names: X_test_DC_q1, X_test_DC_q3, X_test_IC_q1, etc.) 
       Must input ALL datasets, no options for missing validation
       Input all input feature arrays to be transformed, named as:
            X_test_DC_raw= X_test_DC dataset
            X_test_IC_raw= X_test_IC dataset
            X_validate_DC_raw= X_validate_DC dataset
            X_validate_IC_raw= X_validate_IC dataset
            X_train_DC_raw = X_train_DC dataset
            X_train_IC_raw = X_train_IC dataset
       Returns:
            quartiles_dict = dictionary with 25% and 75% quartiles saved (can be a list)
    """
    quartiles_dict = {}
    transform_arrays = [X_test_DC_raw, X_test_IC_raw, X_validate_DC_raw, X_validate_IC_raw, X_train_DC_raw, X_train_IC_raw]
    name_arrays = ["X_test_DC", "X_test_IC", "X_validate_DC", "X_validate_IC","X_train_DC", "X_train_IC"]
    for an_array in range(0,len(name_arrays)):
        quartiles = GetQuartilesList(transform_arrays[an_array])
        quartiles_dict[name_arrays[an_array]+"_q1"] = quartiles[0]
        quartiles_dict[name_arrays[an_array]+"_q3"] = quartiles[1]
    
    print(quartiles_dict)
    return quartiles_dict

def GetMinMaxList(full_data_set):
    """
    Finds the min and max for multiple variables in an N-D dataset
    Assumes last index is each independent variable
    Input:
        full_data_set = N-D array of data with stat variable as last index
    Outputs:
        min_list = list of the minimum values for all variables in dataset
        max_list = list of the maximum values for all variables in dataset
    """
    min_list = []
    max_list = []
    for data_index in range(0,full_data_set.shape[-1]):

        data_list = full_data_set[...,data_index].flatten()
        min_val = np.min(data_list)
        max_val = np.max(data_list)
        min_list.append(min_val)
        max_list.append(max_val)
    return min_list, max_list

def GetMaxList(full_data_set):
    """ 
    Finds the max for multiple variables in an N-D dataset
    Assumes last index is each independent variable
    Input:
        full_data_set = N-D array of data with stat variable as last index
    Outputs:
        max_list = list of the maximum values for all variables in dataset
    """
    max_list = []
    for data_index in range(0,full_data_set.shape[-1]):

        data_list = full_data_set[...,data_index].flatten()
        max_val = np.max(data_list)
        max_list.append(max_val)
    return max_list

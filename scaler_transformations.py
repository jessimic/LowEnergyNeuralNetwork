#################################
# Functions to scale data for neural network
#   transform_charge = moves true 0 to -1, log transformation
#   transform_time = moves any value not in hit window to null hit value
#   transform_null = changes what is defined as the null hit (meant for timing)
#   new_transform = applies transform_charge, transform_time, transform_null to 5 variable dataset
#   RobustScaler = applies Robust Scaler shift to data  
#   MinMaxScaler = applies MinMax scaler shift to data
#   TransformData = applies chosen scaler to N-D dataset
#################################

import numpy as np

def transform_charge(a_list):
    """
    Transform charge values so that 0 --> -1, fractional --> 1, log transform (keep 0 at -1)
    Input:
        a_list = flattened, 1D list of sum of charge
    Returns:
        a_list = transformed so no hit is at -1 and the rest are log distributed
    """
    a_list = np.array(a_list)
    assert any(a_list) >= 0, "There are values less than zero! Are you sure this is charge input data?"
    mask_zero_charge = a_list==0
    mask_some_charge = a_list>0
    mask_fractional_charge = np.logical_and(a_list<1,a_list>0)
    
    #Change all fractional charge to 1, for simplicity
    #a_list[mask_fractional_charge] = 1
    
    #Move all zero charge to -1, since log(0) = -inf
    a_list[mask_zero_charge] = -4
    
    #Apply log transform to only > 1 things
    a_list[mask_some_charge] = np.log(a_list[mask_some_charge])
    
    return a_list

def time_bin_transform(full_data_set):
    """
    Apply specific charge transformations to time binned data (event, dom, string, time bin, charge)
    Apply transform_charge values so that 0 --> -1, fractional --> 1, log transform (keep 0 at -1)
    Input:
        full_data_set = N-D dataset with charge variable stored in last index
    Output:
        transformed_data_set = N-D dataset with all variables transformed
    """
    transformed_data_set = np.copy(full_data_set)    
    data_list = full_data_set[...,0].flatten()
    data_transformed = transform_charge(data_list)
    transformed_data_set[...,0] = data_transformed.reshape(full_data_set.shape[:-1])
        
    return transformed_data_set

def transform_time(a_list,low_window=-500,high_window=4000,null_hit_value=-1000):
    """
    Transform time so all hits not in window are moved to null hit value
    SHOULD BE DONE AT CREATE_SINGLE_TRAINING step (1)
    Can use to transform null hit value instead of transform_null
    Inputs:
        a_list = flattened, 1D array of pulse time hits
        low_window = lower bound for time window
        high_window = upper bound for time window
        null_hit_value = value that you want "no hit" to register as (avoid 0 since hits happen there)
    Outputs:
        a_list = transformed so all pulse times are in a window and the rest are at null_hit_value
    """
    a_list = np.array(a_list)
    assert null_hit_value<low_window, "Lower bound on window will not include null hits"
    mask_outside_window = np.logical_or(a_list<low_window,a_list>high_window)
    a_list[mask_outside_window] = null_hit_value
    
    return a_list

def transform_null(a_list,old_null=-20000,new_null=-1000):
    """
    Move the null hit value (assumed for pulse time variables)
    Input:
        a_list = flattened, 1D list
        old_null = original null hit value
        new_null = new null hit value
    Output:
        a_list = 1D list with null hit value changed
    """
    a_list = np.array(a_list)
    mask_null = a_list==old_null
    a_list[mask_null] = new_null
    
    return a_list


def new_transform(full_data_set):
    """
    Apply specific charge and pulse time transformations
    Apply transform_charge values so that 0 --> -1, log transform (keep 0 at -1)
    Apply transform_time so all hits not in window are moved to null hit value
    Apply transform_null to move the null hit values (for mean and standard deviation)
    Input:
        full_data_set = N-D dataset with variable stored in last index
    Output:
        transformed_data_set = N-D dataset with all variables transformed
    """
    transformed_data_set = np.copy(full_data_set)
    for variable_index in range(0,full_data_set.shape[-1]):
        
        data_list = full_data_set[...,variable_index].flatten()
        
        if variable_index == 0:
            #data_transformed = transform_charge(data_list)
            data_transformed = data_list
            print("currently not transforming charge")
        elif variable_index == 1 or variable_index == 2:
            data_transformed = transform_time(data_list)
            #print("doing time")
        elif variable_index == 3 or variable_index == 4:
            data_transformed = transform_null(data_list)
            #print("doing null change")
        else:
            #print("doing nothing")
            data_transformed = data_list
        
        transformed_data_set[...,variable_index] = data_transformed.reshape(full_data_set.shape[:-1])
        
    return transformed_data_set

def RobustScaler(a_list,q1,q3):
    """Robust Scaler calculation, uses the first quartile (q1) and third quartile (q3)"""
    return [(x-q1)/(q3-q1) for x in a_list]

def MinMaxScaler(a_list,min_val,max_val):
    """Robust Scaler calculation, uses the first quartile (q1) and third quartile (q3)"""
    return [(x-min_val)/(max_val-min_val) for x in a_list]

def TransformData(full_data_set,low_stats=None,high_stats=None,scaler="MaxAbs"):
    """
    Performs Robust, MinMax, or MaxAbs Scaler transformations
    Can find statistics of dataset (if you feed it whole dataset) or 
    use given values (if found earlier when dataset was whole)
    Inputs:
        full_data_set = the expected 4D data (training input data)
        low_stats = list or single value with either q1 or min values
        high_stats = list or single value with either q3 or max vavlues
        scaler = name of scaler to use, currently set up Robust and MaxAbs and MinMax
    Outputs:
        transformed_data_set = same dimensions as input, but with Robuset transformed output
    """
    transformed_data_set = np.copy(full_data_set)

    for data_index in range(0,full_data_set.shape[-1]):

        data_list = full_data_set[...,data_index].flatten()
        
        if scaler == "Robust":
            if type(low_stats) == None: #Find quartiles on this given dataset
                print("Not given q1, so finding q1 and q3 from this dataset")
                from get_statistics import GetAQuartile
                q1, q3 = GetAQuartile(data_list)
            else:
                if type(high_stats) == list or type(high_stats) == np.ndarray:
                    q1 = low_stats[data_index]
                    q3 = high_stats[data_index]
                else:
                    q1 = low_stats
                    q3 = high_stats
            data_scaled = RobustScaler(data_list,q1,q3)
        
        elif scaler == "MinMax":
            if low_stats is None: #Find quartiles on this given dataset
                print("Not given min, so finding min and max from this dataset")
                min_val = np.min(data_list)
                max_val = np.max(data_list)
            else:
                if type(high_stats) == list or type(high_stats) == np.ndarray:
                    min_val = low_stats[data_index]
                    max_val = high_stats[data_index]
                else:
                    min_val = low_stats
                    max_val = high_stats
            data_scaled = MinMaxScaler(data_list,min_val,max_val)
        
        elif scaler == "MaxAbs":
            if high_stats is None:
                print("Not given max values, so finding max from this dataset") 
                max_val = max(abs(data_list))
            else:
                if type(high_stats) == list or type(high_stats) == np.ndarray:
                    max_val = high_stats[data_index]
                else:
                    max_val = high_stats
            print("Scaling by %f"%max_val)
            data_scaled = data_list/float(max_val)
        
        else:
            print("I dont know what scaler to use. Try Robust, MinMax, or MaxAbs")
            break

        data_scaled = np.array(data_scaled)

        print("Working on index %i of %i"%(data_index,full_data_set.shape[-1]))
        transformed_data_set[...,data_index] = data_scaled.reshape(full_data_set.shape[:-1])

    return transformed_data_set


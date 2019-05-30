##########################################
# Python script to do Robust Scaler on concatted files
#  Takes in file name and path
#  Outputs file with transformed obserable feature values
#  Does not transform labels, leaves all labels
#  Can handle old reco (comparison to pegleg)
###########################################

import numpy
import h5py
import time
import os, sys
import random
from collections import OrderedDict
import itertools
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_files",type=str,default='Level5_IC86.2013_genie_nue.012640.100.cascade.lt60_vertexDC.hdf5',
                    dest="input_files", help="names for input files")
parser.add_argument("-d", "--path",type=str,default='/mnt/scratch/micall12/training_files/',
                    dest="path", help="path to input files")
parser.add_argument("-r","--reco",type=bool, default=False,
                    dest="reco", help="bool if the file has pegleg reco info + initial stats, etc. (if using Level5p files)")
args = parser.parse_args()
working_dir = args.path
input_file = working_dir + args.input_files
if args.reco == "True" or args.reco == "true":
    use_old_reco = True
else:
    use_old_reco = False

### Import Files ###
f = h5py.File(input_file, 'r')
features_DC = f['features_DC'][:]
features_IC = f['features_IC'][:]
labels = f['labels'][:]
if use_old_reco:
    reco = f['reco_labels'][:]
f.close()
del f

assert features_DC.shape[0]==features_IC.shape[0], "DC events not equal to IC events"
assert features_DC.shape[0]==labels.shape[0], "DC events not equatl to IC events"

### Split into training and testing set ###
num_train = int(features_DC.shape[0]*0.9) # 90% of data is training data (traininig+validation), 10% is test data
print("training on {} samples, testing on {} samples".format(num_train, features_DC.shape[0]-num_train))

features_DC_train = features_DC[:num_train]
features_IC_train = features_IC[:num_train]
labels_train = labels[:num_train]
if use_old_reco:
    reco_train = labels[:num_train]

features_DC_test = features_DC[num_train:]
features_IC_test = features_IC[num_train:]
labels_test = labels[num_train:]
if use_old_reco:
    reco_test = reco[num_train:]

### Specify type for training and testing ##
(X_train_DC_raw, X_train_IC_raw, Y_train_raw) = (features_DC_train, features_IC_train, labels_train)
X_train_DC_raw = X_train_DC_raw.astype("float32")
X_train_IC_raw = X_train_IC_raw.astype("float32")
Y_train_raw = Y_train_raw.astype("float32")
if use_old_reco:
    reco_train_raw = reco_train.astype("float32")

(X_test_DC_raw, X_test_IC_raw, Y_test_raw) = (features_DC_test, features_IC_test, labels_test)
X_test_DC_raw = X_test_DC_raw.astype("float32")
X_test_IC_raw = X_test_IC_raw.astype("float32")
Y_test_raw = Y_test_raw.astype("float32")
if use_old_reco:
    reco_test_raw = reco_test.astype("float32")

## ROBUST SCALER BY HAND ## ####
def RobustScaler(a_list,q1,q3):
    """Robust Scaler calculation, uses the first quartile (q1) and third quartile (q3)"""
    return [(x-q1)/(q3-q1) for x in a_list]

def GetQuartiles(a_list):
    """Masks zeros and finds the first and third quartiles"""
    mask_zeros = numpy.logical_or(a_list>0,a_list<0)
    a_list_nozero = a_list[mask_zeros]
    q1, q3 = numpy.percentile(a_list_nozero,[25,75])

    return q1, q3

def TransformData(full_data_set):
    """
    Performs Robust Scaler transformation
    Inputs:
        full_data_set = the expected 4D data (training input data)
    Outputs:
        transformed_data_set = same dimensions as input, but with Robuset transformed output
    """
    for data_index in range(0,full_data_set.shape[3]):

        data_list = full_data_set[:,:,:,data_index].flatten()
        
        #Find quartiles on THIS file
        q1, q3 = GetQuartiles(data_list)
        data_rb = RobustScaler(data_list,q1,q3)
        
        data_rb = numpy.array(data_rb)
        transformed_data_set = numpy.copy(full_data_set)

    return transformed_data_set

### Transform input features into specified range ###
X_train_DC_full = TransformData(X_train_DC_raw)
X_test_DC_full  = TransformData(X_test_DC_raw)
X_train_IC_full = TransformData(X_train_IC_raw)
X_test_IC_full  = TransformData(X_test_IC_raw)
print("Finished transforming the data using Robust Scaler")

Y_train = numpy.copy(Y_train_raw)
Y_test       = numpy.copy(Y_test_raw)
X_train_DC = numpy.copy(X_train_DC_full)
X_test_DC           = numpy.copy(X_test_DC_full)
X_train_IC = numpy.copy(X_train_IC_full)
X_test_IC           = numpy.copy(X_test_IC_full)
if use_old_reco:
    reco_train = numpy.copy(reco_train_raw)
    reco_test = numpy.copy(reco_test_raw)

#Save output to hdf5 file
output_file = input_file[:-4] + "transformed.hdf5"
f = h5py.File(output_file, "w")
f.create_dataset("Y_train", data=Y_train)
f.create_dataset("Y_test", data=Y_test)
f.create_dataset("X_train_DC", data=X_train_DC)
f.create_dataset("X_test_DC", data=X_test_DC)
f.create_dataset("X_train_IC", data=X_train_IC)
f.create_dataset("X_test_IC", data=X_test_IC)
if use_old_reco:
    f.create_dataset("reco_train",data=reco_train)
    f.create_dataset("reco_test",data=reco_test)
f.close()

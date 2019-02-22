#################################################################
# Playing with transformation scalers for training data
#
#   Takes arguments input file and output file indicator name
#
#   Contains functions:
#       MinMaxArray = performs maxmin scaler transformation
#       RobustArray = performs robust scaler transformation
#       SimpleHistPlot = fills in subplot with histogram
#       TransformData = Takes data in, strips zeros, transforms, plots
#################################################################

import numpy
import h5py
import time
import os, sys
import random
from collections import OrderedDict
import itertools
import matplotlib.pyplot as plt
import argparse

#There is a MacOS problem...workaround it
os.environ['KMP_DUPLICATE_LIB_OK']='True'

## Create ability to change settings from terminal ##
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input",type=str,default='Level5_IC86.2013_genie_numu.014640.10.hdf5',
                    dest="input_file", help="name of the input file")
parser.add_argument("--filename",type=str,default="10",
                    dest="filename", help="number of files used for input")

args = parser.parse_args()
input_file = args.input_file
filename = args.filename

#Create directory to save to, based on settings
save_folder_name = "CNN_test_files_%s/"%(filename)

### Import Files ###
f = h5py.File(input_file, 'r')
features_DC = f['features_DC'][:]
features_IC = f['features_IC'][:]
labels = f['labels'][:]
f.close()
del f

# ONLY USE ENERGY FOR NOW
labels = labels[:,0]

assert features_DC.shape[0]==features_IC.shape[0], "DC events not equal to IC events"
assert features_DC.shape[0]==labels.shape[0], "DC events not equatl to IC events"

### Split into training and testing set ###
num_train = int(features_DC.shape[0]*0.9) # 90% of data is training data (traininig+validation), 10% is test data
print("training on {} samples, testing on {} samples".format(num_train, features_DC.shape[0]-num_train))

features_DC_train = features_DC[:num_train]
features_IC_train = features_IC[:num_train]
labels_train = labels[:num_train]

features_DC_test = features_DC[num_train:]
features_IC_test = features_IC[num_train:]
labels_test = labels[num_train:]

### Specify type for training and testing ##
(X_train_DC_raw, X_train_IC_raw, Y_train_raw) = (features_DC_train, features_IC_train, labels_train)
X_train_DC_raw = X_train_DC_raw.astype("float32")
X_train_IC_raw = X_train_IC_raw.astype("float32")
Y_train_raw = Y_train_raw.astype("float32")

(X_test_DC_raw, X_test_IC_raw, Y_test_raw) = (features_DC_test, features_IC_test, labels_test)
X_test_DC_raw = X_test_DC_raw.astype("float32")
X_test_IC_raw = X_test_IC_raw.astype("float32")
Y_test_raw = Y_test_raw.astype("float32")

# Information: sum charges, time first pulse, Time of last pulse, Charge weighted mean time of pulses, Charge weighted standard deviation of pulse times

#Make Original Hist
def SimpleHistPlot(data,f,ax,title,ax1=0,ax2=0,log=False):
    """Create simple histograms in 4 subplots
    Inputs: data = 1D array or list
            f = plt figure
            ax = plt axis
            ax1 = first value in subplot position
            ax2 = second value in subplot position
            log = yscale log if True
    Output: creates histogram in subplot position
    """
    #plt.figure()
    ax[ax1, ax2].hist(data,bins=20,histtype='step',linewidth=2,fill=False)
    ax[ax1, ax2].set_title("%s "%title)
    if log==True:
        ax[ax1, ax2].set_yscale("log")

def RobustArray(a_list,q1=None,q3=None):
    """ Robust Scaler transform data
    Input:  a_list = a list or 1D array
            q1 = first quartile
            q3 = third quartile
    Output: list of transformed data
    """
    if q1==None or q3==None:
        q1, q3 = numpy.percentile(a_list,[25,75])
    return [(x-q1)/(q3-q1) for x in a_list]

#Max Min by hand
def MinMaxArray(a_list):
    """ Max-Min Scaler transform data
    Input:  a_list = a list or 1D array
    Output: list of transformed data
    """
    min_val = min(a_list)
    max_val = max(a_list)
    return [(x-min_val)/(max_val-min_val) for x in a_list]


def TransformData(full_data_set,plot=True,transform=False):
    name_plots = ["Charges", "First Pulse Time", "Last Pulse", "Charge weighted mean of time", "Charge weighted variance of time"]
    log_scale = [True, False, False, False, False]
    for data_index in range(0,full_data_set.shape[3]): ####
        
        data_list = full_data_set[:,:,:,data_index].flatten() ####
        mask_zeros = numpy.logical_or(data_list>0,data_list<0) ####
        data_list_nozero = data_list[mask_zeros] ####
        q1, q3 = numpy.percentile(data_list_nozero,[25,75]) ####
        print("Working on %s with %i events and %f fraction non-zero"%(name_plots[data_index],len(data_list),len(data_list_nozero)/float(len(data_list))))
        print(q1,q3)

        if len(data_list_nozero) == 0:
            print("No non-zero values in %s...skipping..."%name_plots[data_index])
            continue
        
        #data_minmax = MinMaxArray(data_list)
        data_rb = RobustArray(data_list,q1,q3) ####
        data_rb = numpy.array(data_rb)

        transformed_data_set = numpy.copy(full_data_set) ####
        if transform==True:
            transformed_data_set[:,:,:,0] = data_rb.reshape(full_data_set.shape[0],full_data_set.shape[1],full_data_set.shape[2]) ####

        #Make some plots
        if plot==True:
            transformed_data_list = transformed_data_set[:,:,:,data_index].flatten()
            f, ax = plt.subplots(2, 2,figsize=(10,10))
            f.suptitle("%s"%name_plots[data_index])
            SimpleHistPlot(data_list_nozero,f,ax,"Pre (removed zeros)",0,0,log=log_scale[data_index])
            SimpleHistPlot(data_list,f,ax,"Pre (with zeros)",0,1,log=True)
            #SimpleHistPlot(data_minmax, f,ax,"MinMax",1,0,log=log_scale[data_index])
            SimpleHistPlot(transformed_data_set[:,:,:,0].flatten(), f,ax,"Robust (with zeros)",1,1,log=True)
            SimpleHistPlot(transformed_data_set[:,:,:,0].flatten()[mask_zeros], f,ax,"Robust (no zeros)",1,0,log=log_scale[data_index])

    return transformed_data_set

### Transform input features into specified range ###
#transformer = RobustScaler().fit(X_train_DC_raw) # determine how to scale on the training set
X_train_DC_full = TransformData(X_train_DC_raw,plot=True,transform=False)
#X_test_DC_full  = TransformData(X_test_DC_raw,plot=False,transform=True)
#X_train_IC_full = TransformData(X_train_IC_raw,plot=False,transform=True)
#X_test_IC_full  = TransformData(X_test_IC_raw,plot=False,transform=True)

#Y_train_full = numpy.copy(Y_train_raw)
#Y_test       = numpy.copy(Y_test_raw)
#X_train_DC_selected = numpy.copy(X_train_DC_full)
#X_test_DC           = numpy.copy(X_test_DC_full)
#X_train_IC_selected = numpy.copy(X_train_IC_full)
#X_test_IC           = numpy.copy(X_test_IC_full)

plt.show()

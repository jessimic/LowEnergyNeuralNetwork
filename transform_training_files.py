##########################################
# Python script to do Transformations and Scaling
#  Takes in file name and path
#  Outputs file with transformed obserable feature values
#  Does not transform labels, leaves all labels
#  Can handle old reco (comparison to pegleg)
#  Inputs:
#   -i input_files: name of input file
#   -d path:        path for input and output files
#   -r reco:        True if there is a reco array in file
#   -v validate:    bool if output should have validation arrays
#   --scaler:       name of scaler to use (MaxAbs, MinMax, Robust)
#   --read_statistics:  True if incoming file has the stats included
#   --shuffle:      shuffles data
#   --trans_output: transforms energy and zenith output
#   --emax:         max energy (MaxAbs) to transform output energy
###########################################

import numpy
import h5py
import time
import os, sys
import random
from collections import OrderedDict
import itertools
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_file",type=str,default=None,
                    dest="input_file", help="name for input file (can only take one at a time)")
parser.add_argument("-d", "--path",type=str,default='/mnt/scratch/micall12/training_files/',
                    dest="path", help="path to input files")
parser.add_argument("-r","--reco",type=str, default=False,
                    dest="reco", help="bool if the file has pegleg reco info + initial stats, etc. (if using Level5p files)")
parser.add_argument("-v","--validate",type=str, default=True,
                    dest="validate", help="bool if the output file should already have validation separated")
parser.add_argument("--scaler",type=str, default='MaxAbs',
                    dest="scaler", help="name of transformation scaler type (Robust, MaxAbs, MinMax)")
parser.add_argument("--read_statistics",type=str, default=False,
                    dest="statistics", help="bool if the input file has the quartiles or minmax in it")
parser.add_argument("--shuffle",type=str, default=False,
                    dest="shuffle", help="True if you want to shuffle")
parser.add_argument("--trans_output",type=str, default=False,
                    dest="trans_output", help="True if you want to transform the energy and zenith")
parser.add_argument("--emax",type=float,default=None,
                    dest="emax", help="max energy to divide by for transforming output")
args = parser.parse_args()
working_dir = args.path
transform = args.scaler
max_energy = args.emax
if args.reco == "True" or args.reco == "true":
    use_old_reco = True
else:
    use_old_reco = False
if args.validate == "False" or args.validate == "false":
    create_validation = False
else:
    create_validation = True
if args.statistics == "True" or args.statistics == "true":
    read_statistics = True
else:
    read_statistics = False
    print("MAKING OWN QUARTILES or MINMAX! Not taking it from file")
if args.shuffle == "True" or args.shuffle == "true":
    shuffle = True
else:
    shuffle = False
if args.trans_output == "True" or args.trans_output == "true":
    transform_output = True
else:
    transform_output = False
if not max_energy:
    max_energy_print = "None"

print("Saving PEGLEG info: %s \nMake Validation set: %s \nScaler Transformation used: %s \nRead statistics from input file: %s \nShuffling: %s \nTransform output energy & zenith: %s \nMaximum Energy to divide output by: %s GeV"%(use_old_reco,create_validation,transform,read_statistics,shuffle,transform_output,max_energy))

### Import Files ###
input_file = working_dir + args.input_file
f = h5py.File(input_file, 'r')
features_DC = f['features_DC'][:]
features_IC = f['features_IC'][:]
labels = f['labels'][:]
if use_old_reco:
    reco = f['reco_labels'][:]
else: 
    reco = None
if read_statistics:
    low_stat_DC = f['low_stat_DC'][:]
    high_stat_DC = f['high_stat_DC'][:]
    low_stat_IC = f['low_stat_IC'][:]
    high_stat_IC = f['high_stat_IC'][:]
    low_stat_labels = f['low_stat_labels'][:]
    high_stat_labels = f['high_stat_labels'][:]
    if use_old_reco:
        low_stat_reco = f['low_stat_reco'][:]
        high_stat_reco = f['high_stat_reco'][:]
    print("USING STATISTICS FOUND IN FILE")
else:
    low_stat_DC      = None
    high_stat_DC     = None
    low_stat_IC      = None
    high_stat_IC     = None
    low_stat_labels  = None 
    high_stat_labels = None
f.close()
del f

if shuffle:
    from handle_data import Shuffler

    features_DC, features_IC, labels, \
    reco, initial_stats, num_pulses = \
    Shuffler(features_DC,features_IC,labels, \
    use_old_reco_flag=use_old_reco)

    print("SHUFFLED")

#Split data
from handle_data import SplitTrainTest
X_train_DC_raw, X_train_IC_raw, Y_train_raw, \
X_test_DC_raw, X_test_IC_raw, Y_test_raw, \
X_validate_DC_raw, X_validate_IC_raw, Y_validate_raw,\
reco_train_raw, reco_test_raw, reco_validate_raw  \
= SplitTrainTest(features_DC,features_IC,labels,\
reco=reco,use_old_reco=use_old_reco,create_validation=create_validation,\
fraction_test=0.1,fraction_validate=0.2)

#Transform Input Data
from scaler_transformations import TransformData, new_transform

X_train_DC_partial = new_transform(X_train_DC_raw) 
X_train_DC_full = TransformData(X_train_DC_partial, low_stats=low_stat_DC, high_stats=high_stat_DC, scaler=transform)
print("Finished train DC")

X_test_DC_partial = new_transform(X_test_DC_raw)
X_test_DC_full  = TransformData(X_test_DC_partial, low_stats=low_stat_DC, high_stats=high_stat_DC, scaler=transform)
print("Finished test DC")

X_train_IC_partial = new_transform(X_train_IC_raw)
X_train_IC_full = TransformData(X_train_IC_partial, low_stats=low_stat_IC, high_stats=high_stat_IC, scaler=transform)
print("Finished train IC")

X_test_IC_partial = new_transform(X_test_IC_raw)
X_test_IC_full  = TransformData(X_test_IC_partial, low_stats=low_stat_IC, high_stats=high_stat_IC, scaler=transform)
print("Finished test IC")

if create_validation:
    X_validate_DC_partial = new_transform(X_validate_DC_raw)
    X_validate_DC_full = TransformData(X_validate_DC_partial, low_stats=low_stat_DC, high_stats=high_stat_DC, scaler=transform)
    print("Finished validate DC")
    
    X_validate_IC_partial = new_transform(X_validate_IC_raw)
    X_validate_IC_full = TransformData(X_validate_IC_partial, low_stats=low_stat_IC, high_stats=high_stat_IC, scaler=transform)
    print("Finished validate IC")

print("Finished transforming the data using %s Scaler"%transform)

Y_train = numpy.copy(Y_train_raw)
Y_test       = numpy.copy(Y_test_raw)
X_train_DC = numpy.copy(X_train_DC_full)
X_test_DC           = numpy.copy(X_test_DC_full)
X_train_IC = numpy.copy(X_train_IC_full)
X_test_IC           = numpy.copy(X_test_IC_full)

if create_validation:
    Y_validate = numpy.copy(Y_validate_raw)
    X_validate_DC = numpy.copy(X_validate_DC_full)
    X_validate_IC = numpy.copy(X_validate_IC_full)

if use_old_reco:
    reco_train = numpy.copy(reco_train_raw)
    reco_test = numpy.copy(reco_test_raw)
    if create_validation:
        reco_validate = numpy.copy(reco_validate_raw)

# Transform Energy and Zenith Data
# MaxAbs on Energy
# Cos on Zenith
if transform_output:
    if not max_energy:
        print("Not given max energy, finding it from the Y_test in the given file!!!")
        max_energy = max(abs(Y_test[:,0]))

    print("Transforming the energy and zenith output. Dividing energy by %f and taking cosine of zenith"%max_energy)

    Y_train[:,0] = Y_train[:,0]/float(max_energy) #energy
    Y_train[:,1] = numpy.cos(Y_train[:,1]) #cos zenith

    Y_validate[:,0] = Y_validate[:,0]/float(max_energy) #energy
    Y_validate[:,1] = numpy.cos(Y_validate[:,1]) #cos zenith

    Y_test[:,0] = Y_test[:,0]/float(max_energy) #energy
    Y_test[:,1] = numpy.cos(Y_test[:,1]) #cos zenith

#Save output to hdf5 file
transform_name = "transformedinput"
if transform_output:
    transform_name = transform_name + "output"
output_file = input_file[:-4] + transform_name + ".hdf5"
print("Output file: %s"%output_file)
f = h5py.File(output_file, "w")
f.create_dataset("Y_train", data=Y_train)
f.create_dataset("Y_test", data=Y_test)
f.create_dataset("X_train_DC", data=X_train_DC)
f.create_dataset("X_test_DC", data=X_test_DC)
f.create_dataset("X_train_IC", data=X_train_IC)
f.create_dataset("X_test_IC", data=X_test_IC)
if create_validation:
    f.create_dataset("Y_validate", data=Y_validate)
    f.create_dataset("X_validate_IC", data=X_validate_IC)
    f.create_dataset("X_validate_DC", data=X_validate_DC)
if use_old_reco:
    f.create_dataset("reco_train",data=reco_train)
    f.create_dataset("reco_test",data=reco_test)
    if create_validation:
        f.create_dataset("reco_validate",data=reco_validate)
f.close()

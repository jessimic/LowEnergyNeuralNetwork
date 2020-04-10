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

import numpy as np
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
parser.add_argument("--tmax",type=float,default=None,
                    dest="tmax", help="max track length to divide by for transforming output")
parser.add_argument("--num_out",type=int,default=1,
                    dest="num_out",help="number of output files you want to split the output into")
parser.add_argument("-c", "--cuts", type=str, default="all",
                    dest="cuts", help="name of cuts applied (see name options below)")
# cut names: track, cascade, CC, NC, track CC, track NC, cascade CC, cascade NC 
args = parser.parse_args()
working_dir = args.path
num_outputs = args.num_out
assert num_outputs<100, "NEED TO CHANGE FILENAME TO ACCOMODATE THIS MANY NUMBERS"
transform = args.scaler
max_energy = args.emax
max_track = args.tmax
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

print("Saving PEGLEG info: %s \nMake Validation set: %s \nScaler Transformation used: %s \nRead statistics from input file: %s \nShuffling: %s \nTransform output energy, zenith, and track: %s \nMaximum Energy to divide output by: %s GeV \nMaximum Track Length to divide output by: %s m"%(use_old_reco,create_validation,transform,read_statistics,shuffle,transform_output,max_energy,max_track))

static_stats = [25., 4000., 4000., 4000., 2000.] 
if read_statistics:
    print("Using static values to transform training input variables.")
    print("Diviving [sum of charge, time of first pulse, time of last pulse, charge weighted mean, charge weighted standard deviations] by", static_stats)

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
    low_stat_DC = static_stats 
    high_stat_DC = static_stats 
    low_stat_IC = static_stats 
    high_stat_IC = static_stats 

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

#Shuffle Option
if shuffle:
    from handle_data import Shuffler

    features_DC, features_IC, labels, \
    reco, initial_stats, num_pulses = \
    Shuffler(features_DC,features_IC,labels, \
    use_old_reco_flag=use_old_reco)

    print("Finished shuffling...")

#Cut Option
if cut != "all" or emax < max(labels[0]):
    from handle_data import CutMask
    assert labels.shape[-1] == 12, "output labels not expected and CutMask could be cutting the wrong thing"

    mask = CutMask(labels)
    e_mask = np.array(labels[:,0])<emax
    keep_index = np.logical_and(mask[cut_name],e_mask)
    number_events = sum(keep_index)

    features_DC = np.array(features_DC)[keep_index]
    features_IC = np.array(features_IC)[keep_index]
    labels = np.array(labels)[keep_index]
    reco = np.array(reco)[keep_index]

    print("Keeping %i events"%(number_events))
    print(features_DC.shape)

#Transform Input Data
from scaler_transformations import TransformData, new_transform

features_DC_partial_transform = new_transform(features_DC)
features_DC_full_transform = TransformData(features_DC_partial_transform, low_stats=low_stat_DC, high_stats=high_stat_DC, scaler=transform)
print("Finished DC")

features_IC_partial_transform = new_transform(features_IC)
features_IC_full_transform = TransformData(features_IC_partial_transform, low_stats=low_stat_IC, high_stats=high_stat_IC, scaler=transform)
print("Finished IC")

print("Finished transforming the data using %s Scaler"%transform)

# Transform Energy and Zenith Data
# MaxAbs on Energy
# Cos on Zenith
if transform_output:
    if not max_energy:
        print("Not given max energy, finding it from the Y_test in the given file!!!")
        max_energy = max(abs(labels[:,0]))
    if not max_track:
        print("Not given max track, finding it from the Y_test in the given file!!!")
        max_track = max(labels[:,7])

    labels_full = np.copy(labels)

    labels_full[:,0] = labels[:,0]/float(max_energy) #energy
    labels_full[:,1] = np.cos(labels[:,1]) #cos zenith
    labels_full[:,2] = labels[:,7]/float(max_track) #MAKE TRACK THIRD INPUT
    labels_full[:,7] = labels[:,2] #MOVE AZIMUTH TO WHERE TRACK WAS

    print("Transforming the energy and zenith output. Dividing energy by %f and taking cosine of zenith"%max_energy)
    print("Transforming track output. Dividing track by %f and MOVING IT TO INDEX 2 IN ARRAY. AZIMUTH NOW AT 7"%max_track)
else:
    labels_full = np.array(labels)

#Split data
from handle_data import SplitTrainTest
X_train_DC, X_train_IC, Y_train, \
X_test_DC, X_test_IC, Y_test, \
X_validate_DC, X_validate_IC, Y_validate,\
reco_train, reco_test, reco_validate  \
= SplitTrainTest(features_DC_full_transform,features_IC_full_transform,labels_full,\
reco=reco,use_old_reco=use_old_reco,create_validation=create_validation,\
fraction_test=0.1,fraction_validate=0.2)


#Save output to hdf5 file
cut_name_nospaces = cut_name.replace(" ","")
cut_file_name = cut_name_nospaces + ".lt" + str(int(emax)) + '.'
transform_name = "transformedinput"
if read_statistics:
    transform_name = transform_name + "static"
if transform_output:
    transform_name = transform_name + "_transformed3output"

events_per_file = int(X_train_DC.shape[0]/num_outputs) + 1
print("Saving %i events per %i file(s)"%(events_per_file,num_outputs))
for sep_file in range(0,num_outputs):
    output_file = input_file[:-4] + cut_file_name + transform_name + "_file%02d.hdf5"%sep_file
    print("Output file: %s"%output_file)

    start = events_per_file*sep_file
    if sep_file < num_outputs-1:
        end = events_per_file*(sep_file+1)
    else:
        end = X_train_DC.shape[0] 

    f = h5py.File(output_file, "w")
    f.create_dataset("Y_train", data=Y_train[start:end])
    f.create_dataset("Y_test", data=Y_test[start:end])
    f.create_dataset("X_train_DC", data=X_train_DC[start:end])
    f.create_dataset("X_test_DC", data=X_test_DC[start:end])
    f.create_dataset("X_train_IC", data=X_train_IC[start:end])
    f.create_dataset("X_test_IC", data=X_test_IC[start:end])
    if create_validation:
        f.create_dataset("Y_validate", data=Y_validate[start:end])
        f.create_dataset("X_validate_IC", data=X_validate_IC[start:end])
        f.create_dataset("X_validate_DC", data=X_validate_DC[start:end])
    if use_old_reco:
        f.create_dataset("reco_train",data=reco_train[start:end])
        f.create_dataset("reco_test",data=reco_test[start:end])
        if create_validation:
            f.create_dataset("reco_validate",data=reco_validate[start:end])
    f.close()

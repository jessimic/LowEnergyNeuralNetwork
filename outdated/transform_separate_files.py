#################################
# Concatonates, cuts, and shuffles hdf5 training data sets
#   Inputs:
#       -i input files: name of file (can use * and ?)
#       -d  path:       path to input files
#       -o  output:      name of output file, placed in path directory
#       -c  cuts:       name of cuts you want to apply (i.e. track only = track)
#       -r  reco:       bool if files have pegleg reco in them
#       --num_out:      number of output files to split output into (default = 1, i.e. no split)
#       --emax:         Energy cut, keep all events below value
#       --shuffle       True if you want to shuffle order of events, default is true
#       --trans_output: transforms energy and zenith output
#       -v validate:    bool if output should have validation arrays
#       --scaler:       name of scaler to use (MaxAbs, MinMax, Robust)
#################################

import numpy as np
import glob
import h5py
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_file",default=None,
                    type=str,dest="input_file", help="name for ONE input file")
parser.add_argument("-d", "--path",type=str,default='/mnt/scratch/micall12/training_files/',
                    dest="path", help="path to input files")
parser.add_argument("-o", "--output",type=str,default='cut_concat_separated',
                    dest="output", help="names for output files")
parser.add_argument("-c", "--cuts", type=str, default="all",
                    dest="cuts", help="name of cuts applied (see name options on line 38)")
# cut names: track, cascade, CC, NC, track CC, track NC, cascade CC, cascade NC 
parser.add_argument("-r","--reco",type=str, default=False,
                    dest="reco", help="bool if the file has pegleg reco info + initial stats, etc. (if using Level5p files)")
parser.add_argument("--num_out",type=int,default=1,
                    dest="num_out",help="number of output files you want to split the output into")
parser.add_argument("--emax",type=float,default=200.0,
                    dest="emax",help="Max energy to keep, cut anything above")
parser.add_argument("--shuffle",type=str, default=True,
                    dest="shuffle", help="False if you don't want to shuffle")
parser.add_argument("--find_statistics",type=str, default=False,
                    dest="statistics", help="bool if you want to find the minmax in file instead of using static")
parser.add_argument("--trans_output",type=str, default=False,
                    dest="trans_output", help="True if you want to transform the energy and zenith")
parser.add_argument("-v","--validate",type=str, default=True,
                    dest="validate", help="bool if the output file should already have validation separated")
parser.add_argument("--scaler",type=str, default='MaxAbs',
                    dest="scaler", help="name of transformation scaler type (Robust, MaxAbs, MinMax)")
parser.add_argument("--tmax",type=float,default=None,
                    dest="tmax", help="max track length to divide by for transforming output")
args = parser.parse_args()

input_file = args.input_file
path = args.path
output = args.output
num_outputs = args.num_out
assert num_outputs<100, "NEED TO CHANGE OUTPUT FILENAME TO ACCOMODATE THIS MANY NUMBERS"

transform = args.scaler
max_energy = args.emax
max_track = args.tmax
cut = args.cuts
find_statistics = args.statistics

if args.reco == "True" or args.reco == "true":
    use_old_reco = True
    print("Expecting old reco values in file, from pegleg, etc.")
else:    
    use_old_reco = False
if args.shuffle == "False" or args.shuffle == "false":
    shuffle = False
else:    
    shuffle = True
if args.validate == "False" or args.validate == "false":
    create_validation = False
else:
    create_validation = True

static_stats = [25., 4000., 4000., 4000., 2000.]
if find_statistics:
    low_stat_DC      = None
    high_stat_DC     = None
    low_stat_IC      = None
    high_stat_IC     = None
else:
    low_stat_DC = static_stats
    high_stat_DC = static_stats
    low_stat_IC = static_stats
    high_stat_IC = static_stats
if args.trans_output == "True" or args.trans_output == "true":
    transform_output = True
else:
    transform_output = False

print("Saving PEGLEG info: %s \nNumber output files: %i \nShuffling: %s \n"%(use_old_reco,num_outputs,shuffle)) 
if not find_statistics:
    print("Using static values to transform training input variables.")
    print("Diviving [sum of charge, time of first pulse, time of last pulse, charge weighted mean, charge weighted standard deviations] by", static_stats)

file_name = path + input_file
assert file_name,"No file loaded, please check path."

old_reco = None
initial_stats = None
num_pulses = None

# Labels: [ nu energy, nu zenith, nu azimuth, nu time, nu x, nu y, nu z, track length (0 for cascade), isTrack (track = 1, cascasde = 0), flavor, type (anti = 1), isCC (CC=1, NC = 0)]

f = h5py.File(file_name, "r")
features_DC = f["features_DC"][:]
features_IC = f["features_IC"][:]
labels = f["labels"][:]
if use_old_reco:
    old_reco = f["reco_labels"][:]
f.close()
del f

print("Transforming %i events from %s"%(labels.shape[0],file_name))
 
#Cut Option
if cut != "all" or max_energy < max(labels[0]):
    print("Applying event type cut or energy cut...")
    from handle_data import CutMask
    assert labels.shape[-1] == 12, "output labels not expected and CutMask could be cutting the wrong thing"

    mask = CutMask(labels)
    e_mask = np.array(labels[:,0])<max_energy
    keep_index = np.logical_and(mask[cut],e_mask)
    number_events = sum(keep_index)

    features_DC = np.array(features_DC)[keep_index]
    features_IC = np.array(features_IC)[keep_index]
    labels = np.array(labels)[keep_index]
    if use_old_reco:
        old_reco = np.array(old_reco)[keep_index]

    print("Keeping %i events"%(number_events))
    print(features_DC.shape)

#Shuffle Option
if shuffle:
    print("Starting shuffle...")
    from handle_data import Shuffler

    features_DC, features_IC, labels, \
    old_reco, initial_stats, num_pulses = \
    Shuffler(features_DC,features_IC,labels, \
    old_reco, initial_stats, num_pulses, use_old_reco_flag=use_old_reco)

    print("Finished shuffling...")

#Transform Input Data
print("Starting transformation of input features...")
from scaler_transformations import TransformData, new_transform

features_DC_partial_transform = new_transform(features_DC)
del features_DC
features_DC_full_transform = TransformData(features_DC_partial_transform, low_stats=low_stat_DC, high_stats=high_stat_DC, scaler=transform)
del features_DC_partial_transform
print("Finished DC")

features_IC_partial_transform = new_transform(features_IC)
del features_IC 
features_IC_full_transform = TransformData(features_IC_partial_transform, low_stats=low_stat_IC, high_stats=high_stat_IC, scaler=transform)
del features_IC_partial_transform  
print("Finished IC")

print("Finished transforming the data using %s Scaler"%transform)

# Transform Energy and Zenith Data
# MaxAbs on Energy
# Cos on Zenith
if transform_output:
    print("Starting transformation of output features...")
    if not max_energy:
        print("Not given max energy, finding it from the Y_test in the given file!!!")
        max_energy = max(abs(labels[:,0]))
    if not max_track:
        print("Not given max track, finding it from the Y_test in the given file!!!")
        max_track = max(labels[:,7])

    labels_transform = np.copy(labels)

    labels_transform[:,0] = labels[:,0]/float(max_energy) #energy
    labels_transform[:,1] = np.cos(labels[:,1]) #cos zenith
    labels_transform[:,2] = labels[:,7]/float(max_track) #MAKE TRACK THIRD INPUT
    labels_transform[:,7] = labels[:,2] #MOVE AZIMUTH TO WHERE TRACK WAS

    print("Transforming the energy and zenith output. Dividing energy by %f and taking cosine of zenith"%max_energy)
    print("Transforming track output. Dividing track by %f and MOVING IT TO INDEX 2 IN ARRAY. AZIMUTH NOW AT 7"%max_track)
else:
    labels_transform = np.array(labels)
del labels

#Split data
from handle_data import SplitTrainTest
X_train_DC, X_train_IC, Y_train, \
X_test_DC, X_test_IC, Y_test, \
X_validate_DC, X_validate_IC, Y_validate,\
reco_train, reco_test, reco_validate  \
= SplitTrainTest(features_DC_full_transform,features_IC_full_transform,labels_transform,\
reco=old_reco,use_old_reco=use_old_reco,create_validation=create_validation,\
fraction_test=0.1,fraction_validate=0.2)

print("Total events saved: %i"%features_IC_full_transform.shape[0])
del features_DC_full_transform,features_IC_full_transform,labels_transform

#Save output to hdf5 file
cut_name_nospaces = cut.replace(" ","")
cut_file_name = cut_name_nospaces + ".lt" + str(int(max_energy)) + '.'
transform_name = "transformedinput"
if not find_statistics:
    transform_name = transform_name + "static"
if transform_output:
    transform_name = transform_name + "_transformed3output"

train_per_file = int(X_train_DC.shape[0]/num_outputs) + 1
test_per_file = int(X_test_DC.shape[0]/num_outputs) + 1
validate_per_file = int(X_validate_DC.shape[0]/num_outputs) + 1
print("Saving %i train events, %i test events, and %i validate events per %i file(s)"%(train_per_file,test_per_file,validate_per_file,num_outputs))
for sep_file in range(0,num_outputs):
    if num_outputs > 1:
        filenum = "_file%02d"%sep_file
    else:
        filenum = ""
    output_file = path + output + cut_file_name + transform_name + filenum + ".hdf5"

    train_start = train_per_file*sep_file
    test_start = test_per_file*sep_file
    validate_start = validate_per_file*sep_file
    if sep_file < num_outputs-1:
        train_end = train_per_file*(sep_file+1)
        test_end = test_per_file*(sep_file+1)
        validate_end = validate_per_file*(sep_file+1)
    else:
        train_end = X_train_DC.shape[0] 
        test_end = X_test_DC.shape[0] 
        validate_end = X_validate_DC.shape[0] 

    print("I put %i - %i train events, %i - %i test events, and %i - %i validate events into %s"%(train_start,train_end,test_start,test_end,validate_start,validate_end,output_file))
    f = h5py.File(output_file, "w")
    f.create_dataset("Y_train", data=Y_train[train_start:train_end])
    f.create_dataset("Y_test", data=Y_test[test_start:test_end])
    f.create_dataset("X_train_DC", data=X_train_DC[train_start:train_end])
    f.create_dataset("X_test_DC", data=X_test_DC[test_start:test_end])
    f.create_dataset("X_train_IC", data=X_train_IC[train_start:train_end])
    f.create_dataset("X_test_IC", data=X_test_IC[test_start:test_end])
    if create_validation:
        f.create_dataset("Y_validate", data=Y_validate[validate_start:validate_end])
        f.create_dataset("X_validate_IC", data=X_validate_IC[validate_start:validate_end])
        f.create_dataset("X_validate_DC", data=X_validate_DC[validate_start:validate_end])
    if use_old_reco:
        f.create_dataset("reco_train",data=reco_train[train_start:train_end])
        f.create_dataset("reco_test",data=reco_test[test_start:test_end])
        if create_validation:
            f.create_dataset("reco_validate",data=reco_validate[validate_start:validate_end])
    f.close()
                   

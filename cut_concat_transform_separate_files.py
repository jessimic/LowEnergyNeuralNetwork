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
parser.add_argument("-i", "--input_files",default=None,
                    type=str,dest="input_files", help="names for input files")
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
input_files = args.input_files
path = args.path
output = args.output
num_outputs = args.num_out
assert num_outputs<100, "NEED TO CHANGE FILENAME TO ACCOMODATE THIS MANY NUMBERS"
transform = args.scaler
cut_name = args.cuts
emax = args.emax
max_energy = emax
max_track = args.tmax
max_track
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

print("Keeping %s event types"%cut_name)
print("Saving PEGLEG info: %s \nNumber output files: %i \nEnergy Max: %f GeV \nShuffling: %s \nKeeping event types: %s"%(use_old_reco,num_outputs,emax,shuffle,cut_name)) 
if not find_statistics:
    print("Using static values to transform training input variables.")
    print("Diviving [sum of charge, time of first pulse, time of last pulse, charge weighted mean, charge weighted standard deviations] by", static_stats)

file_names = path + input_files
event_file_names = sorted(glob.glob(file_names))

#event_file_names = ["/mnt/scratch/micall12/training_files/NuMu_140000_level2.zst_cleaned_lt100_CC_flat_95bins_36034evtperbinall.lt100_file00.hdf5",\
#                    "/mnt/scratch/micall12/training_files/NuMu_140000_level2.zst_cleaned_lt100_CC_flat_95bins_36034evtperbinall.lt100_file01.hdf5",\
#                    "/mnt/scratch/micall12/training_files/NuMu_140000_level2.zst_cleaned_lt100_CC_flat_95bins_36034evtperbinall.lt100_file02.hdf5",\
#                    "/mnt/scratch/micall12/training_files/NuE_120000_level2_cleaned_lt100_vertexDC_CC_flat_95bins_15478evtperbinall.allfiles.CC.lt100_file00.hdf5"]
#print("I AM USING HARDCODED FILENAMES, IGNORING YOUR INPUT ARG!!!!!!")
assert event_file_names,"No files loaded, please check path."

full_features_DC = None
full_features_IC = None
full_labels = None
full_reco = None
full_initial_stats = None
full_num_pulses = None

# Labels: [ nu energy, nu zenith, nu azimuth, nu time, nu x, nu y, nu z, track length (0 for cascade), isTrack (track = 1, cascasde = 0), flavor, type (anti = 1), isCC (CC=1, NC = 0)]

for a_file in event_file_names:

    f = h5py.File(a_file, "r")
    file_features_DC = f["features_DC"][:]
    file_features_IC = f["features_IC"][:]
    file_labels = f["labels"][:]
    if use_old_reco:
        file_reco = f["reco_labels"][:]
        file_initial_stats = f["initial_stats"][:]
        file_num_pulses = f["num_pulses_per_dom"][:]
    f.close()
    del f

    if file_labels.shape[-1] != 12:
        print("Skipping file %s, output labels not expected and CutMask could be cutting the wrong thing"%a_file)
        continue
   
    from handle_data import CutMask
    mask = CutMask(file_labels)
    e_mask = np.array(file_labels[:,0])<emax
    keep_index = np.logical_and(mask[cut_name],e_mask)
    number_events = sum(keep_index)

    if full_features_DC is None:
        full_features_DC = file_features_DC[keep_index]
    else:
        full_features_DC = np.concatenate((full_features_DC, file_features_DC[keep_index]))
    
    if full_features_IC is None:
        full_features_IC = file_features_IC[keep_index]
    else:
        full_features_IC = np.concatenate((full_features_IC, file_features_IC[keep_index]))

    if full_labels is None:
        full_labels = file_labels[keep_index]
    else:
        full_labels = np.concatenate((full_labels, file_labels[keep_index]))

    if use_old_reco:
        if full_reco is None:
            full_reco = file_reco[keep_index]
        else:
            full_reco = np.concatenate((full_reco, file_reco[keep_index]))

        if full_initial_stats is None:
            full_initial_stats = file_initial_stats[keep_index]
        else:
            full_initial_stats  = np.concatenate((full_initial_stats , file_initial_stats[keep_index]))

        if full_num_pulses is None:
            full_num_pulses = file_num_pulses[keep_index]
        else:
            full_num_pulses = np.concatenate((full_num_pulses, file_num_pulses[keep_index]))
        

    print("Events this file: %i, Saved this file: %i, Cumulative saved: %i\n Finsihed file: %s"%(number_events,np.count_nonzero(keep_index),full_labels.shape[0],a_file))

if shuffle:
    print("Finished concatonating all the files. Now I will shuffle..")
    from handle_data import Shuffler

    shuffled_features_DC, shuffled_features_IC, shuffled_labels, \
    shuffled_reco, shuffled_initial_stats, shuffled_num_pulses = \
    Shuffler(full_features_DC,full_features_IC,full_labels, \
    full_reco, full_initial_stats,full_num_pulses,use_old_reco_flag=use_old_reco)
else: 
    shuffled_features_DC, shuffled_features_IC, shuffled_labels, \
    shuffled_reco, shuffled_initial_stats, shuffled_num_pulses = \
    full_features_DC,full_features_IC,full_labels, \
    full_reco, full_initial_stats,full_num_pulses

#Transform Input Data
from scaler_transformations import TransformData, new_transform

features_DC_partial_transform = new_transform(shuffled_features_DC)
features_DC_full_transform = TransformData(features_DC_partial_transform, low_stats=low_stat_DC, high_stats=high_stat_DC, scaler=transform)
print("Finished DC")

features_IC_partial_transform = new_transform(shuffled_features_IC)
features_IC_full_transform = TransformData(features_IC_partial_transform, low_stats=low_stat_IC, high_stats=high_stat_IC, scaler=transform)
print("Finished IC")

print("Finished transforming the data using %s Scaler"%transform)

# Transform Energy and Zenith Data
# MaxAbs on Energy
# Cos on Zenith
if transform_output:
    if not max_energy:
        print("Not given max energy, finding it from the Y_test in the given file!!!")
        max_energy = max(abs(shuffled_labels[:,0]))
    if not max_track:
        print("Not given max track, finding it from the Y_test in the given file!!!")
        max_track = max(shuffled_labels[:,7])

    labels_transform = np.copy(shuffled_labels)

    labels_transform[:,0] = shuffled_labels[:,0]/float(max_energy) #energy
    labels_transform[:,1] = np.cos(shuffled_labels[:,1]) #cos zenith
    labels_transform[:,2] = shuffled_labels[:,7]/float(max_track) #MAKE TRACK THIRD INPUT
    labels_transform[:,7] = shuffled_labels[:,2] #MOVE AZIMUTH TO WHERE TRACK WAS

    print("Transforming the energy and zenith output. Dividing energy by %f and taking cosine of zenith"%max_energy)
    print("Transforming track output. Dividing track by %f and MOVING IT TO INDEX 2 IN ARRAY. AZIMUTH NOW AT 7"%max_track)
else:
    labels_transform = np.array(shuffled_labels)

#Split data
from handle_data import SplitTrainTest
X_train_DC, X_train_IC, Y_train, \
X_test_DC, X_test_IC, Y_test, \
X_validate_DC, X_validate_IC, Y_validate,\
reco_train, reco_test, reco_validate  \
= SplitTrainTest(features_DC_full_transform,features_IC_full_transform,labels_transform,\
reco=shuffled_reco,use_old_reco=use_old_reco,create_validation=create_validation,\
fraction_test=0.1,fraction_validate=0.2)

#Save output to hdf5 file
print("Total events saved: %i"%full_features_DC.shape[0])

#Save output to hdf5 file
cut_name_nospaces = cut_name.replace(" ","")
cut_file_name = cut_name_nospaces + ".lt" + str(int(emax)) + '.'
transform_name = "transformedinput"
if not find_statistics:
    transform_name = transform_name + "static"
if transform_output:
    transform_name = transform_name + "_transformed3output"

events_per_file = int(full_features_DC_transform.shape[0]/num_outputs) + 1
print("Saving %i events per %i file(s)"%(events_per_file,num_outputs))
for sep_file in range(0,num_outputs):
    output_file = output + cut_file_name + transform_name + "_file%02d.hdf5"%sep_file

    start = events_per_file*sep_file
    if sep_file < num_outputs-1:
        end = events_per_file*(sep_file+1)
    else:
        end = X_train_DC.shape[0] 

    print("I put evnts %i - %i into %s"%(start,end,output_file))
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
                   

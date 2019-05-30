#################################
# Concatonates hdf5 training data sets
#   Inputs:
#       -i input files: name of file (can use * and ?)
#       -d path: path to input files
#       -o ouput: name of output file, placed in path directory
#
#################################

import numpy as np
import glob
import h5py
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_files",type=str,default='Level5_IC86.2013_genie_numu.014640.*.hdf5',
                    dest="input_files", help="names for input files")
parser.add_argument("-d", "--path",type=str,default='/mnt/scratch/micall12/training_files/',
                    dest="path", help="path to input files")
parser.add_argument("-o", "--output",type=str,default='Level5_IC86.2013_genie_numu.014640.all',
                    dest="output", help="names for output files")
parser.add_argument("--reco", type=bool,default=False,
                    dest="reco",help="do you have the reco variables and initial stats?")
args = parser.parse_args()
input_files = args.input_files
path = args.path
output = args.output

file_names = path + input_files
event_file_names = sorted(glob.glob(file_names))
assert event_file_names,"No files loaded, please check path."

full_features_DC = None
full_features_IC = None
full_labels = None
full_reco = None
full_initial_stats = None
full_num_pulses = None
for a_file in event_file_names:
    print("Pulling data from %s"%a_file)

    f = h5py.File(a_file, "r")
    file_features_DC = f["features_DC"][:]
    file_features_IC = f["features_IC"][:]
    file_labels = f["labels"][:]
    if args.reco:
        file_reco = f["reco_labels"][:]
        file_initial_stats = f["initial_stats"][:]
        file_num_pulses = f["num_pulses_per_dom"][:]
    f.close()
    del f

    
    #print(file_features_IC.shape)
    if full_features_DC is None:
        full_features_DC = file_features_DC
    else:
        full_features_DC = np.concatenate((full_features_DC, file_features_DC))

    if full_features_IC is None:
        full_features_IC = file_features_IC
    else:
        full_features_IC = np.concatenate((full_features_IC, file_features_IC))

    if full_labels is None:
        full_labels = file_labels
    else:
        full_labels = np.concatenate((full_labels, file_labels))

    if args.reco:
        if full_reco is None:
            full_reco = file_reco
        else:
            full_reco = np.concatenate((full_reco, file_reco))

        if full_initial_stats is None:
            full_initial_stats = file_initial_stats 
        else:
            full_initial_stats  = np.concatenate((full_initial_stats , file_initial_stats ))

        if full_num_pulses is None:
            full_num_pulses = file_num_pulses
        else:
            full_num_pulses = np.concatenate((full_num_pulses, file_num_pulses))

#Save output to hdf5 file
print(full_features_DC.shape)
print(full_features_IC.shape)
print(full_labels.shape)
output_name = path + output + ".hdf5" 
print(output_name)
f = h5py.File(output_name, "w")
f.create_dataset("features_DC", data=full_features_DC)
f.create_dataset("features_IC", data=full_features_IC)
f.create_dataset("labels", data=full_labels)
if args.reco:
    f.create_dataset("reco_labels",data=full_reco)
    f.create_dataset("initial_stats",data=full_initial_stats)
    f.create_dataset("num_pulses_per_dom",data=full_num_pulses)
f.close()

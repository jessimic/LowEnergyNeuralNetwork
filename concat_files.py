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
for a_file in event_file_names:
    print("Pulling data from %s"%a_file)

    f = h5py.File(a_file, "r")
    file_features_DC = f["features_DC"][:]
    file_features_IC = f["features_IC"][:]
    file_labels = f["labels"][:]
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
f.close()

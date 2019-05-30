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
parser.add_argument("-i", "--input_files",default='Level5_IC86.2013_genie_numu.014640.*.hdf5',
                    type=str,dest="input_files", help="names for input files")
parser.add_argument("-d", "--path",type=str,default='/mnt/scratch/micall12/training_files/',
                    dest="path", help="path to input files")
parser.add_argument("-o", "--output",type=str,default='Level5_IC86.2013_genie_numu.014640.energy_sorted',
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

count_energy = np.zeros((60))
max_per_bin = 1600
for a_file in event_file_names:

    f = h5py.File(a_file, "r")
    file_features_DC = f["features_DC"][:]
    file_features_IC = f["features_IC"][:]
    file_labels = f["labels"][:]
    f.close()
    del f

    energy = file_labels[:,0]
    keep_index = [False]*len(energy)
    index = 0

    # Check how many events already in each bin, save if under max
    for e in energy:
        e_bin = int(e)
        if count_energy[e_bin] < max_per_bin:
            keep_index[index] = True
            count_energy[e_bin] += 1
        index += 1

    assert len(file_features_DC.shape) == 4, "Features shape is wrong, code can't handle this"

    for event_index in range(0,index):

        if keep_index[event_index]:
            if full_features_DC is None:
                full_features_DC = file_features_DC[event_index]
            else:
                full_features_DC = np.concatenate((full_features_DC, file_features_DC[event_index]))
            
            if full_features_IC is None:
                full_features_IC = file_features_IC[event_index]
            else:
                full_features_IC = np.concatenate((full_features_IC, file_features_IC[event_index]))

            if full_labels is None:
                full_labels = file_labels[keep_index,:]
            else:
                full_labels = np.concatenate((full_labels, file_labels[event_index]))

    print("Total events saved: %i, Saved this file: %i, Finsihed file: %s"%(sum(count_energy),keep_index.count(True),a_file))
    
    if np.all(count_energy == max_per_bin-1):
        break

#Save output to hdf5 file
print(count_energy)
output_name = path + output + ".hdf5" 
print(output_name)
f = h5py.File(output_name, "w")
f.create_dataset("features_DC", data=full_features_DC)
f.create_dataset("features_IC", data=full_features_IC)
f.create_dataset("labels", data=full_labels)
f.close()

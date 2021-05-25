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

full_Y_test = None
full_X_test_DC = None
full_X_test_IC = None
full_Y_train = None
full_X_train_DC = None
full_X_train_IC = None
full_Y_val = None
full_X_val_DC = None
full_X_val_IC = None
full_reco_test = None
full_reco_train = None
full_reco_val = None
full_weights_test = None
full_weights_train = None
full_weights_val = None
full_output_label_names = None
full_output_transform_factors = None
full_input_transform_factors = None

do_reco = True
do_validate = True
do_weights = True
do_train = True

for a_file in event_file_names:
    print("Pulling data from %s"%a_file)

    f = h5py.File(a_file, "r")
    file_Y_test = f["Y_test"][:]
    file_X_test_DC = f["X_test_DC"][:]
    file_X_test_IC = f["X_test_IC"][:]
    try:
        file_reco_test = f["reco_test"][:]
    except:
        do_reco = False
        file_reco_test = None
    try:
        file_weights_test = f["weights_test"][:]
    except:
        do_weights = False
        file_weights_test = None

    try:
        file_Y_train = f["Y_train"][:]
        file_X_train_DC = f["X_train_DC"][:]
        file_X_train_IC = f["X_train_IC"][:]
    except:
        do_train = False
        file_Y_train = None
        file_X_train_DC = None
        file_X_train_IC = None
        
    try:
        file_Y_val = f["Y_validate"][:]
        file_X_val_DC = f["X_validate_DC"][:]
        file_X_val_IC = f["X_validate_IC"][:]
    except:
        do_validate = False
        file_Y_val = None
        file_X_val_DC = None
        file_X_val_IC = None

    if do_reco:
        if do_train:
            file_reco_train = f["reco_train"][:]
        else:
            file_reco_train = None
        if do_validate:
            file_reco_val = f["reco_validate"][:]
        else:
            file_reco_val = None
    if do_weights:
        if do_train:
            file_weights_train = f["weights_train"][:]
        else:
            file_weights_train = None
        if do_validate:
            file_weights_val = f["weights_validate"][:]
        else: 
            file_weights_val = None

    f.close()
    del f

    
    # Test first
    if full_Y_test is None:
        full_Y_test = file_Y_test
    else:
        full_Y_test = np.concatenate((full_Y_test, file_Y_test))
    if full_X_test_DC is None:
        full_X_test_DC = file_X_test_DC
    else:
        full_X_test_DC = np.concatenate((full_X_test_DC, file_X_test_DC))
    if full_X_test_IC is None:
        full_X_test_IC = file_X_test_IC
    else:
        full_X_test_IC = np.concatenate((full_X_test_IC, file_X_test_IC))
    if do_weights:
        if full_weights_test is None:
            full_weights_test = file_weights_test
        else:
            full_weights_test = np.concatenate((full_weights_test, file_weights_test))
    if do_reco:
        if full_reco_test is None:
            full_reco_test = file_reco_test
        else:
            full_reco_test = np.concatenate((full_reco_test, file_reco_test))
    
    # TRAIN 
    if do_train:
        if full_Y_train is None:
            full_Y_train = file_Y_train
        else:
            full_Y_train = np.concatenate((full_Y_train, file_Y_train))
        if full_X_train_DC is None:
            full_X_train_DC = file_X_train_DC
        else:
            full_X_train_DC = np.concatenate((full_X_train_DC, file_X_train_DC))
        if full_X_train_IC is None:
            full_X_train_IC = file_X_train_IC
        else:
            full_X_train_IC = np.concatenate((full_X_train_IC, file_X_train_IC))
        if do_weights:
            if full_weights_train is None:
                full_weights_train = file_weights_train
            else:
                full_weights_train = np.concatenate((full_weights_train, file_weights_train))
        if do_reco:
            if full_reco_train is None:
                full_reco_train = file_reco_train
            else:
                full_reco_train = np.concatenate((full_reco_train, file_reco_train))
    
    # VALIDATE
    if do_val:
        if full_Y_val is None:
            full_Y_val = file_Y_val
        else:
            full_Y_val = np.concatenate((full_Y_val, file_Y_val))
        if full_X_val_DC is None:
            full_X_val_DC = file_X_val_DC
        else:
            full_X_val_DC = np.concatenate((full_X_val_DC, file_X_val_DC))
        if full_X_val_IC is None:
            full_X_val_IC = file_X_val_IC
        else:
            full_X_val_IC = np.concatenate((full_X_val_IC, file_X_val_IC))
        if do_weights:
            if full_weights_val is None:
                full_weights_val = file_weights_val
            else:
                full_weights_val = np.concatenate((full_weights_val, file_weights_val))
        if do_reco:
            if full_reco_val is None:
                full_reco_val = file_reco_val
            else:
                full_reco_val = np.concatenate((full_reco_val, file_reco_val))

    #Check input/output transformations & names arrays for continuity
    if file_input_transform is not None:
        if full_input_transform is None:
            full_input_transform = file_input_transform
        else:
            #check they match
            comparison = full_input_transform == file_input_transform
            assert comparison.all(), "Mismatched input transform at file %i"%a_file
    if file_output_transform is not None:
        if full_output_transform is None:
            full_output_transform = file_output_transform
        else:
            #check they match
            comparison = full_output_transform == file_output_transform
            assert comparison.all(), "Mismatched input transform at file %i"%a_file
    if file_output_names is not None:
        if full_output_names is None:
            full_output_names = file_output_names
        else:
            #check they match
            comparison = full_output_names == file_output_names
            assert comparison.all(), "Mismatched input name at file %i"%a_file

#Save output to hdf5 file
output_name = path + output + ".hdf5" 
print("Saving to %s"%output_name)

savedlist = []

f = h5py.File(output_name, "w")
f.create_dataset("Y_test", data=full_Y_test)
f.create_dataset("X_test_DC", data=full_X_test_DC)
f.create_dataset("X_test_IC", data=full_X_test_IC)
savedlist.append("test")
if full_weights is not None:
    f.create_dataset("weights_test", data=full_weights_test)
    savedlist.append("weights")

f.create_dataset("Y_train", data=full_Y_train)
f.create_dataset("X_train_DC", data=full_X_train_DC)
f.create_dataset("X_train_IC", data=full_X_train_IC)
savedlist.append("train")
if full_weights is not None:
    f.create_dataset("weights_train", data=full_weights_train)

if validation is not None:
    f.create_dataset("Y_validate", data=full_Y_val)
    f.create_dataset("X_validate_IC", data=full_X_val_IC)
    f.create_dataset("X_validate_DC", data=full_X_val_DC)
    savedlist.append("validation")
    if full_weights is not None:
        f.create_dataset("weights_validate", data=full_weights_val)

if full_reco is not None:
    f.create_dataset("reco_test",data=full_reco_test)
    f.create_dataset("reco_train",data=full_reco_train)
    savedlist.append("old reco")
    if create_validation:
        f.create_dataset("reco_validate",data=full_reco_val)

if full_output_names is not None:
    f.create_dataset("output_label_names",data=full_output_names)
    savedlist.append("output label names")
if full_output_transform is not None:
    f.create_dataset("output_transform_factors",data=full_output_transform)
    savedlist.append("output transform factors")
if full_input_transform is not None:
    f.create_dataset("input_transform_factors",data=full_input_transform)
    savedlist.append("input transform factors")

f.close()


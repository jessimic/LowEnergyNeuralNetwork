#################################
# Concatonates, cuts, and shuffles hdf5 training data sets
#   Inputs:
#       -i input files: name of file (can use * and ?)
#       -d path: path to input files
#       -o ouput: name of output file, placed in path directory
#       -c cuts: name of cuts you want to apply (i.e. track only = track)
#       -r reco: bool if files have pegleg reco in them
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
parser.add_argument("-o", "--output",type=str,default='Level5_IC86.2013_genie_numu.014640.',
                    dest="output", help="names for output files")
parser.add_argument("-r","--reco",type=str, default=False,
                    dest="reco", help="bool if the file has pegleg reco info + initial stats, etc. (if using Level5p files)")
parser.add_argument("-v","--validate",type=str, default=True,
                    dest="validate", help="bool if validation is already split")
args = parser.parse_args()
input_files = args.input_files
path = args.path
output = args.output
if args.reco == "True" or args.reco == "true":
    use_old_reco = True
    print("Expecting old reco values in file, from pegleg, etc.")
else:    
    use_old_reco = False
if args.validate == "False" or args.validate == "false":
    use_validate = False
else:
    use_validate = True



file_names = path + input_files
event_file_names = sorted(glob.glob(file_names))
assert event_file_names,"No files loaded, please check path."

full_X_train_DC = None
full_X_test_DC = None
full_X_validate_DC = None
full_X_train_IC = None
full_X_test_IC = None
full_X_validate_IC = None
full_Y_train = None
full_Y_test = None
full_Y_validate = None
full_reco_train = None
full_reco_test = None
full_reco_validate = None
full_initial_stats = None
full_num_pulses = None

for a_file in event_file_names:

    print("Reading file %s"%a_file)

    f = h5py.File(a_file, "r")
    file_Y_train = f["Y_train"][:]
    file_Y_test = f["Y_test"][:]
    file_X_train_DC = f["X_train_DC"][:]
    file_X_test_DC = f["X_test_DC"][:]
    file_X_train_IC = f["X_train_IC"][:]
    file_X_test_IC = f["X_test_IC"][:]
    if use_validate:
        file_Y_validate = f[ "Y_validate"][:]
        file_X_validate_IC = f["X_validate_IC"][:]
        file_X_validate_DC = f["X_validate_DC"][:]
    if use_old_reco:
        file_reco_test = f["reco_test"][:]
        file_reco_train = f["reco_train"][:]
        if use_validate:
            file_reco_validate = f["reco_validate"][:]
    f.close()
    del f

    # X_train and X_test and X_validate DC
    if full_X_test_DC is None:
        full_X_test_DC = file_X_test_DC
    else:
        full_X_test_DC = np.concatenate((full_X_test_DC, file_X_test_DC))
    if full_X_train_DC is None:
        full_X_train_DC = file_X_train_DC
    else:
        full_X_train_DC = np.concatenate((full_X_train_DC, file_X_train_DC))
    if use_validate:
        if full_X_validate_DC is None:
            full_X_validate_DC = file_X_validate_DC
        else:
            full_X_validate_DC = np.concatenate((full_X_validate_DC, file_X_validate_DC)) 
    
    # X_train and X_test and X_validate IC
    if full_X_test_IC is None:
        full_X_test_IC = file_X_test_IC
    else:
        full_X_test_IC = np.concatenate((full_X_test_IC, file_X_test_IC))
    if full_X_train_IC is None:
        full_X_train_IC = file_X_train_IC
    else:
        full_X_train_IC = np.concatenate((full_X_train_IC, file_X_train_IC))
    if use_validate:
        if full_X_validate_IC is None:
            full_X_validate_IC = file_X_validate_IC
        else:
            full_X_validate_IC = np.concatenate((full_X_validate_IC, file_X_validate_IC)) 
    
    # Y_train and Y_test and Y_validate
    if full_Y_test is None:
        full_Y_test = file_Y_test
    else:
        full_Y_test = np.concatenate((full_Y_test, file_Y_test))
    if full_Y_train is None:
        full_Y_train = file_Y_train
    else:
        full_Y_train = np.concatenate((full_Y_train, file_Y_train))
    if use_validate:
        if full_Y_validate is None:
            full_Y_validate = file_Y_validate
        else:
            full_Y_validate = np.concatenate((full_Y_validate, file_Y_validate)) 

    if use_old_reco:
        if full_reco_test is None:
            full_reco_test = file_reco_test
        else:
            full_reco_test = np.concatenate((full_reco_test, file_reco_test))
        if full_reco_train is None:
            full_reco_train = file_reco_train
        else:
            full_reco_train = np.concatenate((full_reco_train, file_reco_train))
        if use_validate:
            if full_reco_validate is None:
                full_reco_validate = file_reco_validate
            else:
                full_reco_validate = np.concatenate((full_reco_validate, file_reco_validate)) 

    print("Finished file: %s"%(a_file))
    

validation_number = full_X_validate_DC.shape[0]
#Save output to hdf5 file
print("Total events saved: %i training, %i validation, %i testing"%(full_X_train_DC.shape[0],validation_number,full_X_test_DC.shape[0]))
output_name = path + output +".hdf5" 
print("I put everything into %s"%output_name)
f = h5py.File(output_name, "w")
f.create_dataset("Y_train", data=full_Y_train)
f.create_dataset("Y_test", data=full_Y_test)
f.create_dataset("X_train_DC", data=full_X_train_DC)
f.create_dataset("X_test_DC", data=full_X_test_DC)
f.create_dataset("X_train_IC", data=full_X_train_IC)
f.create_dataset("X_test_IC", data=full_X_test_IC)
if use_validate:
    f.create_dataset("Y_validate", data=full_Y_validate)
    f.create_dataset("X_validate_IC", data=full_X_validate_IC)
    f.create_dataset("X_validate_DC", data=full_X_validate_DC)
if use_old_reco:
    f.create_dataset("reco_train",data=full_reco_train)
    f.create_dataset("reco_test",data=full_reco_test)
    if use_validate:
        f.create_dataset("reco_validate",data=full_reco_validate)
f.close()

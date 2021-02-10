#######################
# Reconcat
# Meant to redistribute number of files, after transform
# Takes in multiple input files
# Concats them together
# Puts out new number of output files as specified
#########################

import numpy as np
import h5py
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_file",default=None,
                    type=str,dest="input_file", help="name for ONE input file")
parser.add_argument("-d", "--path",type=str,default='/data/icecube/jmicallef/processed_CNN_files/',
                    dest="path", help="path to input files")
parser.add_argument("-o", "--outdir",default=None,
                    dest="outdir", help="out directory for new file")
parser.add_argument("-n", "--name",default=None,
                    dest="name", help="base name for new file")
parser.add_argument("--num_out",type=int,default=1,
                    dest="num_out",help="number of output files you want to split the output into")
args = parser.parse_args()

file_names = args.path + args.input_file
event_file_names = sorted(glob.glob(file_names))
outpath = args.outdir
new_name = args.name
num_outputs = args.num_out
if outpath is None:
    outpath = args.path
if new_name is None:
    new_name = event_file_names[0]

first_file = event_file_names[0]
#Load in test data for first file
f = h5py.File(first_file, 'r')
Y_test = f['Y_test'][:]
X_test_DC = f['X_test_DC'][:]
X_test_IC = f['X_test_IC'][:]
try:
    reco_test = f['reco_test'][:]
    has_reco = True
    print("Has old reco, using...")
except:
    has_reco = False
try:
    Y_train = f['Y_train'][:]
    X_train_DC = f['X_train_DC'][:]
    X_train_IC = f['X_train_IC'][:]
    Y_validate = f['Y_validate'][:]
    X_validate_DC = f['X_validate_DC'][:]
    X_validate_IC = f['X_validate_IC'][:]
    train_validate = True
    if has_reco:
        reco_train = f["reco_train"][:]
        reco_validate = f["reco_validate"][:]
    print("Has train and validation sets, using...")
    print(Y_train.shape[0],Y_validate.shape[0])
except:
    train_validate = False

f.close()
del f

full_Y_test = Y_test
full_X_test_DC = X_test_DC
full_X_test_IC = X_test_IC
if has_reco:
    full_reco_test = reco_test
if train_validate:
    full_Y_train = Y_train
    full_X_train_DC = X_train_DC
    full_X_train_IC = X_train_IC
    full_Y_validate = Y_validate
    full_X_validate_DC = X_validate_DC
    full_X_validate_IC = X_validate_IC
    if has_reco:
        full_reco_train = reco_train
        full_reco_validate = reco_validate
print("Test events this file: %i, Cumulative saved: %i\n Finsihed file: %s"%(Y_test.shape[0],full_Y_test.shape[0],first_file)) 

for a_file in event_file_names[1:]:
    f = h5py.File(a_file, 'r')
    Y_test = f['Y_test'][:]
    X_test_DC = f['X_test_DC'][:]
    X_test_IC = f['X_test_IC'][:]
    if has_reco:
         reco_test = f['reco_test'][:]
    if train_validate:
        Y_train = f['Y_train'][:]
        X_train_DC = f['X_train_DC'][:]
        X_train_IC = f['X_train_IC'][:]
        Y_validate = f['Y_validate'][:]
        X_validate_DC = f['X_validate_DC'][:]
        X_validate_IC = f['X_validate_IC'][:]
        if has_reco:
            reco_train = f["reco_train"][:]
            reco_validate = f["reco_validate"][:]
    f.close()
    del f

    full_Y_test = np.concatenate((full_Y_test, Y_test))
    full_X_test_DC = np.concatenate((full_X_test_DC, X_test_DC))
    full_X_test_IC = np.concatenate((full_X_test_IC, X_test_IC))
    if has_reco:
        full_reco_test = np.concatenate((full_reco_test, reco_test))
    if train_validate:
        full_Y_train = np.concatenate((full_Y_train, Y_train))
        full_X_train_DC = np.concatenate((full_X_train_DC, X_train_DC))
        full_X_train_IC = np.concatenate((full_X_train_IC, X_train_IC))
        full_Y_validate = np.concatenate((full_Y_validate, Y_validate))
        full_X_validate_DC = np.concatenate((full_X_validate_DC, X_validate_DC))
        full_X_validate_IC = np.concatenate((full_X_validate_IC, X_validate_IC))
        if has_reco:
            full_reco_train = np.concatenate((full_reco_train, reco_train))
            full_reco_validate = np.concatenate((full_reco_validate, reco_validate))

    print("Test events this file: %i, Cumulative saved: %i\n Finsihed file: %s"%(Y_test.shape[0],full_Y_test.shape[0],a_file)) 

test_events_per_file = int(full_Y_test.shape[0]/num_outputs) + 1
if train_validate:
    train_events_per_file = int(full_Y_train.shape[0]/num_outputs) + 1
    val_events_per_file = int(full_Y_validate.shape[0]/num_outputs) + 1
for sep_file in range(0,num_outputs):
    test_start = test_events_per_file*sep_file
    train_start = train_events_per_file*sep_file
    val_start = val_events_per_file*sep_file
    if sep_file < num_outputs-1:
        test_end = test_events_per_file*(sep_file+1)
        train_end = train_events_per_file*(sep_file+1)
        val_end = val_events_per_file*(sep_file+1)
    else:
        test_end = full_Y_test.shape[0]
        train_end = full_Y_train.shape[0]
        val_end = full_Y_validate.shape[0]
    output_file = outpath + new_name + "_file%02d.hdf5"%sep_file
    print("I put test events %i - %i into %s"%(test_start,test_end,output_file))
    f = h5py.File(output_file, "w")
    f.create_dataset("Y_test", data=full_Y_test[test_start:test_end])
    f.create_dataset("X_test_DC", data=full_X_test_DC[test_start:test_end])
    f.create_dataset("X_test_IC", data=full_X_test_IC[test_start:test_end])
    if has_reco:
        f.create_dataset("reco_test", data=full_reco_test[test_start:test_end])
    if train_validate:
        print("I put train events %i - %i and validate events %i - %i into file"%(train_start,train_end,val_start,val_end))
        f.create_dataset("Y_train", data=full_Y_train[train_start:train_end])
        f.create_dataset("X_train_DC", data=full_X_train_DC[train_start:train_end])
        f.create_dataset("X_train_IC", data=full_X_train_IC[train_start:train_end])
        f.create_dataset("Y_validate", data=full_Y_validate[val_start:val_end])
        f.create_dataset("X_validate_DC", data=full_X_validate_DC[val_start:val_end])
        f.create_dataset("X_validate_IC", data=full_X_validate_IC[val_start:val_end])
        if has_reco:
            f.create_dataset("reco_train", data=full_reco_train[train_start:train_end])
            f.create_dataset("reco_validate", data=full_reco_validate[val_start:val_end])
    f.close()

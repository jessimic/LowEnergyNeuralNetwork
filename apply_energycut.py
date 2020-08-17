import numpy as np
import h5py
import os, sys

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_file",default=None,
                    type=str,dest="input_file", help="name for ONE input file")
parser.add_argument("-d", "--path",type=str,default='/data/icecube/jmicallef/processed_CNN_files/',
                    dest="path", help="path to input files")
parser.add_argument("--emax",type=float,default=None,
                    dest="emax",help="Cut max energy")
parser.add_argument("--emin",type=float,default=None,
                    dest="emin",help="Cut min energy")
parser.add_argument("--efactor",type=float,default=None,
                    dest="efactor",help="Factor to multiple energy by")
args = parser.parse_args()

input_file = args.path + args.input_file
emax = args.emax
emin = args.emin
cut_name = "%iemax_%iemin"%(int(emax),int(emin))
efactor = args.efactor
if efactor is None:
    efactor = emax
    print("ASSUMING EFACTOR IS THE SAME AS EMAX (%f)"%emax)
emin = emin/efactor
emax = emax/efactor


f = h5py.File(input_file, 'r')
Y_test = f['Y_test'][:]
X_test_DC = f['X_test_DC'][:]
X_test_IC = f['X_test_IC'][:]
try:
    reco_test = f["reco_test"]
    reco = True
except:
    reco = False
print(Y_test.shape[0])

try:
    Y_train = f['Y_train'][:]
    X_train_DC = f['X_train_DC'][:]
    X_train_IC = f['X_train_IC'][:]
    Y_validate = f['Y_validate'][:]
    X_validate_DC = f['X_validate_DC'][:]
    X_validate_IC = f['X_validate_IC'][:]
    train_validate = True
    if reco:
        reco_train = f["reco_train"]
        reco_validate = f["reco_validate"]
    print(Y_train.shape[0],Y_validate.shape[0])
except:
    train_validate = False
    
f.close()
del f

# Apply Energy Cuts
test_mask = np.logical_and(Y_test[:,0] > emin, Y_test[:,0] < emax)
Y_test = Y_test[test_mask]
X_test_DC = X_test_DC[test_mask]
X_test_IC = X_test_IC[test_mask]
if reco:
    reco_test = reco_test[test_mask]
print(Y_test.shape[0])

if train_validate:
    train_mask = np.logical_and(Y_train[:,0] > emin, Y_train[:,0] < emax)
    Y_train = Y_train[train_mask]
    X_train_DC = X_train_DC[train_mask]
    X_train_IC = X_train_IC[train_mask]
    if reco:
        reco_train = reco_train[train_mask]
    
    val_mask = np.logical_and(Y_validate[:,0] > emin, Y_validate[:,0] < emax)
    Y_validate = Y_validate[val_mask]
    X_validate_DC = X_validate_DC[val_mask]
    X_validate_IC = X_validate_IC[val_mask]
    if reco:
        reco_validate = reco_validate[val_mask]

    print(Y_train.shape[0],Y_validate.shape[0])

output_file = input_file[:-5] + "_%s.hdf5"%cut_name
print("Putting new file at %s"%output_file)

f = h5py.File(output_file, "w")
f.create_dataset("Y_test", data=Y_test)
f.create_dataset("X_test_DC", data=X_test_DC)
f.create_dataset("X_test_IC", data=X_test_IC)
if reco:
    f.create_dataset("reco_test", data=reco_test)

if train_validate:
    f.create_dataset("Y_train", data=Y_train)
    f.create_dataset("X_train_DC", data=X_train_DC)
    f.create_dataset("X_train_IC", data=X_train_IC)
    f.create_dataset("Y_validate", data=Y_validate)
    f.create_dataset("X_validate_DC", data=X_validate_DC)
    f.create_dataset("X_validate_IC", data=X_validate_IC)
    if reco:
        f.create_dataset("reco_train", data=reco_train)
        f.create_dataset("reco_validate", data=reco_validate)
f.close()

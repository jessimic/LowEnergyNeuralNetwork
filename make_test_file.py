import glob
import numpy
import h5py
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_files",type=str,default=None,
                    dest="input_file", help="names for input files")
parser.add_argument("-d", "--path",type=str,default='/data/icecube/jmicallef/processed_CNN_files/',
                    dest="path", help="path to input files")
parser.add_argument("-n", "--name",type=str,default=None,
                    dest="name", help="name for output file")
parser.add_argument("--old_reco",default=False,action='store_true',
                    dest="old_reco",help="use flag if concatonating all train, test, val into one file")
args = parser.parse_args()

use_old_reco = args.old_reco
file_name_base = args.path + args.input_file
if not use_old_reco:
    file_names = sorted(glob.glob(file_name_base))
    print("Using %i files with names like %s"%(len(file_names), file_names[0]))
else:
    file_names = file_name_base
    print("Using file %s"%file_names)

name = args.name
if name is None:
    split_file_name = file_name_base[:-4]
    new_name = split_file_name[0]
    for name in range(1,len(split_file_name)-1):
        new_name = new_name + "_" + split_file_name[name]
    new_name += ".testonly.hdf5"
    output_file =  new_name
else:
    output_file = args.path + name + ".testonly.hdf5"

# Put all the test sets together
Y_test_use = None
X_test_DC_use = None
X_test_IC_use = None

if use_old_reco:
    f = h5py.File(file_names, 'r')
    Y_test = f['Y_test'][:]
    X_test_DC = f['X_test_DC'][:]
    X_test_IC = f['X_test_IC'][:]
    Y_train = f['Y_train'][:]
    X_train_DC = f['X_train_DC'][:]
    X_train_IC = f['X_train_IC'][:]
    Y_validate = f['Y_validate'][:]
    X_validate_DC = f['X_validate_DC'][:]
    X_validate_IC = f['X_validate_IC'][:]
    reco_test = f['reco_test'][:]
    reco_train = f['reco_train'][:]
    reco_validate = f['reco_validate'][:]
    f.close()
    del f

    print("Loaded all %i events"%(Y_test.shape[0]+Y_train.shape[0]+Y_validate.shape[0]))

    Y_test_use = numpy.concatenate((Y_test,Y_train,Y_validate))
    print("Concatted Y")
    del Y_test
    del Y_train
    del Y_validate
    X_test_DC_use = numpy.concatenate((X_test_DC,X_train_DC,X_validate_DC))
    print("Concatted DC")
    del X_test_DC
    del X_train_DC
    del X_validate_DC
    X_test_IC_use =numpy.concatenate((X_test_IC,X_train_IC,X_validate_IC))
    print("Concatted IC")
    del X_test_IC
    del X_train_IC
    del X_validate_IC
    reco_test_use = numpy.concatenate((reco_test,reco_train,reco_validate))
    del reco_test
    del reco_train
    del reco_validate
    print("Concatted reco")

else:
    for file in file_names:
        f = h5py.File(file, 'r')
        Y_test = f['Y_test'][:]
        X_test_DC = f['X_test_DC'][:]
        X_test_IC = f['X_test_IC'][:]
        reco_test = f['reco_test'][:]
        f.close()
        del f

        if Y_test_use is None:
            Y_test_use = Y_test
            X_test_DC_use = X_test_DC
            X_test_IC_use = X_test_IC
            reco_test_use = reco_test
        else:
            Y_test_use = numpy.concatenate((Y_test_use, Y_test))
            X_test_DC_use = numpy.concatenate((X_test_DC_use, X_test_DC))
            X_test_IC_use = numpy.concatenate((X_test_IC_use, X_test_IC))
            reco_test_use = numpy.concatenate((reco_test_use, reco_test))

print(Y_test_use.shape)

print("Saving output file: %s"%output_file)
f = h5py.File(output_file, "w")
f.create_dataset("Y_test", data=Y_test_use)
f.create_dataset("X_test_DC", data=X_test_DC_use)
f.create_dataset("X_test_IC", data=X_test_IC_use)
#if use_old_reco:
f.create_dataset("reco_test", data=reco_test_use)
f.close()

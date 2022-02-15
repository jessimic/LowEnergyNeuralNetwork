import glob
import numpy
import h5py
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_files",type=str,default=None,
                    dest="input_file", help="names for input files")
parser.add_argument("-d", "--path",type=str,default='/mnt/research/IceCube/jmicallef/DNN_files/',
                    dest="path", help="path to input files")
parser.add_argument("-n", "--name",type=str,default=None,
                    dest="name", help="name for output file")
parser.add_argument("--old_reco",default=False,action='store_true',
                    dest="old_reco",help="use flag if concatonating all train, test, val into one file")
parser.add_argument("--train_into_test",default=False,action='store_true',
                    dest="train_into_test",help="use flag if concatonating all train, test, val into one file")
args = parser.parse_args()

train_into_test = args.train_into_test
use_old_reco = args.old_reco
file_name_base = args.path + args.input_file
file_names = sorted(glob.glob(file_name_base))
num_files = len(file_names)
if num_files > 1:
    print("Using %i files with names like %s"%(num_files, file_names[0]))
else:
    #file_names = file_name_base
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
weights_test_use = None
reco_test_use = None

def concat_train_test(a_file,use_old_reco=False):

    f = h5py.File(a_file, 'r')
    Y_test = f['Y_test'][:]
    X_test_DC = f['X_test_DC'][:]
    X_test_IC = f['X_test_IC'][:]
    Y_train = f['Y_train'][:]
    X_train_DC = f['X_train_DC'][:]
    X_train_IC = f['X_train_IC'][:]
    Y_validate = f['Y_validate'][:]
    X_validate_DC = f['X_validate_DC'][:]
    X_validate_IC = f['X_validate_IC'][:]
    if use_old_reco:
        reco_test = f['reco_test'][:]
        reco_train = f['reco_train'][:]
        reco_validate = f['reco_validate'][:]
    weights_test = f['weights_test'][:]
    weights_train = f['weights_train'][:]
    weights_validate = f['weights_validate'][:]
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
    if use_old_reco:
        reco_test_use = numpy.concatenate((reco_test, reco_train, reco_validate))
        del reco_test
        del reco_train
        del reco_validate
        print("Concatted reco")
    else:
        reco_test_use = None
    weights_test_use = numpy.concatenate((weights_test, weights_train, weights_validate))
    del weights_test
    del weights_train
    del weights_validate
    print("Concatted weights")

    return Y_test_use, X_test_DC_use, X_test_IC_use, reco_test_use, weights_test_use

def append_to_test(Y_test,X_test_DC,X_test_IC,reco_test,weights_test,Y_test_use=None,X_test_DC_use=None,X_test_IC_use=None,reco_test_use=None,weights_test_use=None):

    if Y_test_use is None:
        Y_test_use = Y_test
        X_test_DC_use = X_test_DC
        X_test_IC_use = X_test_IC
        reco_test_use = reco_test
        weights_test_use = weights_test
        print("Created new array with %i events"%Y_test_use.shape[0])
    else:
        Y_test_use = numpy.concatenate((Y_test_use, Y_test))
        X_test_DC_use = numpy.concatenate((X_test_DC_use, X_test_DC))
        X_test_IC_use = numpy.concatenate((X_test_IC_use, X_test_IC))
        if reco_test_use is not None:
            reco_test_use = numpy.concatenate((reco_test_use, reco_test))
        if weights_test_use is not None:
            weights_test_use = numpy.concatenate((weights_test_use, weights_test))
    
        print("Added %i events"%Y_test_use.shape[0])
    
    return Y_test_use, X_test_DC_use, X_test_IC_use, reco_test_use, weights_test_use

Y_test_full=None
X_test_DC_full=None
X_test_IC_full=None
reco_test_full=None
weights_test_full=None
for a_file in file_names:
    if train_into_test or use_old_reco:
        Y_test, X_test_DC, X_test_IC, reco_test, weights_test = concat_train_test(a_file,use_old_reco=use_old_reco)
    
        if num_files > 1:
            Y_test_full, X_test_DC_full, X_test_IC_full, reco_test_full, weights_test_full = append_to_test(Y_test, X_test_DC, X_test_IC, reco_test, weights_test,  Y_test_use=Y_test_full, X_test_DC_use=X_test_DC_full, X_test_IC_use=X_test_IC_full, reco_test_use=reco_test_full, weights_test_use=weights_test_full)
        else:
            Y_test_full = Y_test
            Y_test_DC_full = X_test_DC
            Y_test_IC_full = X_test_IC
            reco_test_full = reco_test
            weights_test_full = weights_test

    else:
        f = h5py.File(a_file, 'r')
        Y_test = f['Y_test'][:]
        X_test_DC = f['X_test_DC'][:]
        X_test_IC = f['X_test_IC'][:]
        try:
            reco_test = f['reco_test'][:]
        except:
            reco_test = None
        try:
            weights_test = f['weights_test'][:]
        except:
            weights_test = None
        f.close()
        del f
        Y_test_full, X_test_DC_full, X_test_IC_full, reco_test_full, weights_test_full = append_to_test(Y_test, X_test_DC, X_test_IC, reco_test, weights_test, Y_test_use=Y_test_full, X_test_DC_use=X_test_DC_full, X_test_IC_use=X_test_IC_full, reco_test_use=reco_test_full,weights_test_use=weights_test_full)

true_PID = Y_test_full[:,9]
NuE = true_PID == 12
NuMu = true_PID == 14
cut = NuMu
Y_test_full = Y_test_full[cut]
X_test_DC_full = X_test_DC_full[cut]
X_test_IC_full = X_test_IC_full[cut]
weights_test_full = weights_test_full[cut]

print("Cutting events and only saving %i"%sum(cut))

print("Saving output file: %s"%output_file)
f = h5py.File(output_file, "w")
f.create_dataset("Y_test", data=Y_test_full)
f.create_dataset("X_test_DC", data=X_test_DC_full)
f.create_dataset("X_test_IC", data=X_test_IC_full)
if weights_test_use is not None:
    f.create_dataset("weights_test", data=weights_test_full)
if use_old_reco:
    f.create_dataset("reco_test", data=reco_test_full)
f.close()

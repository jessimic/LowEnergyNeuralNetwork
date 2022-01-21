import glob
import numpy
import h5py
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_files",type=str,default=None,
                    dest="input_file", help="names for input files")
parser.add_argument("-d", "--path",type=str,default='/mnt/research/IceCube/jmicallef/DNN_files/',
                    dest="path", help="path to input files")
parser.add_argument("-o", "--outdir",default=None,
                    dest="output_dir", help="path of ouput file, if different that the output_plot model dir")
parser.add_argument("-n", "--name",type=str,default="prediction_values",
                    dest="name", help="name for output file")
args = parser.parse_args()

file_name_base = args.path + args.input_file
file_names = sorted(glob.glob(file_name_base))
num_files = len(file_names)
print("Using %i files with names like %s"%(num_files, file_names[0]))

name = args.name
output_file =args.output_dir + name + ".hdf5"

# Put all the test sets together
Y_test_full = None
predict_full = None
weights_test_full = None
reco_test_full = None
info_test_full = None

for a_file in file_names:

    f = h5py.File(a_file, "r")
    list(f.keys())
    Y_test = f["Y_test_use"][:]
    predict = f["Y_predicted"][:]
    try:
        info_test = f["additional_info"][:]
    except:
        info_test = None
    try:
        weights_test = f["weights_test"][:]
    except:
        weights_test = None
    try:
        reco_test = f["reco_test"][:]
    except:
        reco_test = None
    f.close()
    del f 

    if Y_test_full is None:
        Y_test_full = Y_test
        predict_full = predict
        reco_test_full = reco_test
        weights_test_full = weights_test
        info_test_full = info_test
        print("Created new array with %i events"%Y_test_full.shape[0])
    else:
        Y_test_full = numpy.concatenate((Y_test_full, Y_test))
        predict_full = numpy.concatenate((predict_full, predict))
        if reco_test_full is not None:
            reco_test_full = numpy.concatenate((reco_test_full, reco_test))
        if weights_test_full is not None:
            weights_test_full = numpy.concatenate((weights_test_full, weights_test))
        if info_test_full is not None:
            info_test_full = numpy.concatenate((info_test_full, info_test))
    
        print("Added %i events"%Y_test_full.shape[0])
    
print("Saving output file: %s"%output_file)
f = h5py.File(output_file, "w")
f.create_dataset("Y_test_use", data=Y_test_full)
f.create_dataset("Y_predicted", data=predict_full)
if weights_test_full is not None:
    f.create_dataset("weights_test", data=weights_test_full)
if reco_test_full is not None:
    f.create_dataset("reco_test", data=reco_test_full)
if info_test_full is not None:
    f.create_dataset("additional_info", data=info_test_full)
f.close()

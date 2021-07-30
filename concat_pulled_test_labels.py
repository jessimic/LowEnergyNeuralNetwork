import h5py
import argparse
import os, sys
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-i","--inputs",nargs="+",default=[],
                    dest="input_files",help="list of strings, paths + filesnames")
parser.add_argument("-n","--outname",default=None,
                    dest="outname",help="name for output file (do not need .hdf5)")
parser.add_argument("-o","--outdir",type=str,default="/mnt/home/micall12/LowEnergyNeuralNetwork/output_plots/",
                    dest="output_dir",help="path for output file")
args = parser.parse_args()

input_files = args.input_files
outdir = args.output_dir
if args.outname is None:
    output_name = "prediction_values"
else:
    output_name = args.outname

count_files_concatted = 0
truth = None
predict = None
reco = None
weights = None
info = None
for input_file in input_files:
    print("Concatting file %s"%input_file)
    f = h5py.File(input_file, "r")
    file_truth = f["Y_test_use"][:]
    file_predict = f["Y_predicted"][:]
    try:
        file_reco = f["reco_test"][:]
    except:
        file_reco = None
    try:
        file_weights = f["weights_test"][:]
    except:
        file_weights = None
    try:
        file_info = f["additional_info"][:]
    except: 
        file_info = None
    f.close()
    del f

    if truth is None:
        truth = file_truth
    else:
        truth = np.concatenate((truth, file_truth))
    if predict is None:
        predict = file_predict
    else:
        predict = np.concatenate((predict, file_predict))
    if reco is None:
        reco = file_reco
    else:
        reco = np.concatenate((reco, file_reco))
    if weights is None:
        weights = file_weights
    else:
        weights = np.concatenate((weights, file_weights))
    if info is None:
        info = file_info
    else:
        info = np.concatenate((info, file_info))

    count_files_concatted += 1

print("Concatted %i files together"%count_files_concatted)
f = h5py.File("%s/%s.hdf5"%(outdir,output_name), "w")
f.create_dataset("Y_predicted", data=predict)
f.create_dataset("Y_test_use", data=truth)
f.create_dataset("reco_test", data=reco)
f.create_dataset("additional_info", data=info)
f.create_dataset("weights_test", data=weights)
f.close()

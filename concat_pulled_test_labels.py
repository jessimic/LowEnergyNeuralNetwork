import h5py
import argparse
import os, sys
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import matplotlib.colors as colors

input_file = "/mnt/home/micall12/LowEnergyNeuralNetwork/output_plots/Test_Level6.5/L7_nue_official/prediction_values.hdf5"
input_file2 = "/mnt/home/micall12/LowEnergyNeuralNetwork/output_plots/Test_Level6.5/L7_nue_official/prediction_values.hdf5"
outdir = "/mnt/home/micall12/LowEnergyNeuralNetwork/output_plots/Test_Level6.5/L7_PID"
output_name = "prediction_values"

f = h5py.File(input_file, "r")
truth = f["Y_test_use"][:]
predict = f["Y_predicted"][:]
try:
    reco = f["reco_test"][:]
except:
    reco = None
try:
    weights = f["weights_test"][:]
except:
    weights = None
try:
    info = f["additional_info"][:]
except: 
    info = None
f.close()
del f

f = h5py.File(input_file2, "r")
truth2 = f["Y_test_use"][:]
predict2 = f["Y_predicted"][:]
try:
    reco2 = f["reco_test"][:]
except:
    reco2 = None
try:
    weights2 = f["weights_test"][:]
except:
    weights2 = None
try:
    info2 = f["additional_info"][:]
except: 
    info2 = None
f.close()
del f

truth = np.concatenate((truth, truth2))
predict = np.concatenate((predict, predict2))
reco = np.concatenate((reco, reco2))
weights = np.concatenate((weights, weights2))
info = np.concatenate((info, info2))

f = h5py.File("%s/%s.hdf5"%(outdir,output_name), "w")
f.create_dataset("Y_predicted", data=predict)
f.create_dataset("Y_test_use", data=truth)
f.create_dataset("reco_test", data=reco)
f.create_dataset("additional_info", data=info)
f.create_dataset("weights_test", data=weights)
f.close()

import h5py
import argparse
import os, sys
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import matplotlib.colors as colors

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input",type=str,default=None,
                    dest="input_file", help="path and name of the input file")
parser.add_argument("--input2",type=str,default=None,
                    dest="input_file2", help="path and name of the input file")
parser.add_argument("-o", "--outdir",type=str,default='/mnt/home/micall12/LowEnergyNeuralNetwork/output_plots/',
                    dest="output_dir", help="path of ouput file")
args = parser.parse_args()

input_file = args.input_file
input_file2 = args.input_file2
save_folder_name = args.output_dir

f = h5py.File(input_file, "r")
truth = f["Y_test_use"][:]
predict = f["Y_predicted"][:]
reco = f["reco_test"][:]
weights = f["weights_test"][:]
try:
    info = f["additional_info"][:]
except: 
    info = None
f.close()
del f

cnn_energy = np.array(predict[:,0])*100
true_energy = np.array(truth[:,0])
true_CC = np.array(truth[:,11])

#hits8 = info[:,9]
check_energy_gt5 = true_energy > 5.
assert sum(check_energy_gt5)>0, "No events > 5 GeV in true energy, is this transformed?"

print(min(true_energy),max(true_energy))


#Plot
from PlottingFunctions import plot_distributions
from PlottingFunctions import plot_2D_prediction

save=True
if save ==True:
    print("Saving to %s"%save_folder_name)

plot_name = "Energy"
plot_units = "(GeV)"
maxabs_factors = 100.

cuts = true_CC == 1
save_base_name = save_folder_name
minval = 0
maxval = 100
bins = 100
syst_bin = 20
true_weights = None #weights[cuts]/1510.


print(true_energy[cuts][:10], cnn_energy[cuts][:10])


plot_2D_prediction(true_energy[cuts], cnn_energy[cuts],weights=true_weights,\
                        save=save, savefolder=save_folder_name,bins=bins,\
                        variable=plot_name, units=plot_units, reco_name="CNN")

plot_distributions(true_energy[cuts], cnn_energy[cuts], weights=true_weights,\
                   use_old_reco = False,\
                   minaxis=-maxval, maxaxis=maxval, bins=bins,\
                   save=save, savefolder=save_folder_name,\
                   variable=plot_name, units=plot_units, reco_name="CNN")

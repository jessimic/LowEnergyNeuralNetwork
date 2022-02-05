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
parser.add_argument("-o", "--outdir",type=str,default='/mnt/home/micall12/LowEnergyNeuralNetwork/output_plots/',
                    dest="output_dir", help="path of ouput file")
args = parser.parse_args()

input_file = args.input_file
save_folder_name = args.output_dir

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

cnn_energy = np.array(predict[:,0])
true_energy = np.array(truth[:,0])
true_CC = np.array(truth[:,11])

numu_files = 391 #1519
nue_files = 602
#weights
if weights is not None:
    weights = np.array(weights[:,8])
    #modify by number of files
    mask_numu = np.array(truth[:,9]) == 14
    mask_nue = np.array(truth[:,9]) == 12
    if sum(mask_numu) > 1:
        weights[mask_numu] = weights[mask_numu]/numu_files
    if sum(mask_nue) > 1:
        weights[mask_nue] = weights[mask_nue]/nue_files

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
minval = 1
maxval = 100
bins = 100
syst_bin = 100
true_weights = weights[cuts]


print(true_energy[cuts][:10], cnn_energy[cuts][:10])

plot_distributions(true_energy[cuts],cnn_energy[cuts]
                    weights=true_weights
                    save=save, savefolder=save_folder_name,
                    cnn_name = "CNN", variable=plot_name,
                    units= plot_units,
                    minval=minval,maxval=maxval,
                    bins=bins)

plot_2D_prediction(true_energy[cuts], cnn_energy[cuts],
                    weights=true_weights,bins=bins,
                    save=save, savefolder=save_folder_name,
                    variable=plot_name, units=plot_units, 
                    reco_name="CNN")

plot_single_resolution(true_energy[cuts], cnn_energy[cuts],
                        weights=true_weights,
                        use_old_reco = False,
                        minaxis=-2, maxaxis=2, 
                        bins=bins, use_fraction=True,\
                        save=save, savefolder=save_folder_name,\
                        variable=plot_name, units=plot_units,
                        reco_name="CNN")

plot_bin_slices(true_energy[cuts], cnn_energy[cuts],
                weights=true_weights,
                use_fraction = True, bins=syst_bin,
                min_val=minval, max_val=maxval,
                save=save, savefolder=save_folder_name,
                variable=plot_name, units=plot_units,
                cnn_name="CNN")

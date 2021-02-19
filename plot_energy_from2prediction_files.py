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
f2 = h5py.File(input_file2, "r")
truth2 = f2["Y_test_use"][:]
predict2 = f2["Y_predicted"][:]
reco2 = f2["reco_test"][:]
weights2 = f2["weights_test"][:]
try:
    info2 = f2["additional_info"][:]
except: 
    info2 = None
f2.close()
del f2

cnn_energy = np.array(predict[:,0])*100
true_energy = np.array(truth[:,0])
true_CC = np.array(truth[:,11])
cnn_energy2 = np.array(predict2[:,0])*100
true_energy2 = np.array(truth2[:,0])
true_CC2 = np.array(truth2[:,11])


#hits8 = info[:,9]
check_energy_gt5 = true_energy > 5.
assert sum(check_energy_gt5)>0, "No events > 5 GeV in true energy, is this transformed?"

#Vertex Position
#x_origin = np.ones((len(true_x)))*46.290000915527344
#y_origin = np.ones((len(true_y)))*-34.880001068115234
#true_r = np.sqrt( (true_x - x_origin)**2 + (true_y - y_origin)**2 )

print(min(true_energy),max(true_energy))


#Plot
from PlottingFunctions import plot_distributions
from PlottingFunctions import plot_2D_prediction
from PlottingFunctions import plot_single_resolution
from PlottingFunctions import plot_bin_slices
from PlottingFunctions import plot_rms_slices

save=True
if save ==True:
    print("Saving to %s"%save_folder_name)

plot_name = "Energy"
plot_units = "(GeV)"
maxabs_factors = 100.

cuts = true_CC == 1
cuts2 = true_CC2 == 1
save_base_name = save_folder_name
minval = 0
maxval = 100
bins = 100
syst_bin = 20
true_weights = None #weights[cuts]/1510.


print(true_energy[cuts][:10], cnn_energy[cuts][:10])


plot_2D_prediction(true_energy[cuts], cnn_energy[cuts],weights=true_weights,\
                        save=save, savefolder=save_folder_name,bins=bins, switch_axis=True,
                        variable=plot_name, units=plot_units, reco_name="old GCD")
plot_2D_prediction(true_energy2[cuts2], cnn_energy2[cuts2], weights=true_weights,
                        save=save, savefolder=save_folder_name,bins=bins,switch_axis=True,\
                        variable=plot_name, units=plot_units, reco_name="new GCD")
plot_2D_prediction(true_energy[cuts], cnn_energy[cuts],weights=true_weights,\
                        save=save, savefolder=save_folder_name,bins=bins,switch_axis=True,\
                        minval=minval, maxval=maxval, cut_truth=True, axis_square=True,\
                        variable=plot_name, units=plot_units, reco_name="old GCD")
plot_2D_prediction(true_energy2[cuts2], cnn_energy2[cuts2], weights=true_weights,
                        save=save, savefolder=save_folder_name,bins=bins,switch_axis=True,\
                        minval=minval, maxval=maxval, cut_truth=True, axis_square=True,\
                        variable=plot_name, units=plot_units, reco_name="new GCD")

plot_single_resolution(true_energy[cuts], cnn_energy[cuts], weights=true_weights,\
                   use_old_reco = True, old_reco = cnn_energy2[cuts2], old_reco_truth=true_energy2[cuts2],\
                   minaxis=-maxval, maxaxis=maxval, bins=bins,\
                   save=save, savefolder=save_folder_name,\
                   variable=plot_name, units=plot_units, reco_name="new GCD")

plot_bin_slices(true_energy[cuts], cnn_energy[cuts], weights=true_weights,  
                    old_reco = cnn_energy2[cuts2],old_reco_truth=true_energy2[cuts2],\
                    use_fraction = True, bins=syst_bin, min_val=minval, max_val=maxval,\
                    save=save, savefolder=save_folder_name,\
                    variable=plot_name, units=plot_units, reco_name="new GCD")

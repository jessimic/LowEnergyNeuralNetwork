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
parser.add_argument("-n", "--savename",default=None,
                    dest="savename", help="additional directory to save in")
args = parser.parse_args()

input_file = args.input_file
input_file2 = args.input_file2
save_folder_name = args.output_dir + "/"
if args.savename is not None:
    save_folder_name += args.savename + "/"
    if os.path.isdir(save_folder_name) != True:
            os.mkdir(save_folder_name)
print("Saving to %s"%save_folder_name)

#IMPORT FILE 1
f = h5py.File(input_file, "r")
truth1 = f["Y_test_use"][:]
predict1 = f["Y_predicted"][:]
#reco1 = f["reco_test"][:]
raw_weights1 = f["weights_test"][:]
try:
    info1 = f["additional_info"][:]
except: 
    info1 = None
f.close()
del f

numu_files1 = 391
nue_files1 = 1
muon_files1 = 1
nutau_files1 = 1

#Truth
true1_energy = np.array(truth1[:,0])
true1_em_equiv_energy = np.array(truth1[:,14])
true1_x = np.array(truth1[:,4])
true1_y = np.array(truth1[:,5])
true1_z = np.array(truth1[:,6])
x1_origin = np.ones((len(true1_x)))*46.290000915527344
y1_origin = np.ones((len(true1_y)))*-34.880001068115234
true1_r = np.sqrt( (true1_x - x1_origin)**2 + (true1_y - y1_origin)**2 )
true1_isCC = np.array(truth1[:,11])
true1_isCC = np.array(true1_isCC,dtype=bool)
true1_isTrack = np.array(truth1[:,8])
true1_PID = truth1[:,9]

true1_zenith = np.array(truth1[:,12])
true1_coszenith = np.cos(np.array(truth1[:,12]))

#Reconstructed values (CNN)
reco1_energy = np.array(predict1[:,0])
reco1_prob_track = np.array(predict1[:,1])
reco1_zenith = np.array(predict1[:,2])
reco1_coszenith = np.cos(reco1_zenith)
reco1_x = np.array(predict1[:,3])
reco1_y = np.array(predict1[:,4])
reco1_z = np.array(predict1[:,5])
reco1_r = np.sqrt( (reco1_x - x1_origin)**2 + (reco1_y - y1_origin)**2 )
reco1_prob_mu = np.array(predict1[:,6])
reco1_nDOMs = np.array(predict1[:,7])

#PID identification
muon_mask_test1 = (true1_PID) == 13
true1_isMuon = np.array(muon_mask_test1,dtype=bool)
numu_mask_test1 = (true1_PID) == 14
true1_isNuMu = np.array(numu_mask_test1,dtype=bool)
nue_mask_test1 = (true1_PID) == 12
true1_isNuE = np.array(nue_mask_test1,dtype=bool)
nutau_mask_test1 = (true1_PID) == 16
true1_isNuTau = np.array(nutau_mask_test1,dtype=bool)
nu_mask1 = np.logical_or(np.logical_or(numu_mask_test1, nue_mask_test1), nutau_mask_test1)
true1_isNu = np.array(nu_mask1,dtype=bool)

#Weight adjustments
weights1 = raw_weights1[:,8]
if weights1 is not None:
    if sum(true1_isNuMu) > 1:
        weights1[true1_isNuMu] = weights1[true1_isNuMu]/numu_files1
    if sum(true1_isNuE) > 1:
        weights1[true1_isNuE] = weights1[true1_isNuE]/nue_files1
    if sum(true1_isMuon) > 1:
        weights1[true1_isMuon] = weights1[true_isMuon]/muon_files1
    if sum(nutau_mask_test1) > 1:
        weights1[true1_isNuTau] = weights1[true1_isNuTau]/nutau_files1
weights_squared1 = weights1*weights1

#Additional info
if info1 is not None:
    info1_prob_nu = info1[:,1]
    info1_coin_muon = info1[:,0]
    info1_true_ndoms = info1[:,2]
    info1_fit_success = info1[:,3]
    info1_noise_class = info1[:,4]
    info1_nhit_doms = info1[:,5]
    info1_n_top15 = info1[:,6]
    info1_n_outer = info1[:,7]
    info1_prob_nu2 = info1[:,8]
    info1_hits8 = info1[:,9]

#INFO masks
if info1 is not None:
    mask1Hits8 = info1_hits8 == 1
    mask1Nu = info1_prob_nu > 0.4
    mask1Noise = info1_noise_class > 0.95
    mask1nhit = info1_nhit_doms > 2.5
    mask1ntop = info1_n_top15 < 2.5
    mask1nouter = info1_n_outer < 7.5
    mask1Hits = np.logical_and(np.logical_and(mask1nhit, mask1ntop), mask1nouter)
    mask1Class = np.logical_and(mask1Nu,mask1Noise)
    mask1MC = np.logical_and(mask1Hits,mask1Class)

#IMPORT FILE 2
f2 = h5py.File(input_file2, "r")
truth2 = f2["Y_test_use"][:]
predict2 = f2["Y_predicted"][:]
#reco2 = f2["reco_test"][:]
raw_weights2 = f2["weights_test"][:]
try:
    info2 = f2["additional_info"][:]
except: 
    info2 = None
f2.close()
del f2

numu_files2 = 391
nue_files2 = 1
muon_files2 = 1
nutau_files2 = 1

#Truth
true2_energy = np.array(truth2[:,0])
true2_em_equiv_energy = np.array(truth2[:,14])
true2_x = np.array(truth2[:,4])
true2_y = np.array(truth2[:,5])
true2_z = np.array(truth2[:,6])
x2_origin = np.ones((len(true2_x)))*46.290000915527344
y2_origin = np.ones((len(true2_y)))*-34.880001068115234
true2_r = np.sqrt( (true2_x - x2_origin)**2 + (true2_y - y2_origin)**2 )
true2_isCC = np.array(truth2[:,11])
true2_isCC = np.array(true2_isCC,dtype=bool)
true2_isTrack = np.array(truth2[:,8])
true2_PID = truth2[:,9]

true2_zenith = np.array(truth2[:,12])
true2_coszenith = np.cos(np.array(truth2[:,12]))

#Reconstructed values (CNN)
reco2_energy = np.array(predict2[:,0])
reco2_prob_track = np.array(predict2[:,1])
reco2_zenith = np.array(predict2[:,2])
reco2_coszenith = np.cos(reco2_zenith)
reco2_x = np.array(predict2[:,3])
reco2_y = np.array(predict2[:,4])
reco2_z = np.array(predict2[:,5])
reco2_r = np.sqrt( (reco2_x - x2_origin)**2 + (reco2_y - y2_origin)**2 )
reco2_prob_mu = np.array(predict2[:,6])
reco2_nDOMs = np.array(predict2[:,7])

#PID identification
muon_mask_test2 = (true2_PID) == 13
true2_isMuon = np.array(muon_mask_test2,dtype=bool)
numu_mask_test2 = (true2_PID) == 14
true2_isNuMu = np.array(numu_mask_test2,dtype=bool)
nue_mask_test2 = (true2_PID) == 12
true2_isNuE = np.array(nue_mask_test2,dtype=bool)
nutau_mask_test2 = (true2_PID) == 16
true2_isNuTau = np.array(nutau_mask_test2,dtype=bool)
nu_mask2 = np.logical_or(np.logical_or(numu_mask_test2, nue_mask_test2), nutau_mask_test2)
true2_isNu = np.array(nu_mask2,dtype=bool)

#Weight adjustments
weights2 = raw_weights2[:,8]
if weights2 is not None:
    if sum(true2_isNuMu) > 1:
        weights2[true2_isNuMu] = weights2[true2_isNuMu]/numu_files2
    if sum(true2_isNuE) > 1:
        weights2[true2_isNuE] = weights2[true2_isNuE]/nue_files2
    if sum(true2_isMuon) > 1:
        weights2[true2_isMuon] = weights2[true_isMuon]/muon_files2
    if sum(nutau_mask_test1) > 1:
        weights2[true2_isNuTau] = weights2[true2_isNuTau]/nutau_files2
weights_squared2 = weights2*weights2

#Additional info
if info2 is not None:
    info2_prob_nu = info2[:,1]
    info2_coin_muon = info2[:,0]
    info2_true_ndoms = info2[:,2]
    info2_fit_success = info2[:,3]
    info2_noise_class = info2[:,4]
    info2_nhit_doms = info2[:,5]
    info2_n_top15 = info2[:,6]
    info2_n_outer = info2[:,7]
    info2_prob_nu2 = info2[:,8]
    info2_hits8 = info2[:,9]

#Print Summary of the two files
print("Events file 1: %i, NuMu Rate: %.2e"%(len(true1_energy),sum(weights1[true1_isNuMu])))
print("Events file 2: %i, NuMu Rate: %.2e"%(len(true2_energy),sum(weights2[true2_isNuMu])))

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

mask1 = true1_isCC #np.logical_and(c1,c1_2)
mask2 = true2_isCC #np.logical_and(c2,c2_2)
save_base_name = save_folder_name
minval = 1
maxval = 200
bins = 100
syst_bin = 100
name1 = "Old Energy"
name2 = "New Energy"
units = "(GeV)"

print(true1_energy[mask1][:10], reco1_energy[mask1][:10])
print(true2_energy[mask2][:10], reco2_energy[mask2][:10])
"""
switch = False
plot_2D_prediction(true_energy[cuts], cnn_energy[cuts],weights=true_weights,\
                        save=save, savefolder=save_folder_name,bins=bins, switch_axis=switch,
                        variable=plot_name, units=plot_units, reco_name=name1)
plot_2D_prediction(true_energy2[cuts2], cnn_energy2[cuts2], weights=true_weights2,
                        save=save, savefolder=save_folder_name,bins=bins,switch_axis=switch,\
                        variable=plot_name, units=plot_units, reco_name=name2)
plot_2D_prediction(true_energy[cuts], cnn_energy[cuts],weights=true_weights,\
                        save=save, savefolder=save_folder_name,bins=bins,switch_axis=switch,\
                        minval=minval, maxval=maxval, axis_square=True,\
                        variable=plot_name, units=plot_units, reco_name=name1)
plot_2D_prediction(true_energy2[cuts2], cnn_energy2[cuts2], weights=true_weights2,
                        save=save, savefolder=save_folder_name,bins=bins,switch_axis=switch,\
                        minval=minval, maxval=maxval, axis_square=True,\
                        variable=plot_name, units=plot_units, reco_name=name2)
switch = True
plot_2D_prediction(true_energy[cuts], cnn_energy[cuts],weights=true_weights,\
                        save=save, savefolder=save_folder_name,bins=bins, switch_axis=switch,
                        variable=plot_name, units=plot_units, reco_name=name1)
plot_2D_prediction(true_energy2[cuts2], cnn_energy2[cuts2], weights=true_weights2,
                        save=save, savefolder=save_folder_name,bins=bins,switch_axis=switch,\
                        variable=plot_name, units=plot_units, reco_name=name2)
plot_2D_prediction(true_energy[cuts], cnn_energy[cuts],weights=true_weights,\
                        save=save, savefolder=save_folder_name,bins=bins,switch_axis=switch,\
                        minval=minval, maxval=maxval, cut_truth=True, axis_square=True,\
                        variable=plot_name, units=plot_units, reco_name=name1)
plot_2D_prediction(true_energy2[cuts2], cnn_energy2[cuts2], weights=true_weights2,
                        save=save, savefolder=save_folder_name,bins=bins,switch_axis=switch,\
                        minval=minval, maxval=maxval, cut_truth=True, axis_square=True,\
                        variable=plot_name, units=plot_units, reco_name=name2)

plot_single_resolution(true_energy[cuts], cnn_energy[cuts], weights=true_weights,old_reco_weights=true_weights2,\
                   use_old_reco = True, old_reco = cnn_energy2[cuts2], old_reco_truth=true_energy2[cuts2],\
                   minaxis=-maxval, maxaxis=maxval, bins=bins,\
                   save=save, savefolder=save_folder_name,\
                   variable=plot_name, units=plot_units, reco_name=name2)
plot_single_resolution(true_energy[cuts], cnn_energy[cuts], weights=true_weights,old_reco_weights=true_weights2,\
                   use_old_reco = True, old_reco = cnn_energy2[cuts2], old_reco_truth=true_energy2[cuts2],\
                   minaxis=-2., maxaxis=2, bins=bins,use_fraction=True,\
                   save=save, savefolder=save_folder_name,\
                   variable=plot_name, units=plot_units, reco_name=name2)

plot_bin_slices(true_energy[cuts], cnn_energy[cuts], weights=true_weights, old_reco_weights=true_weights2,\
                    old_reco = cnn_energy2[cuts2],old_reco_truth=true_energy2[cuts2],\
                    use_fraction = True, bins=syst_bin, min_val=minval, max_val=maxval,\
                    save=save, savefolder=save_folder_name,\
                    variable=plot_name, units=plot_units, cnn_name=name1, reco_name=name2)
"""
plot_bin_slices(true1_energy[mask1], reco1_energy[mask1], 
                old_reco = reco2_energy[mask2],old_reco_truth=true2_energy[mask2],
                weights=weights1[mask1], old_reco_weights=weights2[mask2],\
                use_fraction = True, bins=syst_bin, 
                min_val=minval, max_val=maxval,\
                save=save, savefolder=save_folder_name,
                variable=plot_name, units=plot_units, 
                cnn_name=name1, reco_name=name2)
"""
reco_nan = np.isnan(retro_energy)
not_nan = np.logical_not(reco_nan)
assert sum(not_nan) > 0, "Retro is all nans"
cuts = np.logical_and(cuts,not_nan)
reco_nan2 = np.isnan(retro_energy2)
not_nan2 = np.logical_not(reco_nan2)
assert sum(not_nan2) > 0, "Retro is all nans"
cuts2 = np.logical_and(cuts2,not_nan2)
plot_bin_slices(true_energy[cuts], retro_energy[cuts], weights=weights[cuts], old_reco_weights=weights2[cuts2],\

                    old_reco = retro_energy2[cuts2],old_reco_truth=true_energy2[cuts2],\
                    use_fraction = True, bins=syst_bin, min_val=minval, max_val=maxval,\
                    save=save, savefolder=save_folder_name,\
                    variable=plot_name, units=plot_units, cnn_name=name1, reco_name=name2)
"""

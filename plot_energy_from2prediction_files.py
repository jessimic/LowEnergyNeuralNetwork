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
parser.add_argument("--compare_cnn", default=False,action='store_true',
                        dest='compare_cnn',help="compare to CNN")
args = parser.parse_args()

input_file = args.input_file
input_file2 = args.input_file2
compare_cnn = args.compare_cnn
save_folder_name = args.output_dir + "/"
if args.savename is not None:
    save_folder_name += args.savename + "/"
    if os.path.isdir(save_folder_name) != True:
            os.mkdir(save_folder_name)
print("Saving to %s"%save_folder_name)

#CUT values
r_cut1 = 165
zmin_cut1 = -495
zmax_cut1 = -225
coszen_cut1 = 0.3
emin_cut1 = 5
emax_cut1 = 100
mu_cut1 = 0.01
nDOM_cut1 = 8

r_cut2 = 165
zmin_cut2 = -495
zmax_cut2 = -225
coszen_cut2 = 0.3
emin_cut2 = 5
emax_cut2 = 100
mu_cut2 = 0.01
nDOM_cut2 = 8

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

#RECO masks
mask1Energy = np.logical_and(reco1_energy > emin_cut1, reco1_energy < emax_cut1)
mask1Zenith = reco1_coszenith <= coszen_cut1
mask1R = reco1_r < r_cut1
mask1Z = np.logical_and(reco1_z > zmin_cut1, reco1_z < zmax_cut1)
mask1Vertex = np.logical_and(mask1R, mask1Z)
mask1ProbMu = reco1_prob_mu <= mu_cut1
mask1Reco = np.logical_and(mask1ProbMu, np.logical_and(mask1Zenith, np.logical_and(mask1Energy, mask1Vertex)))
mask1RecoNoEn = np.logical_and(mask1ProbMu, np.logical_and(mask1Zenith, mask1Vertex))
mask1DOM = reco1_nDOMs >= nDOM_cut1

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

#Combined Masks
mask1Analysis = np.logical_and(mask1MC, mask1Reco)

#IMPORT FILE 2
f2 = h5py.File(input_file2, "r")
truth2 = f2["Y_test_use"][:]
predict2 = f2["Y_predicted"][:]
raw_weights2 = f2["weights_test"][:]
try:
    reco2 = f2["reco_test"][:]
except:
    reco2 = None
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

if compare_cnn:
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

    #RECO masks
    mask2Energy = np.logical_and(reco2_energy > emin_cut2, reco2_energy < emax_cut2)
    mask2Zenith = reco2_coszenith <= coszen_cut2
    mask2R = reco2_r < r_cut2
    mask2Z = np.logical_and(reco2_z > zmin_cut2, reco2_z < zmax_cut2)
    mask2Vertex = np.logical_and(mask2R, mask2Z)
    mask2ProbMu = reco2_prob_mu <= mu_cut2
    mask2Reco = np.logical_and(mask2ProbMu, np.logical_and(mask2Zenith, np.logical_and(mask2Energy, mask2Vertex)))
    mask2RecoNoEn = np.logical_and(mask2ProbMu, np.logical_and(mask2Zenith, mask2Vertex))
    mask2DOM = reco2_nDOMs >= nDOM_cut2

else:
    reco2_energy = np.array(reco2[:,0])
    reco2_zenith = np.array(reco2[:,1])
    reco2_coszenith = np.cos(reco2_zenith)
    reco2_time = np.array(reco2[:,3])
    reco2_prob_track = np.array(reco2[:,13])
    reco2_prob_track_full = np.array(reco2[:,12])
    reco2_x = np.array(reco2[:,4])
    reco2_y = np.array(reco2[:,5])
    reco2_z = np.array(reco2[:,6])
    reco2_r = np.sqrt( (reco2_x - x2_origin)**2 + (reco2_y - y2_origin)**2 )
    reco2_iterations = np.array(reco2[:,14])
    reco2_nan = np.isnan(reco2_energy)
    
    mask2Energy = np.logical_and(reco2_energy > emin_cut2, reco2_energy < emax_cut2)
    mask2Zenith = reco2_coszenith <= coszen_cut2
    mask2R = reco2_r < r_cut2
    mask2Z = np.logical_and(reco2_z > zmin_cut2, reco2_z < zmax_cut2)
    mask2Vertex = np.logical_and(mask2R, mask2Z)
    mask2ProbMu = reco2_prob_mu <= mu_cut2
    mask2Reco = np.logical_and(mask2ProbMu, np.logical_and(mask2Zenith, np.logical_and(mask2Energy, mask2Vertex)))
    mask2RecoNoEn = np.logical_and(mask2ProbMu, np.logical_and(mask2Zenith, mask2Vertex))
    mask2DOM = reco2_nDOMs >= nDOM_cut2

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


#INFO masks
if info2 is not None:
    mask2Hits8 = info2_hits8 == 1
    mask2Nu = info2_prob_nu > 0.4
    mask2Noise = info2_noise_class > 0.95
    mask2nhit = info2_nhit_doms > 2.5
    mask2ntop = info2_n_top15 < 2.5
    mask2nouter = info2_n_outer < 7.5
    mask2Hits = np.logical_and(np.logical_and(mask2nhit, mask2ntop), mask2nouter)
    mask2Class = np.logical_and(mask2Nu,mask2Noise)
    mask2MC = np.logical_and(mask2Hits,mask2Class)

    #Combined Masks
    mask2Analysis = np.logical_and(mask2MC, mask2Reco)

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

ana_mask1 = np.logical_and(true1_isCC, mask1Reco)
ana_mask2 = np.logical_and(true2_isCC, mask2Reco)
mask1 = np.logical_and(np.logical_and(true1_isCC, mask1RecoNoEn),mask1DOM)
mask2 = np.logical_and(np.logical_and(true2_isCC, mask2RecoNoEn),mask2DOM)
save_base_name = save_folder_name
minval = 5
maxval = 100
logmax = 10**1.5
bins_log = 10**np.linspace(0,1.5,100)
bins = 95
syst_bin = 95
name1 = "Old Energy"
name2 = "New Energy"
units = "(GeV)"

print(sum(weights1[mask1])/sum(weights1[true1_isCC]), sum(weights1[mask1])/sum(weights1[ana_mask1]))
print(true1_energy[mask1][:10], reco1_energy[mask1][:10])
print(sum(weights2[mask2])/sum(weights2[true2_isCC]), sum(weights2[mask2])/sum(weights2[ana_mask2]))
print(true2_energy[mask2][:10], reco2_energy[mask2][:10])

path=save_folder_name
plt.figure(figsize=(10,7))
plt.hist(true1_energy[mask1], color="green",label="true",
         bins=bins_log,range=[minval,logmax],
         weights=weights1[mask1],alpha=0.5)
plt.hist(reco1_energy[mask1], color="blue",label="CNN",
         bins=bins_log,range=[minval,logmax],
         weights=weights1[mask1],alpha=0.5)
plt.xscale('log')
plt.title("Energy Distribution Weighted for %s events"%len(true1_energy),fontsize=25)
plt.xlabel("Energy (GeV)",fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.axvline(5,linewidth=3,linestyle="--",color='k',label="Cut at 5 GeV")
plt.legend(loc='upper left',fontsize=15)
plt.savefig("%s/%sLogEnergyDist_ZoomInLE.png"%(path,name1.replace(" ","")))

plt.figure(figsize=(10,7))
plt.hist(true1_energy[mask1], label="true",bins=100,
        range=[minval,maxval],weights=weights1[mask1],alpha=0.5)
plt.hist(reco1_energy[mask1], label=name1,bins=100,
        range=[minval,maxval],weights=weights1[mask1],alpha=0.5)
plt.title("Energy Distribution Weighted for %s events"%len(true1_energy),fontsize=25)
plt.xlabel("Energy (GeV)",fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.axvline(5,linewidth=3,linestyle="--",color='k',label="Cut at 5 GeV")
plt.legend(fontsize=15)
plt.savefig("%s/%sEnergyDist.png"%(path,name1.replace(" ","")))

plt.figure(figsize=(10,7))
plt.hist(true2_energy[mask2], color="green",label="true",
         bins=bins_log,range=[minval,logmax],
         weights=weights2[mask2],alpha=0.5)
plt.hist(reco2_energy[mask2], color="blue",label="CNN",
         bins=bins_log,range=[minval,logmax],
         weights=weights2[mask2],alpha=0.5)
plt.xscale('log')
plt.title("Energy Distribution Weighted for %s events"%len(true2_energy),fontsize=25)
plt.xlabel("Energy (GeV)",fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.axvline(5,linewidth=3,linestyle="--",color='k',label="Cut at 5 GeV")
plt.legend(loc='upper left',fontsize=15)
plt.savefig("%s/%sLogEnergyDist_ZoomInLE.png"%(path,name2.replace(" ","")))

plt.figure(figsize=(10,7))
plt.hist(true2_energy[mask2], label="true",bins=100,
        range=[minval,maxval],weights=weights2[mask2],alpha=0.5)
plt.hist(reco2_energy[mask2], label=name2,bins=100,
        range=[minval,maxval],weights=weights2[mask2],alpha=0.5)
plt.title("Energy Distribution Weighted for %s events"%len(true2_energy),fontsize=25)
plt.xlabel("Energy (GeV)",fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.axvline(5,linewidth=3,linestyle="--",color='k',label="Cut at 5 GeV")
plt.legend(fontsize=15)
plt.savefig("%s/%sEnergyDist.png"%(path,name2.replace(" ","")))

switch = False
plot_2D_prediction(true1_energy[mask1], reco1_energy[mask1],
                    weights=weights1[mask1],\
                    save=save, savefolder=save_folder_name,
                    bins=bins, switch_axis=switch,
                    variable=plot_name, units=plot_units, reco_name=name1)

plot_2D_prediction(true2_energy[mask2], reco2_energy[mask2],
                    weights=weights2[mask2],
                    save=save, savefolder=save_folder_name,
                    bins=bins,switch_axis=switch,\
                    variable=plot_name, units=plot_units, reco_name=name2)

plot_2D_prediction(true1_energy[mask1], reco1_energy[mask1],
                    weights=weights1[mask1],\
                    save=save, savefolder=save_folder_name,
                    bins=bins,switch_axis=switch,\
                    minval=minval, maxval=maxval, axis_square=True,\
                    variable=plot_name, units=plot_units, reco_name=name1)

plot_2D_prediction(true2_energy[mask2], reco2_energy[mask2],
                    weights=weights2[mask2],
                    save=save, savefolder=save_folder_name,
                    bins=bins,switch_axis=switch,\
                    minval=minval, maxval=maxval, axis_square=True,\
                    variable=plot_name, units=plot_units, reco_name=name2)

switch = True
plot_2D_prediction(true1_energy[mask1], reco1_energy[mask1],
                    weights=weights1[mask1],\
                    save=save, savefolder=save_folder_name,
                    bins=bins, switch_axis=switch,
                    variable=plot_name, units=plot_units, reco_name=name1)

plot_2D_prediction(true2_energy[mask2], reco2_energy[mask2],
                    weights=weights2[mask2],
                    save=save, savefolder=save_folder_name,
                    bins=bins,switch_axis=switch,\
                    variable=plot_name, units=plot_units, reco_name=name2)

plot_2D_prediction(true1_energy[mask1], reco1_energy[mask1],
                    weights=weights1[mask1],\
                    save=save, savefolder=save_folder_name,
                    bins=bins,switch_axis=switch,\
                    minval=minval, maxval=maxval,
                    cut_truth=True, axis_square=True,\
                    variable=plot_name, units=plot_units, reco_name=name1)

plot_2D_prediction(true2_energy[mask2], reco2_energy[mask2],
                    weights=weights2[mask2],
                    save=save, savefolder=save_folder_name,
                    bins=bins,switch_axis=switch,\
                    minval=minval, maxval=maxval,
                    cut_truth=True, axis_square=True,\
                    variable=plot_name, units=plot_units, reco_name=name2)

#Resolution
plot_single_resolution(true1_energy[mask1], reco1_energy[mask1], 
                   weights=weights1[mask1],old_reco_weights=weights2[mask2],
                   use_old_reco = True, old_reco = reco2_energy[mask2],
                   old_reco_truth=true2_energy[mask2],\
                   minaxis=-maxval, maxaxis=maxval, bins=bins,\
                   save=save, savefolder=save_folder_name,\
                   variable=plot_name, units=plot_units, reco_name=name2)

plot_single_resolution(true1_energy[mask1], reco1_energy[mask1],
                    weights=weights1[mask1],old_reco_weights=weights2[mask2],\
                    use_old_reco = True, old_reco = reco2_energy[mask2],
                    old_reco_truth=true2_energy[mask2],\
                    minaxis=-2., maxaxis=2, bins=bins, use_fraction=True,\
                    save=save, savefolder=save_folder_name,\
                    variable=plot_name, units=plot_units, reco_name=name2)

#Bin Slices
plot_bin_slices(true1_energy[mask1], reco1_energy[mask1], 
                old_reco = reco2_energy[mask2],old_reco_truth=true2_energy[mask2],
                weights=weights1[mask1], old_reco_weights=weights2[mask2],\
                use_fraction = True, bins=syst_bin, 
                min_val=minval, max_val=maxval,\
                save=save, savefolder=save_folder_name,
                variable=plot_name, units=plot_units, 
                cnn_name=name1, reco_name=name2)

plot_bin_slices(true1_energy[mask1], reco1_energy[mask1], 
                energy_truth=true1_energy[mask1],
                old_reco = reco2_energy[mask2],old_reco_truth=true2_energy[mask2],
                reco_energy_truth = true2_energy[mask2],
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

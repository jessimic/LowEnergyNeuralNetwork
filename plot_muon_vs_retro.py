#!/usr/bin/env python

############################################
# Scripts for plotting functions
# Contains functions:
##############################################

import numpy as np
import h5py
import os, sys
import argparse
import matplotlib
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
matplotlib.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score, recall_score
from sklearn.metrics import confusion_matrix
from PlottingFunctions import plot_bin_slices

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input",type=str,default=None,
                    dest="input_file", help="path and name of the input file")
parser.add_argument("-o", "--outdir",type=str,default='/mnt/home/micall12/LowEnergyNeuralNetwork/output_plots/',
                    dest="output_dir", help="path of ouput file")
parser.add_argument("--efactor",type=float,default=1.,
                    dest="efactor", help="change if you want to multiply true energy by a factor (if transformed)")
parser.add_argument("--numu",type=float,default=1518.,
                    dest="numu", help="number of numu files")
parser.add_argument("--nue",type=float,default=602.,
                    dest="nue", help="number of nue files")
parser.add_argument("--nutau",type=float,default=334.,
                    dest="nutau", help="number of nue files")
parser.add_argument("--muon",type=float,default=19991.,
                    dest="muon", help="number of muongun files")
parser.add_argument("--no_old_reco", default=False,action='store_true',
                        dest='no_old_reco',help="no old reco")
parser.add_argument("--no_weights", default=False,action='store_true',
                        dest='no_weights',help="don't use weights")
args = parser.parse_args()

input_file = args.input_file
efactor = args.efactor
numu_files = args.numu
nue_files = args.nue
nutau_files = args.nutau
muon_files = args.muon
no_old_reco = args.no_old_reco
no_weights = args.no_weights
save_folder = args.output_dir
save = True

plot_nu_resolution = True
plot_energy = False
plot_main = True
compare_resolution = False
hists1d = False
print_rates = False
print_retro_rates = False
plot_rates = False
plot_after_threshold = False
plot_vertex = False
plot_coszen = False
do_energy_range = False
do_energy_auc = False
do_energy_cut_plots = False

f = h5py.File(input_file, "r")
truth = f["Y_test_use"][:]
predict = f["Y_predicted"][:]
try:
    info = f["additional_info"][:]
except:
    info = None
if no_old_reco:
    reco = None
else:
    try:
        reco = f["reco_test"][:]
    except:
        reco = None
try:
    weights = f["weights_test"][:]
except:
    weights = None
f.close()
del f

fit_success = info[:,3]
fit_success =  np.array(fit_success,dtype=bool)
hits_gt_8 = info[:,9]
hits_gt_8 =  np.array(hits_gt_8,dtype=bool)
#assert sum(fit_success)==len(fit_success), "there are some fit failures here"
assert sum(hits_gt_8)==len(hits_gt_8), "there are events with < 8 hits"

# energy zenith prob_prob_track vertex_x vertex_y vertex_z prob_muon
cnn_prob_mu = np.array(predict[:,-1])
cnn_prob_nu = np.ones(len(cnn_prob_mu)) - cnn_prob_mu
cnn_zenith = np.array(predict[:,1])
cnn_prob_track = np.array(predict[:,2])
cnn_x = np.array(predict[:,3])
cnn_y = np.array(predict[:,4])
cnn_z = np.array(predict[:,5])
cnn_coszen = np.cos(cnn_zenith)
cnn_energy = np.array(predict[:,0])

retro_prob_nu = np.array(info[:,1]) # upgoing
retro_zenith = np.array(reco[:,1])
retro_prob_track = np.array(reco[:,12]) #upgoing
retro_x = np.array(reco[:,4])
retro_y = np.array(reco[:,5])
retro_z = np.array(reco[:,6])
retro_coszen = np.cos(retro_zenith)
retro_energy = np.array(reco[:,0])

true_energy = np.array(truth[:,0])*efactor
true_zenith = np.array(truth[:,12])
true_coszenith = np.cos(np.array(truth[:,12]))
true_x = np.array(truth[:,4])
true_y = np.array(truth[:,5])
true_z = np.array(truth[:,6])
true_track = np.array(truth[:,8])
true_PID = np.array(truth[:,9])
true_isTrack = np.array(true_track,dtype=bool)
true_CC = np.array(truth[:,11])
true_isCC = true_CC == 1

noise_class = info[:,4]
nhit_doms = info[:,5]
ntop = info[:,6]
nouter = info[:,7]

x_origin = np.ones((len(true_x)))*46.290000915527344
y_origin = np.ones((len(true_y)))*-34.880001068115234
true_r = np.sqrt( (true_x - x_origin)**2 + (true_y - y_origin)**2 )
cnn_r = np.sqrt( (cnn_x - x_origin)**2 + (cnn_y - y_origin)**2 )
retro_r = np.sqrt( (retro_x - x_origin)**2 + (retro_y - y_origin)**2 )

muon_mask_test = (true_PID) == 13
true_isMuon = np.array(muon_mask_test,dtype=bool)
numu_mask_test = (true_PID) == 14
true_isNuMu = np.array(numu_mask_test,dtype=bool)
nue_mask_test = (true_PID) == 12
true_isNuE = np.array(nue_mask_test,dtype=bool)
nutau_mask_test = (true_PID) == 16
true_isNuTau = np.array(nutau_mask_test,dtype=bool)
nu_mask = np.logical_or(np.logical_or(numu_mask_test, nue_mask_test), nutau_mask_test)
true_isNu = np.array(nu_mask,dtype=bool)
assert sum(true_isNu)!=len(true_isNu), "All events saved as true neutrino, no true muon saved"
assert sum(true_isMuon)>0, "No true muon saved"

if weights is not None:
    weights = weights[:,8]
    if sum(true_isNuMu) > 1:
        #print("NuMu:",sum(true_isNuMu),sum(weights[true_isNuMu]))
        weights[true_isNuMu] = weights[true_isNuMu]/numu_files
        #print(sum(weights[true_isNuMu]))
    if sum(true_isNuE) > 1:
        #print("NuE:",sum(true_isNuE),sum(weights[true_isNuE]))
        weights[true_isNuE] = weights[true_isNuE]/nue_files
        #print(sum(weights[true_isNuE]))
    if sum(muon_mask_test) > 1:
        #print("Muon:",sum(true_isMuon),sum(weights[true_isMuon]))
        weights[true_isMuon] = weights[true_isMuon]/muon_files
        #print(sum(weights[true_isMuon]))
    if sum(nutau_mask_test) > 1:
        #print("NuTau:",sum(true_isNuTau),sum(weights[true_isNuTau]))
        weights[true_isNuTau] = weights[true_isNuTau]/nutau_files
        #print(sum(weights[true_isNuTau]))
if no_weights:
    weights = np.ones(truth.shape[0])
    print("NOT USING ANY WEIGHTING")

# Cuts
z_cut_max = -225
z_cut_min = -495
r_cut_val = 165
no_cut = true_energy > 0
mask_upgoing = cnn_coszen < 0.3
mask_upgoing_retro = retro_coszen < 0.3
#retro_prob_nu_range = np.logical_and(retro_prob_nu >= 0, retro_prob_nu <= 1)
r_cut = cnn_r < r_cut_val
z_cut = np.logical_and(cnn_z > z_cut_min, cnn_z < z_cut_max)
vertex_cut = np.logical_and(r_cut, z_cut)

cnn_mu_cut = 0.45
cnn_nu_cut = 1 - cnn_mu_cut
retro_nu_cut = .3
cnn_nu = cnn_prob_mu <= cnn_mu_cut
cnn_mu = cnn_prob_mu > cnn_mu_cut
retro_nu = retro_prob_nu >= retro_nu_cut
retro_mu = retro_prob_nu < retro_nu_cut

noise_cut = noise_class > 0.95
nhits_cut = nhit_doms >= 3
ntop_cut = ntop < 3
nouter_cut = nouter < 8

#cnn_nu = cnn_prob_mu < 0.66
#retro_nu = retro_prob_nu > 0.3
cnn_mask = mask_upgoing_retro & fit_success #retro_prob_nu_range
retro_mask = mask_upgoing_retro & fit_success #retro_prob_nu_range

from PlottingFunctionsClassification import ROC
from PlottingFunctionsClassification import plot_classification_hist
from PlottingFunctionsClassification import confusion_matrix
from PlottingFunctions import plot_2D_prediction

mask_here = mask_upgoing_retro
mask_here_retro = mask_upgoing_retro
mask_name_here=""
if no_weights:
    mask_name_here += "_NoWeights"



print("Rates:\t Mu\t NuE\t NuMu\t NuTau\t Nu\n")
print("CNN:\t %.2e\t %.2e\t %.2e\t %.2e\t %.2e\t"%(sum(weights[true_isMuon&cnn_mask]),sum(weights[true_isNuE&cnn_mask]),sum(weights[true_isNuMu&cnn_mask]),sum(weights[true_isNuTau&cnn_mask]),sum(weights[true_isNu&cnn_mask])))
print("Retro:\t %.2e\t %.2e\t %.2e\t %.2e\t %.2e\t"%(sum(weights[true_isMuon&retro_mask]),sum(weights[true_isNuE&retro_mask]),sum(weights[true_isNuMu&retro_mask]),sum(weights[true_isNuTau&retro_mask]),sum(weights[true_isNu&retro_mask])))

print("Rates After NHit Cut:\t Mu\t NuE\t NuMu\t NuTau\t Nu\n")
quick_mask = retro_mask & ntop_cut
print("nTop3:\t %.2e\t %.2e\t %.2e\t %.2e\t %.2e\t"%(sum(weights[true_isMuon&quick_mask]),sum(weights[true_isNuE&quick_mask]),sum(weights[true_isNuMu&quick_mask]),sum(weights[true_isNuTau&quick_mask]),sum(weights[true_isNu&quick_mask])))
quick_mask= retro_mask & nouter_cut
print("nOuter8:\t %.2e\t %.2e\t %.2e\t %.2e\t %.2e\t"%(sum(weights[true_isMuon&quick_mask]),sum(weights[true_isNuE&quick_mask]),sum(weights[true_isNuMu&quick_mask]),sum(weights[true_isNuTau&quick_mask]),sum(weights[true_isNu&quick_mask])))

print("Rates After Cut:\t Mu\t NuE\t NuMu\t NuTau\t Nu\n")
print("CNN:\t %.2e\t %.2e\t %.2e\t %.2e\t %.2e\t"%(sum(weights[true_isMuon&cnn_mask&cnn_nu]),sum(weights[true_isNuE&cnn_mask&cnn_nu]),sum(weights[true_isNuMu&cnn_mask&cnn_nu]),sum(weights[true_isNuTau&cnn_mask&cnn_nu]),sum(weights[true_isNu&cnn_mask&cnn_nu])))
print("Retro:\t %.2e\t %.2e\t %.2e\t %.2e\t %.2e\t"%(sum(weights[true_isMuon&retro_mask&retro_nu]),sum(weights[true_isNuE&retro_mask&retro_nu]),sum(weights[true_isNuMu&retro_mask&retro_nu]),sum(weights[true_isNuTau&retro_mask&retro_nu]),sum(weights[true_isNu&retro_mask&retro_nu])))
print("fit success muon:",sum(true_isMuon[fit_success]))

minval = 0
maxval = 15
bins=15
plt.figure(figsize=(10,7))
plt.hist(nouter[true_isMuon],range=[minval,maxval],bins=bins,weights=weights[true_isMuon],label=r'$\mu$',alpha=0.5)
plt.hist(nouter[true_isNuMu],range=[minval,maxval],bins=bins,weights=weights[true_isNuMu],label=r'$\nu_\mu$',alpha=0.5)
plt.hist(nouter[true_isNuE],range=[minval,maxval],bins=bins,weights=weights[true_isNuE],label=r'$\nu_e$',alpha=0.5)
plt.hist(nouter[true_isNuTau],range=[minval,maxval],bins=bins,weights=weights[true_isNuTau],label=r'$\nu_\tau$',alpha=0.5)
#plt.yscale('log')
plt.ylabel("Rate (Hz)",fontsize=20)
plt.xlabel("Number DOMs Hit",fontsize=20)
plt.title("DOMs Hit in Outermost IceCube Strings",fontsize=25)
plt.legend()
plt.savefig("%sNOuterHist.png"%(save_folder),bbox_inches='tight')
plt.close()


plt.figure(figsize=(10,7))
plt.hist(ntop[true_isMuon],range=[minval,maxval],bins=bins,weights=weights[true_isMuon],label=r'$\mu$',alpha=0.5)
plt.hist(ntop[true_isNuMu],range=[minval,maxval],bins=bins,weights=weights[true_isNuMu],label=r'$\nu_\mu$',alpha=0.5)
plt.hist(ntop[true_isNuE],range=[minval,maxval],bins=bins,weights=weights[true_isNuE],label=r'$\nu_e$',alpha=0.5)
plt.hist(ntop[true_isNuTau],range=[minval,maxval],bins=bins,weights=weights[true_isNuTau],label=r'$\nu_\tau$',alpha=0.5)
#plt.yscale('log')
plt.ylabel("Rate (Hz)",fontsize=20)
plt.xlabel("Number DOMs Hit",fontsize=20)
plt.title("DOMs Hit in Top 15 Layers of IceCube",fontsize=25)
plt.legend()
plt.savefig("%sNTopHist.png"%(save_folder),bbox_inches='tight')
plt.close()

plt.figure(figsize=(10,7))
plt.hist(cnn_coszen[fit_success&true_isMuon],bins=50,weights=weights[fit_success&true_isMuon],label=r'Full $\mu$ Distribution',alpha=0.5,range=[-1,1])
plt.hist(cnn_coszen[cnn_nu&true_isMuon&fit_success],bins=50,weights=weights[cnn_nu&true_isMuon&fit_success],label=r'After Muon Cut',alpha=0.5,range=[-1,1])
#plt.yscale('log')
plt.ylabel("Rate (Hz)",fontsize=20)
plt.xlabel("Reconstructed Cosine Zenith",fontsize=20)
plt.title("FLERCNN Cosine Zenith for True Muon",fontsize=25)
plt.legend()
plt.savefig("%sCosZenithMuonDistHistCNN.png"%(save_folder),bbox_inches='tight')
plt.close()

plt.figure(figsize=(10,7))
plt.hist(retro_coszen[fit_success&true_isMuon],bins=50,weights=weights[fit_success&true_isMuon],label=r'Full $\mu$ Distribution',alpha=0.5,range=[-1,1])
plt.hist(retro_coszen[retro_nu&true_isMuon&fit_success],bins=50,weights=weights[retro_nu&true_isMuon&fit_success],label=r'After Muon Cut',alpha=0.5,range=[-1,1])
#plt.yscale('log')
plt.ylabel("Rate (Hz)",fontsize=20)
plt.xlabel("Reconstructed Cosine Zenith",fontsize=20)
plt.title("Retro Cosine Zenith for True Muon",fontsize=25)
plt.legend()
plt.savefig("%sCosZenithMuonDistHistRETRO.png"%(save_folder),bbox_inches='tight')
plt.close()

plt.figure(figsize=(10,7))
plt.hist(cnn_r[fit_success&true_isMuon]*cnn_r[fit_success&true_isMuon],bins=50,weights=weights[fit_success&true_isMuon],label=r'Full $\mu$ Distribution',alpha=0.5,range=[0,300*300])
plt.hist(cnn_r[cnn_nu&true_isMuon&fit_success]*cnn_r[cnn_nu&true_isMuon&fit_success],bins=50,weights=weights[cnn_nu&true_isMuon&fit_success],label=r'After Muon Cut',alpha=0.5,range=[0,300*300])
plt.axvline(125*125,color='k',linewidth=3,label="DeepCore")
#plt.yscale('log')
plt.ylim(0,0.00011)
plt.ylabel("Rate (Hz)",fontsize=20)
plt.xlabel("Reconstructed R^2 (m^2)",fontsize=20)
plt.title("FLERCNN R^2 for True Muon",fontsize=25)
plt.legend()
plt.savefig("%sR2MuonDistHistCNN.png"%(save_folder),bbox_inches='tight')
plt.close()

plt.figure(figsize=(10,7))
plt.hist(retro_r[fit_success&true_isMuon]*retro_r[fit_success&true_isMuon],bins=50,weights=weights[fit_success&true_isMuon],label=r'Full $\mu$ Distribution',alpha=0.5,range=[0,300*300])
plt.hist(retro_r[retro_nu&true_isMuon&fit_success]*retro_r[retro_nu&true_isMuon&fit_success],bins=50,weights=weights[retro_nu&true_isMuon&fit_success],label=r'After Muon Cut',alpha=0.5,range=[0,300*300])
plt.axvline(125*125,color='k',linewidth=3,label="DeepCore")
#plt.yscale('log')
plt.ylim(0,0.00011)
plt.ylabel("Rate (Hz)",fontsize=20)
plt.xlabel("Reconstructed R^2 (m^2)",fontsize=20)
plt.title("Retro R^2 for True Muon",fontsize=25)
plt.legend()
plt.savefig("%sR2MuonDistHistRETRO.png"%(save_folder),bbox_inches='tight')
plt.close()

r_large_retro = retro_r > r_cut_val
r_large_cnn = cnn_r > r_cut_val
print(sum(weights[cnn_nu&true_isMuon&fit_success&r_large_cnn]),sum(weights[retro_nu&true_isMuon&fit_success&r_large_retro]))

print("Whole hist: ", sum(weights[fit_success&true_isMuon]), "Post cnn cut: ",  sum(weights[cnn_nu&true_isMuon&fit_success]), "Post retro cut: ", sum(weights[retro_nu&true_isMuon&fit_success]))

"""
weights = weights[mask_here]
true_isMuon = true_isMuon[mask_here]
true_isNuMu = true_isNuMu[mask_here]
true_isNuE = true_isNuE[mask_here]
true_isNuTau = true_isNuTau[mask_here]
true_isNu = true_isNu[mask_here]
true_energy = true_energy[mask_here]
cnn_prob_mu = cnn_prob_mu[mask_here]
cnn_prob_track = cnn_prob_track[mask_here]
truth_PID = truth[:,9][mask_here]
true_r = true_r[mask_here]
true_z = true_z[mask_here]
cnn_r = cnn_r[mask_here]
cnn_z = cnn_z[mask_here]
cnn_coszen = cnn_coszen[mask_here]
cnn_energy = cnn_energy[mask_here]
retro_energy = retro_energy[mask_here]
retro_coszen = retro_coszen[mask_here]
retro_r = retro_r[mask_here]
retro_z = retro_z[mask_here]
retro_prob_nu = retro_prob_nu[mask_here]
retro_prob_track = retro_prob_track[mask_here]
"""


#save_folder += "/%s/"%mask_name_here
save_folder += "/Upgoing_L7RetroCompare_Oct/"
print("Working on %s"%save_folder)
if os.path.isdir(save_folder) != True:
    os.mkdir(save_folder)

if plot_nu_resolution:
    
    zmin_cut = -495
    zmax_cut = -225
    rmax_cut = 165

    variable_name = "Energy"
    units = "(GeV)"
    """
    a_mask = retro_mask & true_isNu & true_isCC
    flavor_name = "Nu"
    plot_bin_slices(true_energy[a_mask], cnn_energy[a_mask],
                            weights=weights[a_mask],old_reco = retro_energy[a_mask],
                            vs_predict = True,use_fraction = True,
                            specific_bins=[1,3,10,15,20,25,30,35,40,50,60,70,80,100,120,140,160,200],
                            min_val=minval, max_val=200,
                            save=save, savefolder=save_folder,\
                            variable=variable_name, units=units,
                            flavor=flavor_name,sample="CC")
    plot_bin_slices(true_energy[a_mask], cnn_energy[a_mask],
                            weights=weights[a_mask],old_reco = retro_energy[a_mask],
                            vs_predict = False,use_fraction = True,
                            specific_bins=[1,3,10,15,20,25,30,35,40,50,60,70,80,100,120,140,160,200],
                            min_val=minval, max_val=200,
                            save=save, savefolder=save_folder,\
                            variable=variable_name, units=units,
                            flavor=flavor_name,sample="CC")
    a_mask = retro_mask & true_isNuMu & true_isCC
    flavor_name = "NuMu"
    plot_bin_slices(true_energy[a_mask], cnn_energy[a_mask],
                            weights=weights[a_mask],old_reco = retro_energy[a_mask],
                            vs_predict = True,use_fraction = True,
                            specific_bins=[1,3,10,15,20,25,30,35,40,50,60,70,80,100,120,140,160,200],
                            min_val=minval, max_val=200,
                            save=save, savefolder=save_folder,\
                            variable=variable_name, units=units,
                            flavor=flavor_name,sample="CC")
    plot_bin_slices(true_energy[a_mask], cnn_energy[a_mask],
                            weights=weights[a_mask],old_reco = retro_energy[a_mask],
                            vs_predict = False,use_fraction = True,
                            specific_bins=[1,3,10,15,20,25,30,35,40,50,60,70,80,100,120,140,160,200],
                            min_val=minval, max_val=200,
                            save=save, savefolder=save_folder,\
                            variable=variable_name, units=units,
                            flavor=flavor_name,sample="CC")
    """

    a_mask_here = true_isNuMu & true_isCC
    a_flavor_here = "NuMu"
    bins_here = 200
    maxval_here = 200
    plot_2D_prediction(true_energy[a_mask_here],
                                cnn_energy[a_mask_here],
                                weights=weights[a_mask_here],
                                save=save, savefolder=save_folder,
                                bins=100,
                                variable="Energy", units="(GeV)",
                                reco_name="CNN",flavor=a_flavor_here,sample="CC")
    plot_2D_prediction(true_energy[a_mask_here],
                                cnn_energy[a_mask_here],
                                weights=weights[a_mask_here],
                                save=save, savefolder=save_folder,
                                bins=bins_here,minval=0,maxval=maxval_here,axis_square=True,
                                variable="Energy", units="(GeV)",
                                reco_name="CNN",flavor=a_flavor_here,sample="CC")
    plot_2D_prediction(true_energy[a_mask_here],
                                retro_energy[a_mask_here],
                                weights=weights[a_mask_here],
                                save=save, savefolder=save_folder,
                                bins=bins_here,minval=0,maxval=maxval_here,axis_square=True,
                                variable="Energy", units="(GeV)",
                                reco_name="Retro",flavor=a_flavor_here,sample="CC")
    plot_bin_slices(true_energy[a_mask_here],
                            cnn_energy[a_mask_here],
                            old_reco = retro_energy[a_mask_here],
                            weights=weights[a_mask_here],
                            vs_predict = True,
                            use_fraction = True, bins=20,
                            min_val=1, max_val=maxval_here,\
                            save=save, savefolder=save_folder,\
                            variable="Energy", units="(GeV)",
                            flavor=a_flavor_here,sample="CC")
    plot_bin_slices(true_energy[a_mask_here],
                            cnn_energy[a_mask_here],
                            old_reco = retro_energy[a_mask_here],
                            weights=weights[a_mask_here],
                            vs_predict = False,reco_name="Likelihood",
                            use_fraction = True, bins=20,
                            min_val=1, max_val=maxval_here,\
                            save=save, savefolder=save_folder,\
                            variable="Energy", units="(GeV)",
                            flavor=a_flavor_here,sample="CC")
    plot_bin_slices(true_coszenith[a_mask_here],
                            cnn_coszen[a_mask_here],
                            old_reco = retro_coszen[a_mask_here],
                            weights=weights[a_mask_here],
                            energy_truth = true_energy[a_mask_here],
                            vs_predict = False,reco_name="Likelihood",
                            use_fraction = False, bins=20,
                            min_val=1, max_val=200,\
                            save=save, savefolder=save_folder,\
                            variable="Cosine Zenith", units="(GeV)",
                            xvariable="True Energy",
                            flavor=a_flavor_here,sample="CC")
    plot_bin_slices(true_coszenith[a_mask_here],
                            cnn_coszen[a_mask_here],
                            old_reco = retro_coszen[a_mask_here],
                            weights=weights[a_mask_here],
                            #energy_truth = true_energy[a_mask_here],
                            vs_predict = False,reco_name="Likelihood",
                            use_fraction = False, bins=20,
                            min_val=-1, max_val=1,\
                            save=save, savefolder=save_folder,\
                            variable="Cosine Zenith", units="",
                            flavor=a_flavor_here,sample="CC")
    plot_bin_slices(true_r[a_mask_here],
                            cnn_r[a_mask_here],
                            old_reco = retro_r[a_mask_here],
                            weights=weights[a_mask_here],
                            #energy_truth = true_energy[a_mask_here]
                            vs_predict = False,reco_name="Likelihood",
                            use_fraction = False, bins=20,
                            min_val=0, max_val=200,\
                            save=save, savefolder=save_folder,\
                            variable="Radius From Center String", units="(m)",
                            flavor=a_flavor_here,sample="CC",legend="lower left",
                            xline=90)
    plot_bin_slices(true_z[a_mask_here],
                            cnn_z[a_mask_here],
                            old_reco = retro_z[a_mask_here],
                            weights=weights[a_mask_here],
                            #energy_truth = true_energy[a_mask_here]
                            vs_predict = False,reco_name="Likelihood",
                            use_fraction = False, #bins=60,
                            #min_val=-650, max_val=200,\
                            specific_bins=[-550,-530,-510,-490,-470,-450,-430,-410,-390,-370,-350,-330,-310,-290,-270,-250,-230,-200,-150],
                            save=save, savefolder=save_folder,\
                            variable="Z Depth Position", units="(m)",
                            legend="lower center",xline=[-500,-200],
                            flavor=a_flavor_here,sample="CC")

    print("NuMu CC")
    weights_here = weights[a_mask_here]
    cnn_rcut = cnn_r[a_mask_here] < 165
    retro_rcut = retro_r[a_mask_here] < 165
    true_rcut = true_r[a_mask_here] < 165
    print("R Cut: CNN, Retro, True")
    print(sum(cnn_rcut), sum(retro_rcut), sum(true_rcut))
    print(sum(weights_here[cnn_rcut]), sum(weights_here[retro_rcut]), sum(weights_here[true_rcut]))
    cnn_zcut = np.logical_and(cnn_z[a_mask_here] > -495, cnn_z[a_mask_here] < -225)
    retro_zcut = np.logical_and(retro_z[a_mask_here] > -495, retro_z[a_mask_here] < -225)
    true_zcut = np.logical_and(true_z[a_mask_here] > -495, true_z[a_mask_here] < -225)
    print("Z Cut: CNN, Retro, True")
    print(sum(cnn_zcut), sum(retro_zcut), sum(true_zcut))
    print(sum(weights_here[cnn_zcut]), sum(weights_here[retro_zcut]), sum(weights_here[true_zcut]))
    
    print("NuMu all")
    a_mask_here = true_isNuMu
    weights_here = weights[a_mask_here]
    cnn_rcut = cnn_r[a_mask_here] < 165
    retro_rcut = retro_r[a_mask_here] < 165
    true_rcut = true_r[a_mask_here] < 165
    print("R Cut: CNN, Retro, True")
    print(sum(cnn_rcut), sum(retro_rcut), sum(true_rcut))
    print(sum(weights_here[cnn_rcut]), sum(weights_here[retro_rcut]), sum(weights_here[true_rcut]))
    cnn_zcut = np.logical_and(cnn_z[a_mask_here] > -495, cnn_z[a_mask_here] < -225)
    retro_zcut = np.logical_and(retro_z[a_mask_here] > -495, retro_z[a_mask_here] < -225)
    true_zcut = np.logical_and(true_z[a_mask_here] > -495, true_z[a_mask_here] < -225)
    print("Z Cut: CNN, Retro, True")
    print(sum(cnn_zcut), sum(retro_zcut), sum(true_zcut))
    print(sum(weights_here[cnn_zcut]), sum(weights_here[retro_zcut]), sum(weights_here[true_zcut]))



    a_mask_here = true_isNuE&true_isCC
    a_flavor_here = "NuE"
    plot_2D_prediction(true_energy[a_mask_here],
                                cnn_energy[a_mask_here],
                                weights=weights[a_mask_here],
                                save=save, savefolder=save_folder,
                                bins=bins_here,minval=0,maxval=maxval_here,axis_square=True,
                                variable="Energy", units="(GeV)",
                                reco_name="CNN",flavor=a_flavor_here,sample="CC")
    plot_2D_prediction(true_energy[a_mask_here],
                                retro_energy[a_mask_here],
                                weights=weights[a_mask_here],
                                save=save, savefolder=save_folder,
                                bins=bins_here,minval=0,maxval=maxval_here,axis_square=True,
                                variable="Energy", units="(GeV)",
                                reco_name="Retro",flavor=a_flavor_here,sample="CC")
    plot_bin_slices(true_energy[a_mask_here],
                            cnn_energy[a_mask_here],
                            old_reco = retro_energy[a_mask_here],
                            weights=weights[a_mask_here],
                            vs_predict = True,\
                            use_fraction = True, bins=20,
                            min_val=1, max_val=maxval_here,\
                            save=save, savefolder=save_folder,\
                            variable="Energy", units="(GeV)",
                            flavor=a_flavor_here,sample="CC")
    plot_bin_slices(true_energy[a_mask_here],
                            cnn_energy[a_mask_here],
                            old_reco = retro_energy[a_mask_here],
                            weights=weights[a_mask_here],
                            vs_predict = False,reco_name="Likelihood",
                            use_fraction = True, bins=20,
                            min_val=1, max_val=maxval_here,\
                            save=save, savefolder=save_folder,\
                            variable="Energy", units="(GeV)",
                            flavor=a_flavor_here,sample="CC")
    plot_bin_slices(true_coszenith[a_mask_here],
                            cnn_coszen[a_mask_here],
                            old_reco = retro_coszen[a_mask_here],
                            weights=weights[a_mask_here],
                            energy_truth = true_energy[a_mask_here],
                            vs_predict = False,reco_name="Likelihood",
                            use_fraction = False, bins=20,
                            min_val=1, max_val=200,\
                            save=save, savefolder=save_folder,\
                            variable="Cosine Zenith", units="(GeV)",
                            xvariable="True Energy",
                            flavor=a_flavor_here,sample="CC")
    plot_bin_slices(true_coszenith[a_mask_here],
                            cnn_coszen[a_mask_here],
                            old_reco = retro_coszen[a_mask_here],
                            weights=weights[a_mask_here],
                            #energy_truth = true_energy[a_mask_here],
                            vs_predict = False,reco_name="Likelihood",
                            use_fraction = False, bins=20,
                            min_val=-1, max_val=1,\
                            save=save, savefolder=save_folder,\
                            variable="Cosine Zenith", units="",
                            flavor=a_flavor_here,sample="CC")
    plot_bin_slices(true_r[a_mask_here],
                            cnn_r[a_mask_here],
                            old_reco = retro_r[a_mask_here],
                            weights=weights[a_mask_here],
                            #energy_truth = true_energy[a_mask_here]
                            vs_predict = False,reco_name="Likelihood",
                            use_fraction = False, bins=20,
                            min_val=0, max_val=200,\
                            save=save, savefolder=save_folder,\
                            variable="Radius From Center String", units="(m)",
                            flavor=a_flavor_here,sample="CC",legend="lower left",
                            xline=90)
    plot_bin_slices(true_z[a_mask_here],
                            cnn_z[a_mask_here],
                            old_reco = retro_z[a_mask_here],
                            weights=weights[a_mask_here],
                            #energy_truth = true_energy[a_mask_here]
                            vs_predict = False,reco_name="Likelihood",
                            use_fraction = False,
                            #min_val=-650, max_val=200,\
                            specific_bins=[-550,-530,-510,-490,-470,-450,-430,-410,-390,-370,-350,-330,-310,-290,-270,-250,-230,-200,-150],
                            save=save, savefolder=save_folder,\
                            variable="Z Depth Position", units="(m)",
                            flavor=a_flavor_here,sample="CC",
                            xline=[-500,-200],legend="lower center")
    
    print("NuE CC")
    cnn_rcut = cnn_r[a_mask_here] < 165
    retro_rcut = retro_r[a_mask_here] < 165
    true_rcut = true_r[a_mask_here] < 165
    print("R Cut: CNN, Retro, True")
    print(sum(cnn_rcut), sum(retro_rcut), sum(true_rcut))
    print(sum(weights_here[cnn_rcut]), sum(weights_here[retro_rcut]), sum(weights_here[true_rcut]))
    cnn_zcut = np.logical_and(cnn_z[a_mask_here] > -495, cnn_z[a_mask_here] < -225)
    retro_zcut = np.logical_and(retro_z[a_mask_here] > -495, retro_z[a_mask_here] < -225)
    true_zcut = np.logical_and(true_z[a_mask_here] > -495, true_z[a_mask_here] < -225)
    print("Z Cut: CNN, Retro, True")
    print(sum(cnn_zcut), sum(retro_zcut), sum(true_zcut))
    print(sum(weights_here[cnn_zcut]), sum(weights_here[retro_zcut]), sum(weights_here[true_zcut]))
    
    print("NuE all")
    a_mask_here = true_isNuE 
    cnn_rcut = cnn_r[a_mask_here] < 165
    retro_rcut = retro_r[a_mask_here] < 165
    true_rcut = true_r[a_mask_here] < 165
    print("R Cut: CNN, Retro, True")
    print(sum(cnn_rcut), sum(retro_rcut), sum(true_rcut))
    print(sum(weights_here[cnn_rcut]), sum(weights_here[retro_rcut]), sum(weights_here[true_rcut]))
    cnn_zcut = np.logical_and(cnn_z[a_mask_here] > -495, cnn_z[a_mask_here] < -225)
    retro_zcut = np.logical_and(retro_z[a_mask_here] > -495, retro_z[a_mask_here] < -225)
    true_zcut = np.logical_and(true_z[a_mask_here] > -495, true_z[a_mask_here] < -225)
    print("Z Cut: CNN, Retro, True")
    print(sum(cnn_zcut), sum(retro_zcut), sum(true_zcut))
    print(sum(weights_here[cnn_zcut]), sum(weights_here[retro_zcut]), sum(weights_here[true_zcut]))
    
    print("NuMu Events: ", sum(true_isNuMu), "NuE Events: ", sum(true_isNuE))
    plt.figure(figsize=(10,7))
    bins = 10**np.linspace(0,2.7,100)
    #plt.hist(true_energy[true_isMuon],range=[emin,emax],bins=bins,weights=weights[true_isMuon],label=r'$\mu$',alpha=0.5)
    plt.hist([true_energy[true_isNuMu], true_energy[true_isNuE]],range=[0,max(true_energy[true_isNuMu])],bins=bins,weights=[weights[true_isNuMu],weights[true_isNuE]],label=(r'$\nu_\mu$',r'$\nu_e$'),stacked=True,alpha=0.5)
    #plt.hist(true_energy[true_isNuE],range=[emin,emax],bins=bins,weights=weights[true_isNuE],label=r'$\nu_e$',alpha=0.5)
    #plt.hist(true_energy[true_isNuTau],range=[emin,emax],bins=bins,weights=weights[true_isNuTau],label=r'$\nu_\tau$',alpha=0.5)
    plt.xscale('log')
    plt.ylabel("Rate (Hz)",fontsize=20)
    plt.xlabel("True Energy (GeV)",fontsize=20)
    plt.title("True Energy Distribution",fontsize=25)
    plt.legend()
    plt.savefig("%sEnergyParticleHist_NoMuon.png"%(save_folder),bbox_inches='tight')
    plt.close()
    
    bins=100
    plt.figure(figsize=(10,7))
    plt.hist([true_coszenith[true_isNuMu], true_coszenith[true_isNuE]],range=[-1,1],bins=bins,weights=[weights[true_isNuMu],weights[true_isNuE]],label=(r'$\nu_\mu$',r'$\nu_e$'),stacked=True,alpha=0.5)
    plt.ylabel("Rate (Hz)",fontsize=20)
    plt.xlabel("True Cosine Zenith",fontsize=20)
    plt.title("True Cosine Zenith Distribution",fontsize=25)
    plt.legend()
    plt.savefig("%sCosZenParticleHist_NoMuon.png"%(save_folder),bbox_inches='tight')
    plt.close()
    
    bins=100
    plt.figure(figsize=(10,7))
    plt.hist([true_r[true_isNuMu], true_r[true_isNuE]],range=[0,300],bins=bins,weights=[weights[true_isNuMu],weights[true_isNuE]],label=(r'$\nu_\mu$',r'$\nu_e$'),stacked=True,alpha=0.5)
    plt.ylabel("Rate (Hz)",fontsize=20)
    plt.xlabel("True Radius From Center String (m)",fontsize=20)
    plt.title("True Radius From Center String Distribution",fontsize=25)
    plt.legend()
    plt.savefig("%sRParticleHist_NoMuon.png"%(save_folder),bbox_inches='tight')
    plt.close()
    
    bins=100
    plt.figure(figsize=(10,7))
    plt.hist([true_z[true_isNuMu], true_z[true_isNuE]],range=[-700,200],bins=bins,weights=[weights[true_isNuMu],weights[true_isNuE]],label=(r'$\nu_\mu$',r'$\nu_e$'),stacked=True,alpha=0.5)
    plt.ylabel("Rate (Hz)",fontsize=20)
    plt.xlabel("True Z Depth Position (m)",fontsize=20)
    plt.title("True Z Depth Position",fontsize=25)
    plt.legend()
    plt.savefig("%sZParticleHist_NoMuon.png"%(save_folder),bbox_inches='tight')
    plt.close()

    a_mask_here = true_isNuTau&true_isCC
    a_flavor_here = "NuTau"
    plot_2D_prediction(true_energy[a_mask_here],
                                cnn_energy[a_mask_here],
                                weights=weights[a_mask_here],
                                save=save, savefolder=save_folder,
                                bins=bins_here,minval=0,maxval=maxval_here,axis_square=True,
                                variable="Energy", units="(GeV)",
                                reco_name="CNN",flavor=a_flavor_here,sample="CC")
    plot_2D_prediction(true_energy[a_mask_here],
                                retro_energy[a_mask_here],
                                weights=weights[a_mask_here],
                                save=save, savefolder=save_folder,
                                bins=bins_here,minval=0,maxval=maxval_here,axis_square=True,
                                variable="Energy", units="(GeV)",
                                reco_name="Retro",flavor=a_flavor_here,sample="CC")
    plot_bin_slices(true_energy[a_mask_here],
                            cnn_energy[a_mask_here],
                            old_reco = retro_energy[a_mask_here],
                            weights=weights[a_mask_here],
                            vs_predict = True,\
                            use_fraction = True, bins=20,
                            min_val=1, max_val=maxval_here,\
                            save=save, savefolder=save_folder,\
                            variable="Energy", units="(GeV)",
                            flavor=a_flavor_here,sample="CC")
    plot_bin_slices(true_energy[a_mask_here],
                            cnn_energy[a_mask_here],
                            old_reco = retro_energy[a_mask_here],
                            weights=weights[a_mask_here],
                            vs_predict = False,\
                            use_fraction = True, bins=20,
                            min_val=1, max_val=maxval_here,\
                            save=save, savefolder=save_folder,\
                            variable="Energy", units="(GeV)",
                            flavor=a_flavor_here,sample="CC")
    a_mask_here = true_isNu&true_isCC
    a_flavor_here = "Nu"
    plot_2D_prediction(true_energy[a_mask_here],
                                cnn_energy[a_mask_here],
                                weights=weights[a_mask_here],
                                save=save, savefolder=save_folder,
                                bins=bins_here,minval=0,maxval=maxval_here,axis_square=True,
                                variable="Energy", units="(GeV)",
                                reco_name="CNN",flavor=a_flavor_here,sample="CC")
    plot_2D_prediction(true_energy[a_mask_here],
                                retro_energy[a_mask_here],
                                weights=weights[a_mask_here],
                                save=save, savefolder=save_folder,
                                bins=bins_here,minval=0,maxval=maxval_here,axis_square=True,
                                variable="Energy", units="(GeV)",
                                reco_name="Retro",flavor=a_flavor_here,sample="CC")
    plot_bin_slices(true_energy[a_mask_here],
                            cnn_energy[a_mask_here],
                            old_reco = retro_energy[a_mask_here],
                            weights=weights[a_mask_here],
                            vs_predict = True,\
                            use_fraction = True, bins=20,
                            min_val=1, max_val=maxval_here,\
                            save=save, savefolder=save_folder,\
                            variable="Energy", units="(GeV)",
                            flavor=a_flavor_here,sample="CC")
    plot_bin_slices(true_energy[a_mask_here],
                            cnn_energy[a_mask_here],
                            old_reco = retro_energy[a_mask_here],
                            weights=weights[a_mask_here],
                            vs_predict = False,\
                            use_fraction = True, bins=20,
                            min_val=1, max_val=maxval_here,\
                            save=save, savefolder=save_folder,\
                            variable="Energy", units="(GeV)",
                            flavor=a_flavor_here,sample="CC")

if plot_energy:
    plt.figure(figsize=(10,7))
    emin = 0
    emax = 500
    bins = 10**np.linspace(0,2.7,100)
    plt.hist(true_energy,range=[emin,emax],bins=bins,weights=weights)
    plt.xlabel("True Energy (GeV)")
    plt.title("True Energy Distribution")
    plt.ylabel("Rate (Hz)")
    plt.savefig("%sEnergyHist.png"%(save_folder),bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10,7))
    bins = 10**np.linspace(0,2.7,100)
    plt.hist(true_energy[true_isMuon],range=[emin,emax],bins=bins,weights=weights[true_isMuon],label=r'$\mu$',alpha=0.5)
    plt.hist(true_energy[true_isNuMu],range=[emin,emax],bins=bins,weights=weights[true_isNuMu],label=r'$\nu_\mu$',alpha=0.5)
    plt.hist(true_energy[true_isNuE],range=[emin,emax],bins=bins,weights=weights[true_isNuE],label=r'$\nu_e$',alpha=0.5)
    plt.hist(true_energy[true_isNuTau],range=[emin,emax],bins=bins,weights=weights[true_isNuTau],label=r'$\nu_\tau$',alpha=0.5)
    #plt.yscale('log')
    plt.ylabel("Rate (Hz)",fontsize=20)
    plt.xlabel("True Energy (GeV)",fontsize=20)
    plt.title("True Energy Distribution",fontsize=25)
    plt.legend()
    plt.savefig("%sEnergyParticleHist.png"%(save_folder),bbox_inches='tight')
    plt.close()
    

if plot_main:
    bins=50
    assert sum(true_isMuon)>0, "No true muon saved"
    plt.figure(figsize=(10,7))
    plt.hist(cnn_prob_mu[cnn_mask&true_isMuon],bins=bins,weights=weights[cnn_mask&true_isMuon],label=r'$\mu$',alpha=0.5)
    plt.hist(cnn_prob_mu[cnn_mask&true_isNuMu],bins=bins,weights=weights[cnn_mask&true_isNuMu],label=r'$\nu_\mu$',alpha=0.5)
    plt.hist(cnn_prob_mu[cnn_mask&true_isNuE],bins=bins,weights=weights[cnn_mask&true_isNuE],label=r'$\nu_e$',alpha=0.5)
    plt.hist(cnn_prob_mu[cnn_mask&true_isNuTau],bins=bins,weights=weights[cnn_mask&true_isNuTau],label=r'$\nu_\tau$',alpha=0.5)
#plt.yscale('log')
    plt.ylabel("Rate (Hz)",fontsize=20)
    plt.xlabel("Muon Probability",fontsize=20)
    plt.title("FLERCNN Muon Probability Distribution",fontsize=25)
    plt.yscale('log')
    plt.legend()
    plt.savefig("%sProbMuonParticleHistCNN_log.png"%(save_folder),bbox_inches='tight')
    plt.close()
    print("MASK + true muon:",sum(cnn_mask&true_isMuon),sum(weights[cnn_mask&true_isMuon]))
    
    plt.figure(figsize=(10,7))
    plt.hist(retro_prob_nu[retro_mask&true_isMuon],bins=bins,weights=weights[retro_mask&true_isMuon],label=r'$\mu$',alpha=0.5)
    plt.hist(retro_prob_nu[retro_mask&true_isNuMu],bins=bins,weights=weights[retro_mask&true_isNuMu],label=r'$\nu_\mu$',alpha=0.5)
    plt.hist(retro_prob_nu[retro_mask&true_isNuE],bins=bins,weights=weights[retro_mask&true_isNuE],label=r'$\nu_e$',alpha=0.5)
    plt.hist(retro_prob_nu[retro_mask&true_isNuTau],bins=bins,weights=weights[retro_mask&true_isNuTau],label=r'$\nu_\tau$',alpha=0.5)
#plt.yscale('log')
    plt.ylabel("Rate (Hz)",fontsize=20)
    plt.xlabel("Neutrino Probability",fontsize=20)
    plt.title("Retro Neutrino Probability Distribution",fontsize=25)
    plt.legend()
    plt.savefig("%sProbNuParticleHistRETRO.png"%(save_folder),bbox_inches='tight')
    plt.close()


    #ROC(true_isMuon,cnn_prob_mu,mask=cnn_mask,reco_mask=retro_mask,mask_name=mask_name_here,save=save,save_folder_name=save_folder,variable="Probability Muon")

    plot_classification_hist(true_isMuon,cnn_prob_mu,mask=cnn_mask,reco_mask=retro_mask,mask_name=mask_name_here, units="",weights=weights,bins=50,log=False,save=save,save_folder_name=save_folder,name_prob0 = "Neutrino", name_prob1 = "Muon")

    #plot_classification_hist(true_isMuon,cnn_prob_mu,mask=no_cuts,mask_name=mask_name_here, variable="Probabiliy Muon",units="",weights=weights,bins=50,log=False,save=save,save_folder_name=save_folder,normed=True,name_prob0 = "Neutrino", name_prob1 = "Muon")

    
    plt.figure(figsize=(10,7))
    plt.hist(cnn_prob_nu[cnn_mask&true_isMuon],bins=50,weights=weights[cnn_mask&true_isMuon],label=r'$\mu$',alpha=0.5)
    plt.hist(cnn_prob_nu[cnn_mask&true_isNuMu],bins=50,weights=weights[cnn_mask&true_isNuMu],label=r'$\nu_\mu$',alpha=0.5)
    plt.hist(cnn_prob_nu[cnn_mask&true_isNuE],bins=50,weights=weights[cnn_mask&true_isNuE],label=r'$\nu_e$',alpha=0.5)
    plt.hist(cnn_prob_nu[cnn_mask&true_isNuTau],bins=50,weights=weights[cnn_mask&true_isNuTau],label=r'$\nu_\tau$',alpha=0.5)
#plt.yscale('log')
    plt.ylabel("Rate (Hz)",fontsize=20)
    plt.yscale('log')
    plt.xlabel("Neutrino Probability",fontsize=20)
    plt.title("FLERCNN Neutrino Probability Distribution",fontsize=25)
    plt.legend()
    plt.savefig("%sProbNuParticleHistCNN.png"%(save_folder),bbox_inches='tight')
    plt.close()

    assert sum(true_isNu)!=len(true_isNu), "All neutrino sample, lost muons"
    assert sum(true_isMuon)>0, "No true muon saved"

    plot_classification_hist(true_isNu,cnn_prob_nu,reco=retro_prob_nu,mask=cnn_mask,reco_mask=retro_mask,mask_name=mask_name_here, units="",weights=weights,bins=50,log=True,save=save,save_folder_name=save_folder,name_prob1 = "Neutrino", name_prob0 = "Muon")
    
    ROC(true_isNu,cnn_prob_nu,reco=retro_prob_nu,mask=cnn_mask,reco_mask=retro_mask,mask_name=mask_name_here,save=save,save_folder_name=save_folder,variable="Probability Neutrino vs Muon",reco_name="Likelihood")

    plot_classification_hist(true_isTrack,cnn_prob_track,reco=retro_prob_track,mask=cnn_mask,reco_mask=retro_mask,mask_name=mask_name_here, units="",weights=weights,bins=50,log=False,save=save,save_folder_name=save_folder,name_prob1 = "Track", name_prob0 = "Cascade")
    
    ROC(true_isTrack,cnn_prob_track,reco=retro_prob_track,mask=cnn_mask,reco_mask=retro_mask,mask_name=mask_name_here,save=save,save_folder_name=save_folder,variable="Probability Track vs Cascade",reco_name="Likelihood")

if compare_resolution:
    plot_2D_prediction(retro_prob_nu[retro_mask], cnn_prob_nu[retro_mask],
                                weights=weights[retro_mask],
                                save=save, savefolder=save_folder,
                                bins=50,minval=-.1,maxval=1.1,axis_square=True,
                                variable="Probability Nu", units="(all events)",
                                reco_name="CNN",variable_type="Retro",
                                save_name="All Events",flavor="All",
                                no_contours=True,
                                yline=cnn_nu_cut,xline=retro_nu_cut)
    plot_2D_prediction(retro_prob_nu[retro_mask&true_isMuon], cnn_prob_nu[retro_mask&true_isMuon],
                                weights=weights[retro_mask&true_isMuon],
                                save=save, savefolder=save_folder,
                                bins=50,minval=-.1,maxval=1.1,axis_square=True,
                                variable="Probability Nu", units="(true Mu)",
                                reco_name="CNN",variable_type="Retro",flavor="Mu",
                                no_contours=True,
                                yline=cnn_nu_cut,xline=retro_nu_cut)
    plot_2D_prediction(retro_prob_nu[retro_mask&true_isNu], cnn_prob_nu[retro_mask&true_isNu],
                                weights=weights[retro_mask&true_isNu],
                                save=save, savefolder=save_folder,
                                bins=50,minval=-.1,maxval=1.1,axis_square=True,
                                variable="Probability Nu", units="(true Nu)",
                                reco_name="CNN",variable_type="Retro",flavor="Nu",
                                no_contours=True,
                                yline=cnn_nu_cut,xline=retro_nu_cut)
    plot_2D_prediction(retro_prob_nu[retro_mask&true_isNu], cnn_prob_nu[retro_mask&true_isNu],
                                weights=weights[retro_mask&true_isNu],
                                save=save, savefolder=save_folder,
                                bins=50,minval=-.1,maxval=1.1,axis_square=True,
                                variable="Probability Nu", units="(true Nu)",
                                reco_name="CNN",variable_type="Retro",flavor="Nu",
                                no_contours=True,zmin=0.000000001,
                                yline=cnn_nu_cut,xline=retro_nu_cut)
    plot_2D_prediction(retro_prob_nu[retro_mask&true_isNuMu], cnn_prob_nu[retro_mask&true_isNuMu],
                                weights=weights[retro_mask&true_isNuMu],
                                save=save, savefolder=save_folder,
                                bins=50,minval=-.1,maxval=1.1,axis_square=True,
                                variable="Probability Nu", units="(true NuMu)",
                                reco_name="CNN",variable_type="Retro",flavor="NuMu",
                                no_contours=True,
                                yline=cnn_nu_cut,xline=retro_nu_cut)

    plot_2D_prediction(retro_prob_nu[retro_mask&true_isNuE], cnn_prob_nu[retro_mask&true_isNuE],
                                weights=weights[retro_mask&true_isNuE],
                                save=save, savefolder=save_folder,
                                bins=50,minval=-.1,maxval=1.1,axis_square=True,
                                variable="Probability Nu", units="(true NuE)",
                                reco_name="CNN",variable_type="Retro",flavor="NuE",
                                no_contours=True,
                                yline=cnn_nu_cut,xline=retro_nu_cut)
    plot_2D_prediction(retro_prob_nu[retro_mask&true_isNuTau], cnn_prob_nu[retro_mask&true_isNuTau],
                                weights=weights[retro_mask&true_isNuTau],
                                save=save, savefolder=save_folder,
                                bins=50,minval=-.1,maxval=1.1,axis_square=True,
                                variable="Probability Nu", units="(true NuTau)",
                                reco_name="CNN",variable_type="Retro",flavor="NuTau",
                                no_contours=True,
                                yline=cnn_nu_cut,xline=retro_nu_cut)

if hists1d:
    from PlottingFunctions import plot_distributions

    #all fake calculated with respect to retro, not data
    fake_correct_mu = np.logical_and(cnn_mu, retro_mu)
    fake_wrong_mu = np.logical_and(cnn_mu, retro_nu)
    fake_correct_nu = np.logical_and(cnn_nu, retro_nu)
    fake_wrong_nu = np.logical_and(cnn_nu, retro_mu)
    a_mask = retro_mask & true_isMuon

    print(sum(weights[retro_mask&true_isNu&retro_nu&cnn_mu]),
          sum(weights[retro_mask&true_isMuon&retro_mu&cnn_nu]),
          sum(weights[retro_mask&true_isNu&retro_mu&cnn_nu]),
          sum(weights[retro_mask&true_isMuon&retro_nu&cnn_mu]))

    a_title = "True Muon: All"
    plot_distributions(true_energy[a_mask], cnn_energy[a_mask],
                          weights=weights[a_mask],\
                          old_reco=retro_energy[a_mask],
                          save=save, savefolder=save_folder, 
                          reco_name = "Retro", variable="Energy",
                          units= "(GeV)",title=a_title,
                          minval=5,maxval=500,bins=100,xlog=True) 
    plot_distributions(true_coszenith[a_mask], cnn_coszen[a_mask],
                          weights=weights[a_mask],\
                          old_reco=retro_coszen[a_mask],
                          save=save, savefolder=save_folder, 
                          reco_name = "Retro", variable="Cosine Zenith",
                          units= "",title=a_title,
                          minval=-1,maxval=1,bins=100) 
    plot_distributions(true_r[a_mask]*true_r[a_mask], cnn_r[a_mask]*cnn_r[a_mask],
                          weights=weights[a_mask],\
                          old_reco=retro_r[a_mask]*retro_r[a_mask],
                          save=save, savefolder=save_folder, 
                          reco_name = "Retro", variable="R^2 Position",
                          units= "(m^2)",xline=125*125,xline_label="DeepCore",
                          minval=0,maxval=300*300,
                          bins=100,title=a_title,) 
    plot_distributions(true_r[a_mask], cnn_r[a_mask],
                          weights=weights[a_mask],\
                          old_reco=retro_r[a_mask],
                          save=save, savefolder=save_folder, 
                          reco_name = "Retro", variable="R Position" ,
                          units= "(m)",xline=125,xline_label="DeepCore",
                          minval=0,maxval=300,
                          bins=100,title=a_title,) 
    plot_distributions(true_z[a_mask], cnn_z[a_mask],
                          weights=weights[a_mask],\
                          old_reco=retro_z[a_mask],
                          save=save, savefolder=save_folder, 
                          reco_name = "Retro", variable="Z Position",
                          units= "(m)",title=a_title,
                          minval=-1000,maxval=300,bins=100) 
    
    a_title = "True Neutrino: All"
    a_mask = retro_mask & true_isNu
    plot_distributions(true_energy[a_mask], cnn_energy[a_mask],
                          weights=weights[a_mask],\
                          old_reco=retro_energy[a_mask],
                          save=save, savefolder=save_folder, 
                          reco_name = "Retro", variable="Energy",
                          units= "(GeV)",title=a_title,
                          minval=3,maxval=500,bins=100,xlog=True) 
    plot_distributions(true_coszenith[a_mask], cnn_coszen[a_mask],
                          weights=weights[a_mask],\
                          old_reco=retro_coszen[a_mask],
                          save=save, savefolder=save_folder, 
                          reco_name = "Retro", variable="Cosine Zenith",
                          units= "",title=a_title,
                          minval=-1,maxval=1,bins=100) 
    plot_distributions(true_r[a_mask]*true_r[a_mask], cnn_r[a_mask]*cnn_r[a_mask],
                          weights=weights[a_mask],\
                          old_reco=retro_r[a_mask]*retro_r[a_mask],
                          save=save, savefolder=save_folder, 
                          reco_name = "Retro", variable="R^2 Position",
                          units= "(m^2)",xline=125*125,xline_label="DeepCore",
                          minval=0,maxval=300*300,
                          bins=100,title=a_title,) 
    plot_distributions(true_r[a_mask], cnn_r[a_mask],
                          weights=weights[a_mask],\
                          old_reco=retro_r[a_mask],
                          save=save, savefolder=save_folder, 
                          reco_name = "Retro", variable="Position R",
                          units= "(m)",xline=125,xline_label="DeepCore",
                          minval=0,maxval=300,
                          bins=100,title=a_title,) 
    plot_distributions(true_z[a_mask], cnn_z[a_mask],
                          weights=weights[a_mask],\
                          old_reco=retro_z[a_mask],
                          save=save, savefolder=save_folder, 
                          reco_name = "Retro", variable="Position Z",
                          units= "(m)",title=a_title,
                          minval=-1000,maxval=300,bins=100) 
    """
    a_title = "True Neutrino: NuMu CC"
    a_mask = retro_mask & true_isNuMu & true_isCC
    plot_distributions(true_energy[a_mask], cnn_energy[a_mask],
                          weights=weights[a_mask],\
                          old_reco=retro_energy[a_mask],
                          save=save, savefolder=save_folder, 
                          reco_name = "Retro", variable="Energy",
                          units= "(GeV)",title=a_title,
                          minval=0,maxval=20,bins=100,xlog=False) 
    """
    plot_distributions(true_coszenith[a_mask], cnn_coszen[a_mask],
                          weights=weights[a_mask],\
                          old_reco=retro_coszen[a_mask],
                          save=save, savefolder=save_folder, 
                          reco_name = "Retro", variable="Cosine Zenith",
                          units= "",title=a_title,
                          minval=-1,maxval=1,bins=100) 
    plot_distributions(true_r[a_mask]*true_r[a_mask], cnn_r[a_mask]*cnn_r[a_mask],
                          weights=weights[a_mask],\
                          old_reco=retro_r[a_mask]*retro_r[a_mask],
                          save=save, savefolder=save_folder, 
                          reco_name = "Retro", variable="R^2 Position",
                          units= "(m^2)",xline=125*125,xline_label="DeepCore",
                          minval=0,maxval=300*300,
                          bins=100,title=a_title,) 
    plot_distributions(true_r[a_mask], cnn_r[a_mask],
                          weights=weights[a_mask],\
                          old_reco=retro_r[a_mask],
                          save=save, savefolder=save_folder, 
                          reco_name = "Retro", variable="Position R",
                          units= "(m)",xline=125,xline_label="DeepCore",
                          minval=0,maxval=300,
                          bins=100,title=a_title,) 
    plot_distributions(true_z[a_mask], cnn_z[a_mask],
                          weights=weights[a_mask],\
                          old_reco=retro_z[a_mask],
                          save=save, savefolder=save_folder, 
                          reco_name = "Retro", variable="Position Z",
                          units= "(m)",title=a_title,
                          minval=-1000,maxval=300,bins=100) 
    

def return_rates(true_PID,prediction,weights,threshold):

    predictionNu = prediction < threshold
    predictionMu = prediction >= threshold

    save_weights = weights[predictionNu]
    true_PID = true_PID[predictionNu]

    muon_mask = true_PID == 13
    true_isMuon = np.array(muon_mask,dtype=bool)
    nue_mask = true_PID == 12
    true_isNuE = np.array(nue_mask,dtype=bool)
    numu_mask = true_PID == 14
    true_isNuMu = np.array(numu_mask,dtype=bool)
    nutau_mask = true_PID == 16
    true_isNuTau = np.array(nutau_mask,dtype=bool)

    
    muon_rate = sum(save_weights[true_isMuon])
    nue_rate = sum(save_weights[true_isNuE])
    numu_rate = sum(save_weights[true_isNuMu])
    nutau_rate = sum(save_weights[true_isNuTau])
    
    return muon_rate, nue_rate, numu_rate, nutau_rate

def return_retro_rates(true_PID,prediction,weights,threshold):

    predictionNu = prediction > threshold
    predictionMu = prediction <= threshold

    save_weights = weights[predictionNu]
    true_PID = true_PID[predictionNu]

    muon_mask = true_PID == 13
    true_isMuon = np.array(muon_mask,dtype=bool)
    nue_mask = true_PID == 12
    true_isNuE = np.array(nue_mask,dtype=bool)
    numu_mask = true_PID == 14
    true_isNuMu = np.array(numu_mask,dtype=bool)
    nutau_mask = true_PID == 16
    true_isNuTau = np.array(nutau_mask,dtype=bool)

    
    muon_rate = sum(save_weights[true_isMuon])
    nue_rate = sum(save_weights[true_isNuE])
    numu_rate = sum(save_weights[true_isNuMu])
    nutau_rate = sum(save_weights[true_isNuTau])
    
    return muon_rate, nue_rate, numu_rate, nutau_rate


step=0.01
cut_values = np.arange(step,1+step,step)
muon_rates = []
numu_rates = []
nue_rates = []
nutau_rates = []
nu_rates = []
for muon_cut in cut_values:
    
    muon_rate, nue_rate, numu_rate, nutau_rate = return_rates(true_PID, cnn_prob_mu, weights, muon_cut)

    muon_rates.append(muon_rate)
    nue_rates.append(nue_rate)
    numu_rates.append(numu_rate)
    nutau_rates.append(nutau_rate)
    nu_rates.append(nue_rate + numu_rate + nutau_rate)

muon_retro_rates = []
numu_retro_rates = []
nue_retro_rates = []
nutau_retro_rates = []
nu_retro_rates = []
for muon_cut in cut_values:
    
    muon_retro_rate, nue_retro_rate, numu_retro_rate, nutau_retro_rate = return_retro_rates(true_PID, retro_prob_nu, weights, muon_cut)

    muon_retro_rates.append(muon_retro_rate)
    nue_retro_rates.append(nue_retro_rate)
    numu_retro_rates.append(numu_retro_rate)
    nutau_retro_rates.append(nutau_retro_rate)
    nu_retro_rates.append(nue_retro_rate + numu_retro_rate + nutau_retro_rate)

if print_rates:
    print(muon_rates)
    print(nue_rates)
    print(numu_rates)
    print(nutau_rates)
    print(nu_rates)
    print(cut_values)

if print_retro_rates:
    print(muon_retro_rates)
    print(nue_retro_rates)
    print(numu_retro_rates)
    print(nutau_retro_rates)
    print(nu_retro_rates)
    print(cut_values)

if plot_rates:
    plt.figure(figsize=(10,7))
    plt.title("Testing Sample",fontsize=25)
    plt.plot(cut_values,muon_rates,color='orange',label=r'CNN $\mu$')
    plt.plot(cut_values,nue_rates,color='g',label=r'CNN $\nu_e$')
    plt.plot(cut_values,numu_rates,color='b',label=r'CNN $\nu_\mu$')
    plt.plot(cut_values,nutau_rates,color='purple',label=r'CNN $\nu_\tau$')
    plt.xlabel("Muon Probability Threshold",fontsize=20)
    plt.ylabel("Rate (Hz)",fontsize=20)
    plt.legend(fontsize=20)
    plt.savefig("%sThresholdRates.png"%(save_folder),bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10,7))
    plt.title("Muon vs Neutrino Rates",fontsize=25)
    plt.plot(nu_rates,muon_rates,color='orange',linewidth=2,label="CNN test")
    plt.plot(nu_retro_rates,muon_retro_rates,linestyle=":",color='purple',linewidth=2,label="Retro test")
    plt.xlabel("Neutrino rates (Hz)",fontsize=20)
    plt.ylabel("Muon rates (Hz)",fontsize=20)
    plt.yscale('log')
    plt.locator_params(axis='x', nbins=6)
    plt.legend(fontsize=20)
    plt.savefig("%sMuonNuetrinoRates.png"%(save_folder),bbox_inches='tight')
    plt.close()


    #Flip sign
    flip_prob = np.ones(len(cut_values)) - np.array(cut_values)

    plt.figure(figsize=(10,7))
    plt.title("Testing Sample",fontsize=25)
    plt.plot(flip_prob,muon_rates,color='orange',linewidth=2,label=r'CNN $\mu$')
    plt.plot(flip_prob,nue_rates,color='g',linewidth=2,label=r'CNN $\nu_e$')
    plt.plot(flip_prob,numu_rates,color='b',linewidth=2,label=r'CNN $\nu_\mu$')
    plt.plot(flip_prob,nutau_rates,color='purple',linewidth=2,label=r'CNN $\nu_\tau$')
    plt.plot(cut_values,muon_retro_rates,linestyle=":",color='red',linewidth=2,label=r'Retro $\mu$')
    plt.plot(cut_values,nue_retro_rates,linestyle=":",color='lime',linewidth=2,label=r'Retro $\nu_e$')
    plt.plot(cut_values,numu_retro_rates,linestyle=":",color='cyan',linewidth=2,label=r'Retro $\nu_\mu$')
    plt.plot(cut_values,nutau_retro_rates,linestyle=":",color='magenta',linewidth=2,label=r'Retro $\nu_\tau$')
    plt.xlabel("Neutrino Probability Threshold",fontsize=20)
    plt.ylabel("Rate (Hz)",fontsize=20)
    plt.legend(fontsize=20)
    plt.savefig("%sNuThresholdRates.png"%(save_folder),bbox_inches='tight')
    plt.close()

# Plot after threshold
if plot_after_threshold:
    correct_mu = np.logical_and(cnn_mu, true_isMuon)
    wrong_mu = np.logical_and(cnn_mu, true_isNu)
    correct_nu = np.logical_and(cnn_nu, true_isNu)
    wrong_nu = np.logical_and(cnn_nu, true_isMuon)
    correct_mu_retro = np.logical_and(retro_mu, true_isMuon)
    wrong_mu_retro = np.logical_and(retro_mu, true_isNu)
    correct_nu_retro = np.logical_and(retro_nu, true_isNu)
    wrong_nu_retro = np.logical_and(retro_nu, true_isMuon)
    
    print(sum(weights[correct_mu]),sum(weights[wrong_mu]),sum(weights[wrong_nu]),sum(weights[correct_nu]))
    
    from plot_1d_slices import plot_1d_binned_slices

    #truth_int = np.array(true_isNu[cnn_mask], dtype=int) 
    #cnn_nu_int = np.array(cnn_nu[cnn_mask], dtype=int) 
    #retro_nu_int = np.array(retro_nu[retro_mask], dtype=int) 
    #truth_retro_int = np.array(true_isNu[retro_mask], dtype=int) 
    #plot_1d_binned_slices(truth_int, cnn_nu_int, reco2=retro_nu_int, xarray1=true_energy[cnn_mask], xarray2=true_energy[retro_mask], truth2 = truth_retro_int, plot_resolution=True,reco1_name = "CNN", reco2_name = "Retro", xmin = 0, xmax = 300, reco1_weight = weights[cnn_mask], reco2_weight = weights[retro_mask], savefolder=save_folder)

    confusion_matrix(true_isMuon, cnn_prob_mu, cnn_mu_cut, mask=no_cuts, mask_name=mask_name_here, weights=weights, save=True, save_folder_name=save_folder, name_prob1 = "Muon", name_prob0 = "Neutrino")

    if plot_vertex:
        plt.figure(figsize=(10,7))
        plt.hist(cnn_r,bins=50,weights=weights,label=r'Full Distribution',alpha=0.5)
        plt.hist(cnn_r[cnn_nu],bins=50,weights=weights[cnn_nu],label=r'After Cut',alpha=0.5)
        #plt.yscale('log')
        plt.ylabel("Rate (Hz)",fontsize=20)
        plt.xlabel("Predicted Radius from String 36 (m)",fontsize=20)
        plt.title("Predicted Radius After Cut",fontsize=25)
        plt.legend()
        plt.savefig("%sRDistHist.png"%(save_folder),bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10,7))
        plt.hist(cnn_r[true_isMuon],bins=50,weights=weights[true_isMuon],label=r'Full $\mu$ Distribution',alpha=0.5)
        plt.hist(cnn_r[cnn_nu&true_isMuon],bins=50,weights=weights[cnn_nu&true_isMuon],label=r'After CNN Cut',alpha=0.5)
#plt.yscale('log')
        plt.ylabel("Rate (Hz)",fontsize=20)
        plt.xlabel("Predicted Radius from String 36 (m)",fontsize=20)
        plt.title("Predicted Radius After Cut for Muon",fontsize=25)
        plt.legend()
        plt.savefig("%sRMuonDistHist.png"%(save_folder),bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(10,7))
        plt.hist(cnn_r[true_isMuon],bins=50,weights=weights[true_isMuon],label=r'Full $\mu$ Distribution',alpha=0.5)
        plt.hist(cnn_r[cnn_nu&true_isMuon],bins=50,weights=weights[cnn_nu&true_isMuon],label=r'After CNN Cut',alpha=0.5)
#plt.yscale('log')
        plt.ylabel("Rate (Hz)",fontsize=20)
        plt.xlabel("Predicted Radius from String 36 (m)",fontsize=20)
        plt.title("Predicted Radius After Cut for Muon",fontsize=25)
        plt.legend()
        plt.savefig("%sRMuonDistHist.png"%(save_folder),bbox_inches='tight')
        plt.close()

        """
        plt.figure(figsize=(10,7))
        plt.plot(cnn_r[correct_mu],cnn_z[correct_mu],"g*",label=r'$\mu$ Cut (correct)',alpha=0.5)
        plt.plot(cnn_r[correct_nu],cnn_z[correct_nu],"gs",label=r'$\nu$ Kept (correct)',alpha=0.5)
        plt.plot(cnn_r[wrong_mu],cnn_z[wrong_mu],"r*",label=r'$\mu$ Kept (wrong)',alpha=0.5)
        plt.plot(cnn_r[wrong_nu],cnn_z[wrong_nu],"rs",label=r'$\nu$ Cut (wrong)',alpha=0.5)
        #plt.yscale('log')
        plt.ylabel("Predicted Z-Depth (m)",fontsize=20)
        plt.xlabel("Predicted Radius from String 36 (m)",fontsize=20)
        plt.title("Position & Cut for Muon vs Neutrino",fontsize=25)
        plt.legend()
        plt.savefig("%sRZDistCuts.png"%(save_folder),bbox_inches='tight')
        plt.close()
        """

    if plot_coszen:
        plt.figure(figsize=(10,7))
        plt.hist(cnn_coszen,bins=50,weights=weights,label=r'Full Distribution',alpha=0.5)
        plt.hist(cnn_coszen[cnn_nu],bins=50,weights=weights[cnn_nu],label=r'After Cut',alpha=0.5)
        #plt.yscale('log')
        plt.ylabel("Rate (Hz)",fontsize=20)
        plt.xlabel("Cosine Zenith",fontsize=20)
        plt.title("Reconstructed Cosine Zenith",fontsize=25)
        plt.legend()
        plt.savefig("%sCosZenithDistHist.png"%(save_folder),bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10,7))
        plt.hist(cnn_coszen[true_isMuon],bins=50,weights=weights[true_isMuon],label=r'Full $\mu$ Distribution',alpha=0.5)
        plt.hist(cnn_coszen[cnn_nu&true_isMuon],bins=50,weights=weights[cnn_nu&true_isMuon],label=r'After CNN Cut',alpha=0.5)
        #plt.yscale('log')
        plt.ylabel("Rate (Hz)",fontsize=20)
        plt.xlabel("Cosine Zenith",fontsize=20)
        plt.title("Reconstructed Cosine Zenith for True Muon",fontsize=25)
        plt.legend()
        plt.savefig("%sCosZenithMuonDistHist.png"%(save_folder),bbox_inches='tight')
        plt.close()


if do_energy_range:
    #Break down energy range
    energy_ranges = [5, 10, 20, 30, 40, 60, 80, 100, 150, 200]
    #energy_ranges = [5, 10, 20, 30, 40, 60, 80, 100, 200]
    for e_index in range(0,len(energy_ranges)-1):
        energy_start = energy_ranges[e_index]
        energy_end = energy_ranges[e_index+1]
        current_mask_cnn = np.logical_and( np.logical_and(true_energy > energy_start, true_energy < energy_end), cnn_mask)
        current_mask_retro = np.logical_and( np.logical_and(true_energy > energy_start, true_energy < energy_end), retro_mask)
        current_name = "True Energy %i-%i GeV"%(energy_start,energy_end)
        print("%s"%current_name, sum(true_isMuon[current_mask_cnn]))
        plot_classification_hist(true_isNu,cnn_prob_nu,mask=current_mask_cnn,mask_name=current_name, weights=weights, name_prob1="Nu", name_prob0 = "Mu",units="",bins=50,log=False,save=save,save_folder_name=save_folder)
        plot_classification_hist(true_isNu,retro_prob_nu,mask=current_mask_retro,mask_name=current_name, weights=weights, name_prob1="Nu", name_prob0 = "Mu",units="",bins=50,log=False,save=save,save_folder_name=save_folder,reco_name="Retro")
        ROC(true_isNu,cnn_prob_nu,reco=retro_prob_nu,mask=current_mask_cnn,mask_name=current_name,reco_mask=current_mask_retro,save=save,save_folder_name=save_folder)

emin = 1
emax = 200
estep = 1
if do_energy_auc:
    # Energy vs AUC
    energy_auc = []
    reco_energy_auc = []
    energy_range = np.arange(emin,emax, estep)

    truth_Nu_cnn = true_isNu[cnn_mask]
    cnn_array = cnn_prob_nu[cnn_mask]
    truth_Nu_retro = true_isNu[retro_mask]
    retro_array = retro_prob_nu[retro_mask]
    AUC_title = "AUC vs. True Energy - %s"%mask_name_here
    save_name_extra = ""
    for energy_bin in energy_range:
        current_mask = np.logical_and(true_energy[cnn_mask] > energy_bin, true_energy[cnn_mask] < energy_bin + 1)

        energy_auc.append(roc_auc_score(truth_Nu_cnn[current_mask], cnn_array[current_mask]))
        if retro_prob_nu is not None:
            current_mask_retro = np.logical_and(true_energy[cnn_mask] > energy_bin, true_energy[cnn_mask] < energy_bin + 1)
            reco_energy_auc.append(roc_auc_score(truth_Nu_retro[current_mask_retro], retro_array[current_mask_retro]))
    
    plt.figure(figsize=(10,7))
    plt.title(AUC_title,fontsize=25)
    plt.ylabel("AUC",fontsize=20)
    plt.xlabel("True Energy (GeV)",fontsize=20)
    plt.plot(energy_range, energy_auc, 'b-',label="CNN")
    if retro_prob_nu is not None:
        plt.plot(energy_range, reco_energy_auc, color="orange",linestyle="-",label="Retro")
        plt.legend(loc="upper left",fontsize=20)
        save_name_extra += "_compareRetro"
    plt.savefig("%sAUCvsEnergy%s.png"%(save_folder,save_name_extra))

# Energy cut plot
if do_energy_cut_plots:
    energy_values = [50, 80, 100, 120, 150, 180, 200, 250, 300]
    muon_erates = []
    numu_erates = []
    nue_erates = []
    nutau_erates = []
    nu_erates = []
    for energy_cut in energy_values:

        muon_rate, nue_rate, numu_rate, nutau_rate = return_rates(truth[:,9], cnn_energy, weights, energy_cut)

        muon_erates.append(muon_rate)
        nue_erates.append(nue_rate)
        numu_erates.append(numu_rate)
        nutau_erates.append(nutau_rate)
        nu_erates.append(nue_rate + numu_rate + nutau_rate)

    plt.figure(figsize=(10,7))
    plt.title("Testing Sample: Energy Cut",fontsize=25)
    plt.plot(energy_values,muon_erates,color='orange',label=r'$\mu$')
    plt.plot(energy_values,nue_erates,color='g',label=r'$\nu_e$')
    plt.plot(energy_values,numu_erates,color='b',label=r'$\nu_\mu$')
    plt.plot(cut_values,nutau_erates,color='purple',label=r'$\nu_\tau$')
    plt.xlabel("Energy Cut",fontsize=20)
    plt.ylabel("Rate (Hz)",fontsize=20)
    plt.legend(fontsize=20)
    plt.savefig("%sEnergyCutRates.png"%(save_folder),bbox_inches='tight')
    plt.close()



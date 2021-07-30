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
import matplotlib.colors as colors
from sklearn.metrics import roc_curve, auc, roc_auc_score, recall_score
from sklearn.metrics import confusion_matrix

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

plot_energy = False
plot_main = False
print_rates = False
plot_rates = False
plot_after_threshold = True
plot_2d_vertex = False
plot_1dhists = False
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


# energy zenith prob_track vertex_x vertex_y vertex_z prob_muon
cnn_predict = np.array(predict[:,-1])
cnn_zenith = np.array(predict[:,1])
cnn_track = np.array(predict[:,2])
cnn_x = np.array(predict[:,3])
cnn_y = np.array(predict[:,4])
cnn_z = np.array(predict[:,5])
cnn_coszen = np.cos(cnn_zenith)
cnn_energy = np.array(predict[:,0])

true_energy = np.array(truth[:,0])*efactor
true_zenith = np.array(truth[:,1])
true_coszenith = np.cos(true_zenith)
true_x = np.array(truth[:,4])
true_y = np.array(truth[:,5])
true_z = np.array(truth[:,6])
true_CC = np.array(truth[:,11])
true_isCC = true_CC == 1

x_origin = np.ones((len(true_x)))*46.290000915527344
y_origin = np.ones((len(true_y)))*-34.880001068115234
true_r = np.sqrt( (true_x - x_origin)**2 + (true_y - y_origin)**2 )
cnn_r = np.sqrt( (cnn_x - x_origin)**2 + (cnn_y - y_origin)**2 )

muon_mask_test = (truth[:,9]) == 13
true_isMuon = np.array(muon_mask_test,dtype=bool)
numu_mask_test = (truth[:,9]) == 14
true_isNuMu = np.array(numu_mask_test,dtype=bool)
nue_mask_test = (truth[:,9]) == 12
true_isNuE = np.array(nue_mask_test,dtype=bool)
nutau_mask_test = (truth[:,9]) == 16
true_isNuTau = np.array(nutau_mask_test,dtype=bool)
nu_mask = np.logical_or(np.logical_or(numu_mask_test, nue_mask_test), nutau_mask_test)
true_isNu = np.array(nu_mask,dtype=bool)

if weights is not None:
    weights = weights[:,8]
    if sum(true_isNuMu) > 1:
        print("NuMu:",sum(true_isNuMu),sum(weights[true_isNuMu]))
        weights[true_isNuMu] = weights[true_isNuMu]/numu_files
        print(sum(weights[true_isNuMu]))
    if sum(true_isNuE) > 1:
        print("NuE:",sum(true_isNuE),sum(weights[true_isNuE]))
        weights[true_isNuE] = weights[true_isNuE]/nue_files
        print(sum(weights[true_isNuE]))
    if sum(muon_mask_test) > 1:
        print("Muon:",sum(true_isMuon),sum(weights[true_isMuon]))
        weights[true_isMuon] = weights[true_isMuon]/muon_files
        print(sum(weights[true_isMuon]))
    if sum(nutau_mask_test) > 1:
        print("NuTau:",sum(true_isNuTau),sum(weights[true_isNuTau]))
        weights[true_isNuTau] = weights[true_isNuTau]/nutau_files
        print(sum(weights[true_isNuTau]))
if no_weights:
    weights = np.ones(truth.shape[0])
    print("NOT USING ANY WEIGHTING")

# Cuts
no_cut = true_energy > 0
upgoing = cnn_coszen < 0.3
r_cut = cnn_r < 150
z_cut = np.logical_and(cnn_z > -500, cnn_z < -200)
vertex_cut = np.logical_and(r_cut, z_cut)
cnn_mu_cut = .75
cnn_nu = cnn_predict < cnn_mu_cut

from PlottingFunctionsClassification import ROC
from PlottingFunctionsClassification import plot_classification_hist
from PlottingFunctionsClassification import confusion_matrix
from PlottingFunctions import plot_2D_prediction
from PlottingFunctions import plot_bin_slices

mask_here = upgoing
mask_name_here="Upgoing"
if no_weights:
    mask_name_here += "_NoWeights"

print("Rates:\t Mu\t NuE\t NuMu\t NuTau\t Nu\n")
print("PreCut:\t %.2e\t %.2e\t %.2e\t %.2e\t %.2e\t"%(sum(weights[true_isMuon]),sum(weights[true_isNuE]),sum(weights[true_isNuMu]),sum(weights[true_isNuTau]),sum(weights[true_isNu])))

weights = weights[mask_here]
true_isMuon = true_isMuon[mask_here]
true_isNuMu = true_isNuMu[mask_here]
true_isNuE = true_isNuE[mask_here]
true_isNuTau = true_isNuTau[mask_here]
true_isNu = true_isNu[mask_here]
true_isCC = true_isCC[mask_here]
true_energy = true_energy[mask_here]
true_coszenith = true_coszenith[mask_here]
cnn_predict = cnn_predict[mask_here]
cnn_nu = cnn_nu[mask_here]
truth_PID = truth[:,9][mask_here]
true_r = true_r[mask_here]
true_z = true_z[mask_here]
cnn_r = cnn_r[mask_here]
cnn_z = cnn_z[mask_here]
cnn_coszen = cnn_coszen[mask_here]
cnn_energy = cnn_energy[mask_here]
no_cuts = true_energy > 0

print("%s Cut:\t %.2e\t %.2e\t %.2e\t %.2e\t %.2e\t"%(mask_name_here,sum(weights[true_isMuon]),sum(weights[true_isNuE]),sum(weights[true_isNuMu]),sum(weights[true_isNuTau]),sum(weights[true_isNu])))

save_folder += "/%s_28July21/"%mask_name_here
print("Working on %s"%save_folder)
if os.path.isdir(save_folder) != True:
    os.mkdir(save_folder)



if plot_energy:
    plt.figure(figsize=(10,7))
    emin = 0
    emax = 500
    plt.hist(true_energy,range=[emin,emax],bins=100,weights=weights)
    plt.xlabel("True Energy (GeV)")
    plt.title("True Energy Distribution")
    plt.ylabel("Rate (Hz)")
    plt.savefig("%sEnergyHist.png"%(save_folder),bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10,7))
    plt.hist(true_energy[true_isMuon],range=[emin,emax],bins=100,weights=weights[true_isMuon],label=r'$\mu$',alpha=0.5)
    plt.hist(true_energy[true_isNuMu],range=[emin,emax],bins=100,weights=weights[true_isNuMu],label=r'$\nu_\mu$',alpha=0.5)
    plt.hist(true_energy[true_isNuE],range=[emin,emax],bins=100,weights=weights[true_isNuE],label=r'$\nu_e$',alpha=0.5)
    plt.hist(true_energy[true_isNuTau],range=[emin,emax],bins=100,weights=weights[true_isNuTau],label=r'$\nu_\tau$',alpha=0.5)
    #plt.yscale('log')
    plt.ylabel("Rate (Hz)",fontsize=20)
    plt.xlabel("True Energy (GeV)",fontsize=20)
    plt.title("True Energy Distribution",fontsize=25)
    plt.legend()
    plt.savefig("%sEnergyParticleHist.png"%(save_folder),bbox_inches='tight')
    plt.close()

    plot_2D_prediction(true_energy[true_isNuMu&true_isCC],
                                cnn_energy[true_isNuMu&true_isCC],
                                weights=weights[true_isNuMu&true_isCC],
                                save=save, savefolder=save_folder,
                                bins=100,minval=0,maxval=200,axis_square=True,
                                variable="Energy", units="(GeV)",
                                reco_name="CNN",flavor="numu",sample="CC")
    plot_2D_prediction(true_energy[true_isNuMu&true_isCC],
                                cnn_energy[true_isNuMu&true_isCC],
                                weights=weights[true_isNuMu&true_isCC],
                                save=save, savefolder=save_folder,
                                bins=100,
                                variable="Energy", units="(GeV)",
                                reco_name="CNN",flavor="numu",sample="CC")
    plot_bin_slices(true_energy[true_isNuMu&true_isCC],
                            cnn_energy[true_isNuMu&true_isCC], 
                            weights=weights[true_isNuMu&true_isCC],
                            vs_predict = True,\
                            use_fraction = True, bins=20,
                            min_val=1, max_val=300,\
                            save=save, savefolder=save_folder,\
                            variable="Energy", units="(GeV)",
                            flavor="NuMu",sample="CC")

if plot_main:
    plt.figure(figsize=(10,7))
    plt.hist(cnn_predict[true_isMuon],bins=50,weights=weights[true_isMuon],label=r'$\mu$',alpha=0.5)
    plt.hist(cnn_predict[true_isNuMu],bins=50,weights=weights[true_isNuMu],label=r'$\nu_\mu$',alpha=0.5)
    plt.hist(cnn_predict[true_isNuE],bins=50,weights=weights[true_isNuE],label=r'$\nu_e$',alpha=0.5)
    plt.hist(cnn_predict[true_isNuTau],bins=50,weights=weights[true_isNuTau],label=r'$\nu_\tau$',alpha=0.5)
#plt.yscale('log')
    plt.ylabel("Rate (Hz)",fontsize=20)
    plt.xlabel("Muon Probability",fontsize=20)
    plt.title("Muon Probability Distribution",fontsize=25)
    plt.legend()
    plt.savefig("%sProbMuonParticleHist.png"%(save_folder),bbox_inches='tight')
    plt.close()


    ROC(true_isMuon,cnn_predict,mask=no_cuts,mask_name=mask_name_here,save=save,save_folder_name=save_folder)

    plot_classification_hist(true_isMuon,cnn_predict,mask=no_cuts,mask_name=mask_name_here, units="",weights=weights,bins=50,log=False,save=save,save_folder_name=save_folder,name_prob0 = "Neutrino", name_prob1 = "Muon")

    #plot_classification_hist(true_isMuon,cnn_predict,mask=no_cuts,mask_name=mask_name_here,units="",weights=weights,bins=50,log=False,save=save,save_folder_name=save_folder,normed=True,name_prob0 = "Neutrino", name_prob1 = "Muon")

    flip_predict = np.ones(len(cnn_predict)) - cnn_predict
    plot_classification_hist(true_isNu,flip_predict,mask=no_cuts,mask_name=mask_name_here, units="",weights=weights,bins=50,log=False,save=save,save_folder_name=save_folder,name_prob1 = "Neutrino", name_prob0 = "Muon")

#plot_2D_prediction(true_energy[true_isMuon], cnn_energy[true_isMuon],
#                                weights=weights[true_isMuon],
#                                save=save, savefolder=save_folder,
#                                bins=100,minval=0,maxval=emax,axis_square=True,
#                                variable="Energy", units="(GeV)",
#                                reco_name="CNN",flavor="numu",sample="CC&NC")
#plot_2D_prediction(true_energy[true_isMuon], cnn_energy[true_isMuon],
#                                weights=weights[true_isMuon],
#                                save=save, savefolder=save_folder,
#                                bins=100,
#                                variable="Energy", units="(GeV)",
#                                reco_name="CNN",flavor="numu",sample="CC&NC")

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


step=0.01
cut_values = np.arange(step,1+step,step)
muon_rates = []
numu_rates = []
nue_rates = []
nutau_rates = []
nu_rates = []
for muon_cut in cut_values:
    
    muon_rate, nue_rate, numu_rate, nutau_rate = return_rates(truth_PID, cnn_predict, weights, muon_cut)

    muon_rates.append(muon_rate)
    nue_rates.append(nue_rate)
    numu_rates.append(numu_rate)
    nutau_rates.append(nutau_rate)
    nu_rates.append(nue_rate + numu_rate + nutau_rate)

if print_rates:
    print(muon_rates)
    print(nue_rates)
    print(numu_rates)
    print(nutau_rates)
    print(nu_rates)
    print(cut_values)

if plot_rates:
    plt.figure(figsize=(10,7))
    plt.title("Testing Sample",fontsize=25)
    plt.plot(cut_values,muon_rates,color='orange',label=r'$\mu$')
    plt.plot(cut_values,nue_rates,color='g',label=r'$\nu_e$')
    plt.plot(cut_values,numu_rates,color='b',label=r'$\nu_\mu$')
    plt.plot(cut_values,nutau_rates,color='purple',label=r'$\nu_\tau$')
    plt.xlabel("Muon Probability Threshold",fontsize=20)
    plt.ylabel("Rate (Hz)",fontsize=20)
    plt.legend(fontsize=20)
    plt.savefig("%sThresholdRates.png"%(save_folder),bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10,7))
    plt.title("Muon vs Neutrino Rates",fontsize=25)
    plt.plot(nu_rates,muon_rates,color='orange',label="test")
    plt.xlabel("neutrino rates (Hz)",fontsize=20)
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
    plt.plot(flip_prob,muon_rates,color='orange',label=r'$\mu$')
    plt.plot(flip_prob,nue_rates,color='g',label=r'$\nu_e$')
    plt.plot(flip_prob,numu_rates,color='b',label=r'$\nu_\mu$')
    plt.plot(flip_prob,nutau_rates,color='purple',label=r'$\nu_\tau$')
    plt.xlabel("Neutrino Probability Threshold",fontsize=20)
    plt.ylabel("Rate (Hz)",fontsize=20)
    plt.legend(fontsize=20)
    plt.savefig("%sNuThresholdRates.png"%(save_folder),bbox_inches='tight')
    plt.close()



if plot_after_threshold:
    # Plot after threshold
    cnn_nu = cnn_predict <= cnn_mu_cut
    cnn_mu = cnn_predict > cnn_mu_cut
    correct_mu = np.logical_and(cnn_mu, true_isMuon)
    wrong_mu = np.logical_and(cnn_mu, true_isNu)
    correct_nu = np.logical_and(cnn_nu, true_isNu)
    wrong_nu = np.logical_and(cnn_nu, true_isMuon)

    print("Rates After Cut at %f:\t Mu\t NuE\t NuMu\t NuTau\t Nu\n"%cnn_mu_cut)
    r_cut = cnn_r < 125
    r_cut_out = cnn_r > 125
    z_cut = np.logical_and(cnn_z > -500, cnn_z < -200)
    z_cut_out = np.logical_or(cnn_z < -500, cnn_z > -200)
    vertex_cut = np.logical_and(r_cut, z_cut)
    vertex_cut_out = np.logical_or(r_cut_out, z_cut_out)
    print("%s Cut:\t %.2e\t %.2e\t %.2e\t %.2e\t %.2e\t"%("R",sum(weights[true_isMuon&r_cut]),sum(weights[true_isNuE&r_cut]),sum(weights[true_isNuMu&r_cut]),sum(weights[true_isNuTau&r_cut]),sum(weights[true_isNu&r_cut])))
    print("%s Cut:\t %.2e\t %.2e\t %.2e\t %.2e\t %.2e\t"%("Z",sum(weights[true_isMuon&z_cut]),sum(weights[true_isNuE&z_cut]),sum(weights[true_isNuMu&z_cut]),sum(weights[true_isNuTau&z_cut]),sum(weights[true_isNu&z_cut])))
    print("%s Cut:\t %.2e\t %.2e\t %.2e\t %.2e\t %.2e\t"%("R&Z",sum(weights[true_isMuon&vertex_cut]),sum(weights[true_isNuE&vertex_cut]),sum(weights[true_isNuMu&vertex_cut]),sum(weights[true_isNuTau&vertex_cut]),sum(weights[true_isNu&vertex_cut])))
    print("%s Cut:\t %.2e\t %.2e\t %.2e\t %.2e\t %.2e\t"%("Muon",sum(weights[true_isMuon&cnn_nu]),sum(weights[true_isNuE&cnn_nu]),sum(weights[true_isNuMu&cnn_nu]),sum(weights[true_isNuTau&cnn_nu]),sum(weights[true_isNu&cnn_nu])))
    print("%s Cut:\t %.2e\t %.2e\t %.2e\t %.2e\t %.2e\t"%("Muon & R & Z",sum(weights[true_isMuon&cnn_nu&vertex_cut]),sum(weights[true_isNuE&cnn_nu&vertex_cut]),sum(weights[true_isNuMu&cnn_nu&vertex_cut]),sum(weights[true_isNuTau&cnn_nu&vertex_cut]),sum(weights[true_isNu&cnn_nu&vertex_cut])))


    # R PLOT
    fig,ax = plt.subplots(2,1,figsize=(10,10))
    fig.suptitle("Reconstructed Radius After Cut",fontsize=25)
    ax[0].hist(cnn_r[true_isNu],bins=50,weights=weights[true_isNu],range=[0,300],label=r'Full $\nu$ Distribution',alpha=0.5)
    ax[0].hist(cnn_r[true_isNu&cnn_nu],bins=50,weights=weights[cnn_nu&true_isNu],range=[0,300],label=r'After Muon Cut',alpha=0.5)
    ax[0].hist(cnn_r[true_isNu&vertex_cut],bins=50,weights=weights[true_isNu&vertex_cut],range=[0,300],label=r'After Vertex Cut',alpha=0.5)
    ax[0].hist(cnn_r[true_isNu&cnn_nu&vertex_cut],bins=50,weights=weights[true_isNu&cnn_nu&vertex_cut],range=[0,300],label=r'After Vertex + Muon Cut',alpha=0.5)
    ax[0].axvline(125,color='k',linewidth=3,label="DeepCore")
    textstr = '\n'.join((
            r'Full Distribution: %.2e Hz' % (sum(weights[true_isNu])),
            r'Afer Muon Cut:     %.2e Hz' % (sum(weights[true_isNu&cnn_nu]) ),
            r'After Vertex Cut:  %.2e Hz' % (sum(weights[true_isNu&vertex_cut])),
            r'After Both Cuts:   %.2e Hz' % (sum(weights[true_isNu&cnn_nu&vertex_cut]) )))
    props = dict(boxstyle='round', facecolor='white', alpha=0.3)
    ax[0].text(0.5, 0.95, textstr, transform=ax[0].transAxes, fontsize=15,
        verticalalignment='top', bbox=props)
    #plt.yscale('log')
    ax[0].set_ylabel("Rate (Hz)",fontsize=20)
    ax[0].set_xlabel("Reconstructed Radius from String 36 (m)",fontsize=20)
    ax[0].set_title("True Nuetrino",fontsize=20)
    ax[0].legend(bbox_to_anchor=(1.04, 1), loc='upper left', ncol=1)

    fig.subplots_adjust(hspace=0.5)
    
    ax[1].hist(cnn_r[true_isMuon],bins=50,weights=weights[true_isMuon],range=[0,300],label=r'Full $\mu$ Distribution',alpha=0.5)
    ax[1].hist(cnn_r[true_isMuon&cnn_nu],bins=50,weights=weights[cnn_nu&true_isMuon],range=[0,300],label=r'After Muon Cut',alpha=0.5)
    ax[1].hist(cnn_r[true_isMuon&vertex_cut],bins=50,weights=weights[true_isMuon&vertex_cut],range=[0,300],label=r'After Vertex Cut',alpha=0.5)
    ax[1].hist(cnn_r[true_isMuon&cnn_nu&vertex_cut],bins=50,weights=weights[true_isMuon&cnn_nu&vertex_cut],range=[0,300],label=r'After Vertex + Muon Cut',alpha=0.5)
    ax[1].axvline(125,color='k',linewidth=3,label="DeepCore")
    textstr = '\n'.join((
            r'Full Distribution: %.2e Hz' % (sum(weights[true_isMuon])),
            r'Afer Muon Cut:     %.2e Hz' % (sum(weights[true_isMuon&cnn_nu]) ),
            r'After Vertex Cut:  %.2e Hz' % (sum(weights[true_isMuon&vertex_cut])),
            r'After Both Cuts:   %.2e Hz' % (sum(weights[true_isMuon&cnn_nu&vertex_cut]) )))
    props = dict(boxstyle='round', facecolor='white', alpha=0.3)
    ax[1].text(0.5, 0.95, textstr, transform=ax[1].transAxes, fontsize=15,
        verticalalignment='top', bbox=props)
    #plt.yscale('log')
    ax[1].set_ylabel("Rate (Hz)",fontsize=20)
    ax[1].set_xlabel("Reconstructed Radius from String 36 (m)",fontsize=20)
    ax[1].set_title("True Muon",fontsize=20)
    #ax[1].legend()
    plt.savefig("%sRDistHist.png"%(save_folder),bbox_inches='tight')
    plt.close()

    # R SQUARED
    a_min = 0
    a_max = 300*300
    fig,ax = plt.subplots(2,1,figsize=(10,10))
    fig.suptitle("Reconstructed Radius^2 After Cut",fontsize=25)
    ax[0].hist(cnn_r[true_isNu]*cnn_r[true_isNu],bins=50,weights=weights[true_isNu],range=[a_min,a_max],label=r'Full $\nu$ Distribution',alpha=0.5)
    ax[0].hist(cnn_r[true_isNu&cnn_nu]*cnn_r[true_isNu&cnn_nu],bins=50,weights=weights[cnn_nu&true_isNu],range=[a_min,a_max],label=r'After Muon Cut',alpha=0.5)
    ax[0].hist(cnn_r[true_isNu&vertex_cut]*cnn_r[true_isNu&vertex_cut],bins=50,weights=weights[true_isNu&vertex_cut],range=[a_min,a_max],label=r'After Vertex Cut',alpha=0.5)
    ax[0].hist(cnn_r[true_isNu&cnn_nu&vertex_cut]*cnn_r[true_isNu&cnn_nu&vertex_cut],bins=50,weights=weights[true_isNu&cnn_nu&vertex_cut],range=[a_min,a_max],label=r'After Vertex + Muon Cut',alpha=0.5)
    ax[0].axvline(125*125,color='k',linewidth=3,label="DeepCore")
    textstr = '\n'.join((
            r'Full Distribution: %.2e Hz' % (sum(weights[true_isNu])),
            r'Afer Muon Cut:     %.2e Hz' % (sum(weights[true_isNu&cnn_nu]) ),
            r'After Vertex Cut:  %.2e Hz' % (sum(weights[true_isNu&vertex_cut])),
            r'After Both Cuts:   %.2e Hz' % (sum(weights[true_isNu&cnn_nu&vertex_cut]) )))
    props = dict(boxstyle='round', facecolor='white', alpha=0.3)
    ax[0].text(0.5, 0.95, textstr, transform=ax[0].transAxes, fontsize=15,
        verticalalignment='top', bbox=props)
    #plt.yscale('log')
    ax[0].set_ylabel("Rate (Hz)",fontsize=20)
    ax[0].set_xlabel("Reconstructed Radius^2 from String 36 (m^2)",fontsize=20)
    ax[0].set_title("True Nuetrino",fontsize=20)
    ax[0].legend(bbox_to_anchor=(1.04, 1), loc='upper left', ncol=1)

    fig.subplots_adjust(hspace=0.5)
    
    ax[1].hist(cnn_r[true_isMuon]*cnn_r[true_isMuon],bins=50,weights=weights[true_isMuon],range=[a_min,a_max],label=r'Full $\mu$ Distribution',alpha=0.5)
    ax[1].hist(cnn_r[true_isMuon&cnn_nu]*cnn_r[true_isMuon&cnn_nu],bins=50,weights=weights[cnn_nu&true_isMuon],range=[a_min,a_max],label=r'After Muon Cut',alpha=0.5)
    ax[1].hist(cnn_r[true_isMuon&vertex_cut]*cnn_r[true_isMuon&vertex_cut],bins=50,weights=weights[true_isMuon&vertex_cut],range=[a_min,a_max],label=r'After Vertex Cut',alpha=0.5)
    ax[1].hist(cnn_r[true_isMuon&cnn_nu&vertex_cut]*cnn_r[true_isMuon&cnn_nu&vertex_cut],bins=50,weights=weights[true_isMuon&cnn_nu&vertex_cut],range=[a_min,a_max],label=r'After Vertex + Muon Cut',alpha=0.5)
    ax[1].axvline(125*125,color='k',linewidth=3,label="DeepCore")
    textstr = '\n'.join((
            r'Full Distribution: %.2e Hz' % (sum(weights[true_isMuon])),
            r'Afer Muon Cut:     %.2e Hz' % (sum(weights[true_isMuon&cnn_nu]) ),
            r'After Vertex Cut:  %.2e Hz' % (sum(weights[true_isMuon&vertex_cut])),
            r'After Both Cuts:   %.2e Hz' % (sum(weights[true_isMuon&cnn_nu&vertex_cut]) )))
    props = dict(boxstyle='round', facecolor='white', alpha=0.3)
    ax[1].text(0.5, 0.95, textstr, transform=ax[1].transAxes, fontsize=15,
        verticalalignment='top', bbox=props)
    #plt.yscale('log')
    ax[1].set_ylabel("Rate (Hz)",fontsize=20)
    ax[1].set_xlabel("Reconstructed Radius^2 from String 36 (m^2)",fontsize=20)
    ax[1].set_title("True Muon",fontsize=20)
    #ax[1].legend()
    plt.savefig("%sR2DistHist.png"%(save_folder),bbox_inches='tight')
    plt.close()


    # Zenith
    a_min = -1.
    a_max = 1.
    fig,ax = plt.subplots(2,1,figsize=(10,10))
    fig.suptitle("Reconstructed Cosine Zenith After Cut",fontsize=25)
    ax[0].hist(cnn_coszen[true_isNu],bins=50,weights=weights[true_isNu],range=[a_min,a_max],label=r'Full $\nu$ Distribution',alpha=0.5)
    ax[0].hist(cnn_coszen[true_isNu&cnn_nu],bins=50,weights=weights[cnn_nu&true_isNu],range=[a_min,a_max],label=r'After Muon Cut',alpha=0.5)
    ax[0].hist(cnn_coszen[true_isNu&vertex_cut],bins=50,weights=weights[true_isNu&vertex_cut],range=[a_min,a_max],label=r'After Vertex Cut',alpha=0.5)
    ax[0].hist(cnn_coszen[true_isNu&cnn_nu&vertex_cut],bins=50,weights=weights[true_isNu&cnn_nu&vertex_cut],range=[a_min,a_max],label=r'After Vertex + Muon Cut',alpha=0.5)
    textstr = '\n'.join((
            r'Full Distribution: %.2e Hz' % (sum(weights[true_isNu])),
            r'Afer Muon Cut:     %.2e Hz' % (sum(weights[true_isNu&cnn_nu]) ),
            r'After Vertex Cut:  %.2e Hz' % (sum(weights[true_isNu&vertex_cut])),
            r'After Both Cuts:   %.2e Hz' % (sum(weights[true_isNu&cnn_nu&vertex_cut]) )))
    props = dict(boxstyle='round', facecolor='white', alpha=0.3)
    ax[0].text(0.5, 0.95, textstr, transform=ax[0].transAxes, fontsize=15,
        verticalalignment='top', bbox=props)
    #plt.yscale('log')
    ax[0].set_ylabel("Rate (Hz)",fontsize=20)
    ax[0].set_xlabel("Reconstructed Cosine Zenith",fontsize=20)
    ax[0].set_title("True Nuetrino",fontsize=20)
    ax[0].legend(bbox_to_anchor=(1.04, 1), loc='upper left', ncol=1)

    fig.subplots_adjust(hspace=0.5)
    
    ax[1].hist(cnn_coszen[true_isMuon],bins=50,weights=weights[true_isMuon],range=[a_min,a_max],label=r'Full $\mu$ Distribution',alpha=0.5)
    ax[1].hist(cnn_coszen[true_isMuon&cnn_nu],bins=50,weights=weights[cnn_nu&true_isMuon],range=[a_min,a_max],label=r'After Muon Cut',alpha=0.5)
    ax[1].hist(cnn_coszen[true_isMuon&vertex_cut],bins=50,weights=weights[true_isMuon&vertex_cut],range=[a_min,a_max],label=r'After Vertex Cut',alpha=0.5)
    ax[1].hist(cnn_coszen[true_isMuon&cnn_nu&vertex_cut],bins=50,weights=weights[true_isMuon&cnn_nu&vertex_cut],range=[a_min,a_max],label=r'After Vertex + Muon Cut',alpha=0.5)
    textstr = '\n'.join((
            r'Full Distribution: %.2e Hz' % (sum(weights[true_isMuon])),
            r'Afer Muon Cut:     %.2e Hz' % (sum(weights[true_isMuon&cnn_nu]) ),
            r'After Vertex Cut:  %.2e Hz' % (sum(weights[true_isMuon&vertex_cut])),
            r'After Both Cuts:   %.2e Hz' % (sum(weights[true_isMuon&cnn_nu&vertex_cut]) )))
    props = dict(boxstyle='round', facecolor='white', alpha=0.3)
    ax[1].text(0.5, 0.95, textstr, transform=ax[1].transAxes, fontsize=15,
        verticalalignment='top', bbox=props)
    #plt.yscale('log')
    ax[1].set_ylabel("Rate (Hz)",fontsize=20)
    ax[1].set_xlabel("Reconstructed Cosine Zenith",fontsize=20)
    ax[1].set_title("True Muon",fontsize=20)
    #ax[1].legend()
    plt.savefig("%sCosZenDistHist.png"%(save_folder),bbox_inches='tight')
    plt.close()


    # ENERGY
    a_min = 0
    a_max = 200
    fig,ax = plt.subplots(2,1,figsize=(10,10))
    fig.suptitle("Reconstructed Energy After Cut",fontsize=25)
    ax[0].hist(cnn_energy[true_isNu],bins=50,weights=weights[true_isNu],range=[a_min,a_max],label=r'Full $\nu$ Distribution',alpha=0.5)
    ax[0].hist(cnn_energy[true_isNu&cnn_nu],bins=50,weights=weights[cnn_nu&true_isNu],range=[a_min,a_max],label=r'After Muon Cut',alpha=0.5)
    ax[0].hist(cnn_energy[true_isNu&vertex_cut],bins=50,weights=weights[true_isNu&vertex_cut],range=[a_min,a_max],label=r'After Vertex Cut',alpha=0.5)
    ax[0].hist(cnn_energy[true_isNu&cnn_nu&vertex_cut],bins=50,weights=weights[true_isNu&cnn_nu&vertex_cut],range=[a_min,a_max],label=r'After Vertex + Muon Cut',alpha=0.5)
    textstr = '\n'.join((
            r'Full Distribution: %.2e Hz' % (sum(weights[true_isNu])),
            r'Afer Muon Cut:     %.2e Hz' % (sum(weights[true_isNu&cnn_nu]) ),
            r'After Vertex Cut:  %.2e Hz' % (sum(weights[true_isNu&vertex_cut])),
            r'After Both Cuts:   %.2e Hz' % (sum(weights[true_isNu&cnn_nu&vertex_cut]) )))
    props = dict(boxstyle='round', facecolor='white', alpha=0.3)
    ax[0].text(0.5, 0.95, textstr, transform=ax[0].transAxes, fontsize=15,
        verticalalignment='top', bbox=props)
    #plt.yscale('log')
    ax[0].set_ylabel("Rate (Hz)",fontsize=20)
    ax[0].set_xlabel("Reconstructed Energy (GeV)",fontsize=20)
    ax[0].set_title("True Nuetrino",fontsize=20)
    ax[0].legend(bbox_to_anchor=(1.04, 1), loc='upper left', ncol=1)

    fig.subplots_adjust(hspace=0.5)
    
    ax[1].hist(cnn_energy[true_isMuon],bins=50,weights=weights[true_isMuon],range=[a_min,a_max],label=r'Full $\mu$ Distribution',alpha=0.5)
    ax[1].hist(cnn_energy[true_isMuon&cnn_nu],bins=50,weights=weights[cnn_nu&true_isMuon],range=[a_min,a_max],label=r'After Muon Cut',alpha=0.5)
    ax[1].hist(cnn_energy[true_isMuon&vertex_cut],bins=50,weights=weights[true_isMuon&vertex_cut],range=[a_min,a_max],label=r'After Vertex Cut',alpha=0.5)
    ax[1].hist(cnn_energy[true_isMuon&cnn_nu&vertex_cut],bins=50,weights=weights[true_isMuon&cnn_nu&vertex_cut],range=[a_min,a_max],label=r'After Vertex + Muon Cut',alpha=0.5)
    textstr = '\n'.join((
            r'Full Distribution: %.2e Hz' % (sum(weights[true_isMuon])),
            r'Afer Muon Cut:     %.2e Hz' % (sum(weights[true_isMuon&cnn_nu]) ),
            r'After Vertex Cut:  %.2e Hz' % (sum(weights[true_isMuon&vertex_cut])),
            r'After Both Cuts:   %.2e Hz' % (sum(weights[true_isMuon&cnn_nu&vertex_cut]) )))
    props = dict(boxstyle='round', facecolor='white', alpha=0.3)
    ax[1].text(0.5, 0.95, textstr, transform=ax[1].transAxes, fontsize=15,
        verticalalignment='top', bbox=props)
    #plt.yscale('log')
    ax[1].set_ylabel("Rate (Hz)",fontsize=20)
    ax[1].set_xlabel("Reconstructed Energy (GeV)",fontsize=20)
    ax[1].set_title("True Muon",fontsize=20)
    #ax[1].legend()
    plt.savefig("%sEnergyDistHist.png"%(save_folder),bbox_inches='tight')
    plt.close()

    # Z
    a_min = -800
    a_max = 0
    fig,ax = plt.subplots(2,1,figsize=(10,10))
    fig.suptitle("Reconstructed Z After Cut",fontsize=25)
    ax[0].hist(cnn_z[true_isNu],bins=50,weights=weights[true_isNu],range=[a_min,a_max],label=r'Full $\nu$ Distribution',alpha=0.5)
    ax[0].hist(cnn_z[true_isNu&cnn_nu],bins=50,weights=weights[cnn_nu&true_isNu],range=[a_min,a_max],label=r'After Muon Cut',alpha=0.5)
    ax[0].hist(cnn_z[true_isNu&vertex_cut],bins=50,weights=weights[true_isNu&vertex_cut],range=[a_min,a_max],label=r'After Vertex Cut',alpha=0.5)
    ax[0].hist(cnn_z[true_isNu&cnn_nu&vertex_cut],bins=50,weights=weights[true_isNu&cnn_nu&vertex_cut],range=[a_min,a_max],label=r'After Vertex + Muon Cut',alpha=0.5)
    ax[0].axvline(-500,color='k',linewidth=3,label="DeepCore")
    ax[0].axvline(-200,color='k',linewidth=3,label="DeepCore")
    textstr = '\n'.join((
            r'Full Distribution: %.2e Hz' % (sum(weights[true_isNu])),
            r'Afer Muon Cut:     %.2e Hz' % (sum(weights[true_isNu&cnn_nu]) ),
            r'After Vertex Cut:  %.2e Hz' % (sum(weights[true_isNu&vertex_cut])),
            r'After Both Cuts:   %.2e Hz' % (sum(weights[true_isNu&cnn_nu&vertex_cut]) )))
    props = dict(boxstyle='round', facecolor='white', alpha=0.3)
    ax[0].text(0.5, 0.95, textstr, transform=ax[0].transAxes, fontsize=15,
        verticalalignment='top', bbox=props)
    #plt.yscale('log')
    ax[0].set_ylabel("Rate (Hz)",fontsize=20)
    ax[0].set_xlabel("Reconstructed Z (m)",fontsize=20)
    ax[0].set_title("True Nuetrino",fontsize=20)
    ax[0].legend(bbox_to_anchor=(1.04, 1), loc='upper left', ncol=1)

    fig.subplots_adjust(hspace=0.5)
    
    ax[1].hist(cnn_z[true_isMuon],bins=50,weights=weights[true_isMuon],range=[a_min,a_max],label=r'Full $\mu$ Distribution',alpha=0.5)
    ax[1].hist(cnn_z[true_isMuon&cnn_nu],bins=50,weights=weights[cnn_nu&true_isMuon],range=[a_min,a_max],label=r'After Muon Cut',alpha=0.5)
    ax[1].hist(cnn_z[true_isMuon&vertex_cut],bins=50,weights=weights[true_isMuon&vertex_cut],range=[a_min,a_max],label=r'After Vertex Cut',alpha=0.5)
    ax[1].hist(cnn_z[true_isMuon&cnn_nu&vertex_cut],bins=50,weights=weights[true_isMuon&cnn_nu&vertex_cut],range=[a_min,a_max],label=r'After Vertex + Muon Cut',alpha=0.5)
    ax[1].axvline(-500,color='k',linewidth=3,label="DeepCore")
    ax[1].axvline(-200,color='k',linewidth=3,label="DeepCore")
    textstr = '\n'.join((
            r'Full Distribution: %.2e Hz' % (sum(weights[true_isMuon])),
            r'Afer Muon Cut:     %.2e Hz' % (sum(weights[true_isMuon&cnn_nu]) ),
            r'After Vertex Cut:  %.2e Hz' % (sum(weights[true_isMuon&vertex_cut])),
            r'After Both Cuts:   %.2e Hz' % (sum(weights[true_isMuon&cnn_nu&vertex_cut]) )))
    props = dict(boxstyle='round', facecolor='white', alpha=0.3)
    ax[1].text(0.5, 0.95, textstr, transform=ax[1].transAxes, fontsize=15,
        verticalalignment='top', bbox=props)
    #plt.yscale('log')
    ax[1].set_ylabel("Rate (Hz)",fontsize=20)
    ax[1].set_xlabel("Reconstructed Z (m)",fontsize=20)
    ax[1].set_title("True Muon",fontsize=20)
    #ax[1].legend()
    plt.savefig("%sZDistHist.png"%(save_folder),bbox_inches='tight')
    plt.close()

    """
    plt.figure(figsize=(10,7))
    plt.hist(cnn_r*cnn_r,bins=50,range=[0,300*300],weights=weights,label=r'Full Distribution',alpha=0.5)
    plt.hist(cnn_r[cnn_nu]*cnn_r[cnn_nu],bins=50,range=[0,300*300],weights=weights[cnn_nu],label=r'After Cut',alpha=0.5)
    plt.axvline(125*125,color='k',linewidth=3,label="DeepCore")
    #plt.yscale('log')
    plt.ylabel("Rate (Hz)",fontsize=20)
    plt.xlabel("Reconstructed Radius^2 from String 36 (m^2)",fontsize=20)
    plt.title("Reconstructed Radius^2 After Cut",fontsize=25)
    plt.legend()
    plt.savefig("%sR2DistHist.png"%(save_folder),bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10,7))
    plt.hist(cnn_r[true_isMuon]*cnn_r[true_isMuon],range=[0,300*300],bins=50,weights=weights[true_isMuon],label=r'Full $\mu$ Distribution',alpha=0.5)
    plt.hist(cnn_r[cnn_nu&true_isMuon]*cnn_r[cnn_nu&true_isMuon],bins=50,range=[0,300*300],weights=weights[cnn_nu&true_isMuon],label=r'After CNN Cut',alpha=0.5)
    plt.axvline(125*125,color='k',linewidth=3,label="DeepCore")
    #plt.yscale('log')
    plt.ylabel("Rate (Hz)",fontsize=20)
    plt.xlabel("Reconstructed Radius^2 from String 36 (m)",fontsize=20)
    plt.title("Reconstructed Radius^2 After Cut for Muon",fontsize=25)
    plt.legend()
    plt.savefig("%sR2MuonDistHist.png"%(save_folder),bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10,7))
    plt.hist(cnn_z,bins=50,range=[-800,400],weights=weights,label=r'Full Distribution',alpha=0.5)
    plt.hist(cnn_z[cnn_nu],bins=50,range=[-800,400],weights=weights[cnn_nu],label=r'After Cut',alpha=0.5)
    #plt.yscale('log')
    plt.ylabel("Rate (Hz)",fontsize=20)
    plt.xlabel("Reconstructed Z Depth (m)",fontsize=20)
    plt.title("Reconstructed Z After Cut",fontsize=25)
    plt.legend()
    plt.savefig("%sZDistHist.png"%(save_folder),bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10,7))
    plt.hist(cnn_z[true_isMuon],bins=50,range=[-800,400],weights=weights[true_isMuon],label=r'Full $\mu$ Distribution',alpha=0.5)
    plt.hist(cnn_z[cnn_nu&true_isMuon],bins=50,range=[-800,400],weights=weights[cnn_nu&true_isMuon],label=r'After CNN Cut',alpha=0.5)
#plt.yscale('log')
    plt.ylabel("Rate (Hz)",fontsize=20)
    plt.xlabel("Reconstructed Z Depth (m)",fontsize=20)
    plt.title("Reconstructed Z After Cut for Muon",fontsize=25)
    plt.legend()
    plt.savefig("%sZMuonDistHist.png"%(save_folder),bbox_inches='tight')
    plt.close()
    """

    if plot_2d_vertex:
        a_mask = true_isMuon
        a_mask_name = "True Muon"
        xmin = 0
        xmax = 300*300
        ymin = -700
        ymax = 100
        plt.figure(figsize=(10,7))
        cts,xbin,ybin,img = plt.hist2d(cnn_r[a_mask]*cnn_r[a_mask], cnn_z[a_mask], bins=50,range=[[xmin,xmax],[ymin,ymax]],cmap='viridis_r', norm=colors.LogNorm(), weights=weights[a_mask], cmin=1e-12)
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('Rate (Hz)', rotation=90)
        plt.ylabel("Reconstructed Z-Depth (m)",fontsize=20)
        plt.xlabel("Resconstructed Radius^2 from String 36 (m^2)",fontsize=20)
        plt.title("CNN Position for %s"%a_mask_name,fontsize=25)
        plt.axvline(125*125,color='k',linewidth=3,label="DeepCore")
        plt.axhline(-200,color='k',linewidth=3)
        plt.axhline(-500,color='k',linewidth=3)
        plt.legend()
        a_plot_name = "RZ2DHist"
        a_plot_name += "%s"%a_mask_name.replace(" ","")
        print(a_plot_name)
        plt.savefig("%s%s.png"%(save_folder,a_plot_name),bbox_inches='tight')
        plt.close()
        
        a_mask = true_isNu
        a_mask_name = "True Neutrino"
        plt.figure(figsize=(10,7))
        cts,xbin,ybin,img = plt.hist2d(cnn_r[a_mask]*cnn_r[a_mask], cnn_z[a_mask], bins=50,range=[[xmin,xmax],[ymin,ymax]],cmap='viridis_r', norm=colors.LogNorm(), weights=weights[a_mask], cmin=1e-12)
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('Rate (Hz)', rotation=90)
        plt.ylabel("Reconstructed Z-Depth (m)",fontsize=20)
        plt.xlabel("Resconstructed Radius^2 from String 36 (m^2)",fontsize=20)
        plt.title("CNN Position for %s"%a_mask_name,fontsize=25)
        plt.axvline(125*125,color='k',linewidth=3,label="DeepCore")
        plt.axhline(-200,color='k',linewidth=3)
        plt.axhline(-500,color='k',linewidth=3)
        plt.legend()
        a_plot_name = "RZ2DHist"
        a_plot_name += "%s"%a_mask_name.replace(" ","")
        print(a_plot_name)
        plt.savefig("%s%s.png"%(save_folder,a_plot_name),bbox_inches='tight')
        plt.close()
        
        a_mask = true_isMuon & cnn_nu
        a_mask_name = "True Muon After Muon Cut"
        plt.figure(figsize=(10,7))
        cts,xbin,ybin,img = plt.hist2d(cnn_r[a_mask]*cnn_r[a_mask], cnn_z[a_mask], bins=50,range=[[xmin,xmax],[ymin,ymax]],cmap='viridis_r', norm=colors.LogNorm(), weights=weights[a_mask], cmin=1e-12)
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('Rate (Hz)', rotation=90)
        plt.ylabel("Reconstructed Z-Depth (m)",fontsize=20)
        plt.xlabel("Resconstructed Radius^2 from String 36 (m^2)",fontsize=20)
        plt.title("CNN Position for %s"%a_mask_name,fontsize=25)
        plt.axvline(125*125,color='k',linewidth=3,label="DeepCore")
        plt.axhline(-200,color='k',linewidth=3)
        plt.axhline(-500,color='k',linewidth=3)
        plt.legend()
        a_plot_name = "RZ2DHist"
        a_plot_name += "%s"%a_mask_name.replace(" ","")
        print(a_plot_name)
        plt.savefig("%s%s.png"%(save_folder,a_plot_name),bbox_inches='tight')
        plt.close()
        
        a_mask = true_isNu & cnn_nu
        a_mask_name = "True Nu After Muon Cut"
        plt.figure(figsize=(10,7))
        cts,xbin,ybin,img = plt.hist2d(cnn_r[a_mask]*cnn_r[a_mask], cnn_z[a_mask], bins=50,range=[[xmin,xmax],[ymin,ymax]],cmap='viridis_r', norm=colors.LogNorm(), weights=weights[a_mask], cmin=1e-12)
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('Rate (Hz)', rotation=90)
        plt.ylabel("Reconstructed Z-Depth (m)",fontsize=20)
        plt.xlabel("Resconstructed Radius^2 from String 36 (m^2)",fontsize=20)
        plt.title("CNN Position for %s"%a_mask_name,fontsize=25)
        plt.axvline(125*125,color='k',linewidth=3,label="DeepCore")
        plt.axhline(-200,color='k',linewidth=3)
        plt.axhline(-500,color='k',linewidth=3)
        plt.legend()
        a_plot_name = "RZ2DHist"
        a_plot_name += "%s"%a_mask_name.replace(" ","")
        print(a_plot_name)
        plt.savefig("%s%s.png"%(save_folder,a_plot_name),bbox_inches='tight')
        plt.close()
        
    plt.figure(figsize=(10,7))
    plt.hist(cnn_coszen,bins=50,range=[-1,1],weights=weights,label=r'Full Distribution',alpha=0.5)
    plt.hist(cnn_coszen[cnn_nu],bins=50,range=[-1,1],weights=weights[cnn_nu],label=r'After Cut',alpha=0.5)
    #plt.yscale('log')
    plt.ylabel("Rate (Hz)",fontsize=20)
    plt.xlabel("Reconstructed Cosine Zenith",fontsize=20)
    plt.title("Reconstructed Cosine Zenith",fontsize=25)
    plt.legend()
    plt.savefig("%sCosZenithDistHist.png"%(save_folder),bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10,7))
    plt.hist(cnn_coszen[true_isMuon],bins=50,range=[-1,1],weights=weights[true_isMuon],label=r'Full $\mu$ Distribution',alpha=0.5)
    plt.hist(cnn_coszen[cnn_nu&true_isMuon],bins=50,range=[-1,1],weights=weights[cnn_nu&true_isMuon],label=r'After CNN Cut',alpha=0.5)
    #plt.yscale('log')
    plt.ylabel("Rate (Hz)",fontsize=20)
    plt.xlabel("Reconstructed Cosine Zenith",fontsize=20)
    plt.title("Reconstructed Cosine Zenith for True Muon",fontsize=25)
    plt.legend()
    plt.savefig("%sCosZenithMuonDistHist.png"%(save_folder),bbox_inches='tight')
    plt.close()
    
    if plot_1dhists:
        from PlottingFunctions import plot_distributions
        a_mask = correct_nu
        a_title = "True Neutrino: CNN Correct"
        plot_distributions(true_energy[a_mask], cnn_energy[a_mask],
                              weights=weights[a_mask],\
                              save=save, savefolder=save_folder, 
                              variable="Reconstructed Energy",
                              units= "(GeV)",title=a_title,
                              minval=0,maxval=500,bins=100) 
        plot_distributions(true_coszenith[a_mask], cnn_coszen[a_mask],
                              weights=weights[a_mask],\
                              save=save, savefolder=save_folder, 
                              reco_name = "Retro", variable="Reconstructed CosZenith",
                              units= "",title=a_title,
                              minval=-1,maxval=1,bins=100) 
        plot_distributions(true_r[a_mask]*true_r[a_mask], cnn_r[a_mask]*cnn_r[a_mask],
                              weights=weights[a_mask],\
                              save=save, savefolder=save_folder, 
                              reco_name = "Retro", variable="Reconstructed R^2",
                              units= "(m^2)",xline=125*125,xline_label="DeepCore",
                              minval=0,maxval=300*300,
                              bins=100,title=a_title)
        plot_distributions(true_r[a_mask], cnn_r[a_mask],
                              weights=weights[a_mask],\
                              save=save, savefolder=save_folder,
                              variable="Reconstructed R",
                              units= "(m)",xline=125,xline_label="DeepCore",
                              minval=0,maxval=300,
                              bins=100,title=a_title)
        plot_distributions(true_z[a_mask], cnn_z[a_mask],
                              weights=weights[a_mask],\
                              save=save, savefolder=save_folder,
                              reco_name = "Retro", variable="Reconstructed Z",
                              units= "(m)",title=a_title,
                              minval=-1000,maxval=200,bins=100)

    print(sum(weights[correct_mu]),sum(weights[wrong_mu]),sum(weights[wrong_nu]),sum(weights[correct_nu]))

    confusion_matrix(true_isMuon, cnn_predict, 0.39, mask=no_cuts, mask_name=mask_name_here, weights=weights, save=True, save_folder_name=save_folder, name_prob1 = "Muon", name_prob0 = "Neutrino")


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



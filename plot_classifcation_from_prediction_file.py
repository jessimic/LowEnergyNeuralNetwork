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
import matplotlib.pyplot as plt
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
parser.add_argument("--no_old_reco", default=False,action='store_true',
                        dest='no_old_reco',help="no old reco")
args = parser.parse_args()

input_file = args.input_file
efactor = args.efactor
numu_files = args.numu
nue_files = args.nue
no_old_reco = args.no_old_reco
save_folder = args.output_dir
save = True

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

#cnn_energy = np.array(predict[:,0])
cnn_predict = np.array(predict[:,0])
#cnn_coszenith = np.cos(np.array(predict[:,2]))
#true
true_energy = np.array(truth[:,0])*efactor
true_x = np.array(truth[:,4])
true_y = np.array(truth[:,5])
true_z = np.array(truth[:,6])
true_CC = np.array(truth[:,11])
true_isTrack = np.array(truth[:,8])
muon_mask_test = (truth[:,9]) == 13
true_isMuon = np.array(muon_mask_test,dtype=int)
x_origin = np.ones((len(true_x)))*46.290000915527344
y_origin = np.ones((len(true_y)))*-34.880001068115234
true_r = np.sqrt( (true_x - x_origin)**2 + (true_y - y_origin)**2 )
no_cut = true_energy > 0

maskCNNE = np.ones(len(true_energy),dtype=bool) #np.logical_and(cnn_energy > 5., cnn_energy < 100.)
maskCNNE2 = np.ones(len(true_energy),dtype=bool) #np.logical_and(cnn_energy > 1., cnn_energy < 100.)
print("NOT REAL CNN ENERGY")
maskCNNZenith = np.ones(len(true_energy),dtype=bool) #cnn_coszenith <= 0.3
print("NOT REAL CNN ZENITH")

#retro
if reco is not None:
    retro_energy = np.array(reco[:,0])
    retro_zenith = np.array(reco[:,1])
    retro_time = np.array(reco[:,3])
    reco_x = np.array(reco[:,4])
    reco_y = np.array(reco[:,5])
    reco_z = np.array(reco[:,6])
    retro_coszen = np.cos(retro_zenith)
    retro_PID_all =  np.array(reco[:,12])
    retro_PID_up =  np.array(reco[:,13])
    reco_r = np.sqrt( (reco_x - x_origin)**2 + (reco_y - y_origin)**2 )

#additional
if info is not None:
    prob_nu = info[:,1]
    coin_muon = info[:,0]
    true_ndoms = info[:,2]
    fit_success = info[:,3]
    noise_class = info[:,4]
    nhit_doms = info[:,5]
    n_top15 = info[:,6]
    n_outer = info[:,7]
    prob_nu2 = info[:,8]
    hits8 = info[:,9]
    #hlc_x = info[:,10]
    #hlc_y = info[:,11]
    #hlc_z = info[:,12]
    #hlc_rho = np.sqrt( (hlc_x - x_origin)**2 + (hlc_y - y_origin)**2 )


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

variable = "Classification"
units = ""
log = False
bins = 50

#True Masks
maskNONE = true_energy > 0.
assert sum(maskNONE)==len(true_energy), "Some true energy < 0? Check!"
maskTrack = true_isTrack == 1
maskCascade = true_isTrack == 0
maskCC = true_CC == 1
maskZ = np.logical_and(true_z > -505, true_z < 192)
maskR = true_r < 90.
maskDC = np.logical_and(maskZ,maskR)
maskE = np.logical_and(true_energy > 5., true_energy < 100.)

#Info Masks
if info is not None:
    maskHits8 = hits8 == 1
    maskNu = prob_nu > 0.4
    maskNoise = noise_class > 0.95
    masknhit = nhit_doms > 2.5
    maskntop = n_top15 < 2.5
    masknouter = n_outer < 7.5
    maskHits = np.logical_and(np.logical_and(masknhit, maskntop), masknouter)
    maskClass = np.logical_and(maskNu,maskNoise)
    maskMC = np.logical_and(maskHits,maskClass)
    #maskHLCZ = np.logical_and(hlc_z > -500., hlc_z < -200.)
    #maskHLCR = hlc_rho < 300.
    maskHLCVertex = np.ones(len(maskMC),dtype=bool) #np.logical_and(maskHLCZ, maskHLCR)
    print("NOT REAL HLC")

#Retro Masks
if reco is not None:
    maskRecoZ = np.logical_and(reco_z > -500., reco_z < -200.)
    maskRecoR = reco_r < 300.
    maskRecoDC = np.logical_and(maskRecoZ, maskRecoR)
    maskRetroZenith = np.cos(retro_zenith) <= 0.3
    maskRetroEnergy = np.logical_and(retro_energy >= 5., retro_energy <= 300.)
    maskRetroTime = retro_time < 14500.
    maskRetro = np.logical_and(np.logical_and(maskRetroZenith, maskRetroEnergy), maskRetroTime)
    maskANA = np.logical_and(np.logical_and(np.logical_and(maskRecoDC,  maskRetro), maskMC),maskHits8)

from PlottingFunctionsClassification import ROC
from PlottingFunctionsClassification import confusion_matrix
from PlottingFunctionsClassification import plot_classification_hist
from PlottingFunctionsClassification import precision

#cut_list = [np.logical_and(maskHits8,maskCC), np.logical_and(maskANA,maskCC), np.logical_and(maskHLCVertex, np.logical_and(np.logical_and(maskCNNE, maskCC),maskHits8)), np.logical_and(maskHLCVertex, np.logical_and(np.logical_and(maskCNNE, maskCC),maskHits8)),np.logical_and(maskCNNZenith,np.logical_and(maskHLCVertex, np.logical_and(np.logical_and(maskCNNE, maskCC),maskHits8))),maskHits8,no_cut]
#cut_names = ["Weighted_Hits8CC", "Weighted_CC", "WeightedCNN5100_HLCVertex_Hits8CC", "WeightedCNN5100_HLCVertex_Hits8CC", "WeightedCNN5100_ZenithHLCVertex_Hits8CC","Hits8","no_cuts"]

#print(sum(np.logical_and(maskCNNZenith,mask_nue)), sum(np.logical_and(mask_nue,maskHLCVertex)), sum(np.logical_and(mask_nue,maskCNNE)), sum(np.logical_and(mask_nue,maskCC)), sum(np.logical_and(mask_nue,maskHits8)), sum(mask_nue) )

cut_list = [no_cut]
cut_names = ["NoCuts"]

emin=5
emax=200
estep=1
a_mask = cut_list[-1]
mask_here = a_mask
mask_name_here = cut_names[-1]
#fit_success_cut = np.array(fit_success[a_mask],dtype=bool)
save_folder += "/%s/"%mask_name_here
print(sum(a_mask),len(a_mask))
print(min(true_energy[a_mask]),max(true_energy[a_mask]))
print(sum(true_isTrack),len(true_isTrack) -sum(true_isTrack), sum(np.logical_and(mask_nue,a_mask)))

print("Working on %s"%save_folder)

if os.path.isdir(save_folder) != True:
    os.mkdir(save_folder)

#All events
do_general_plots = True
if do_general_plots:

    plt.figure(figsize=(10,7))
    muon_energy = true_energy[true_isMuon]
    plt.hist(muon_energy,range=[0,100],label="muon")
    numu_mask_test = (truth[:,9]) == 14
    true_isNuMu = np.array(numu_mask_test,dtype=int)
    numu_energy = true_energy[true_isNuMu]
    plt.hist(numu_energy,range=[0,100],label="numu")
    nue_mask_test = (truth[:,9]) == 12
    true_isNuMu = np.array(nue_mask_test,dtype=int)
    nue_energy = true_energy[true_isNuMu]
    plt.hist(nue_energy,range=[0,100],label="nue")
    plt.savefig("%sEnergyParticleHist.png"%(save_folder))
    plt.close()

    #ROC(true_isTrack,cnn_predict,mask=mask_here,mask_name=mask_name_here,reco=retro_PID_up,save=save,save_folder_name=save_folder)
    ROC(true_isMuon,cnn_predict,mask=mask_here,mask_name=mask_name_here,save=save,save_folder_name=save_folder)
    #plot_classification_hist(true_isTrack,cnn_predict,mask=mask_here,mask_name=mask_name_here, variable="CNN Classification",units="",bins=50,log=False,save=save,save_folder_name=save_folder)
    plot_classification_hist(true_isMuon,cnn_predict,mask=mask_here,mask_name=mask_name_here, variable="Probabiliy Muon",units="",weights=weights,bins=50,log=False,save=save,save_folder_name=save_folder)
    #plot_classification_hist(true_isTrack,retro_PID_up,mask=mask_here,mask_name=mask_name_here, variable="L7_PIDClassifier_Upgoing_ProbTrack",units="",weights=weights,bins=50,log=False,save=save,save_folder_name=save_folder)
    #confusion_matrix(true_isTrack, cnn_predict, track_threshold, mask=mask_here, mask_name=mask_name_here, weights=None,save=save, save_folder_name=save_folder)
    #confusion_matrix(true_isTrack, cnn_predict, track_threshold, mask=mask_here, mask_name=mask_name_here, weights=weights,save=save, save_folder_name=save_folder)
    #precision(true_isTrack, cnn_predict, reco=retro_PID_all, mask=mask_here, mask_name = mask_name_here,save=save,save_folder_name=save_folder)

do_energy_auc = False
if do_energy_auc:
# Energy vs AUC
    energy_auc = []
    reco_energy_auc = []
    energy_thres = []
    energy_recall = []
    energy_range = np.arange(emin,emax, estep)
    
    truth_Track = true_isTrack[a_mask]
    cnn_array = cnn_predict[a_mask]
    if reco is not None:
        reco_array = retro_PID_up[a_mask]
        #reco_array = retro_PID_all[a_mask]
    AUC_title = "AUC vs. True Energy - Analysis Cuts"
    save_name_extra = ""
    for energy_bin in energy_range:
        current_mask = np.logical_and(true_energy[a_mask] > energy_bin, true_energy[a_mask] < energy_bin + 1)
        
        energy_auc.append(roc_auc_score(truth_Track[current_mask], cnn_array[current_mask]))
        if reco is not None:
            reco_energy_auc.append(roc_auc_score(truth_Track[current_mask], reco_array[current_mask]))
        
        #best_threshold = ROC(true_isTrack,cnn_predict,mask=current_mask,mask_name="",save=False,save_folder_name=save_folder)
        #energy_thres.append(best_threshold[0])
        
        #predictionCascade = cnn_predict < best_threshold
        #predictionTrack = cnn_predict >= best_threshold
        #energy_recall.append(recall_score(predictionCascade[current_mask], predictionTrack[current_mask]))


    plt.figure(figsize=(10,7))
    plt.title(AUC_title,fontsize=25)
    plt.ylabel("AUC",fontsize=20)
    plt.xlabel("True Energy (GeV)",fontsize=20)
    plt.plot(energy_range, energy_auc, 'b-',label="CNN")
    if reco is not None:
        plt.plot(energy_range, reco_energy_auc, color="orange",linestyle="-",label="Retro")
        plt.legend(loc="upper left",fontsize=20)
        save_name_extra += "_compareRetro"
    plt.savefig("%sAUCvsEnergy%s.png"%(save_folder,save_name_extra))
    
    #plt.figure(figsize=(10,7))
    #plt.title("Thres vs. True Energy",fontsize=25)
    #plt.ylabel("Threshold",fontsize=20)
    #plt.xlabel("True Energy (GeV)",fontsize=20)
    #plt.plot(energy_range, energy_thres, 'b-')
    #plt.savefig("%sThresvsEnergy.png"%(save_folder))
    
    #plt.figure(figsize=(10,7))
    #plt.title("Sensitivity vs. True Energy",fontsize=25)
    #plt.ylabel("Sensitivity TP/(TP + FN)",fontsize=20)
    #plt.xlabel("True Energy (GeV)",fontsize=20)
    #plt.plot(energy_range, energy_thres, 'b-')
    #plt.savefig("%SensvsEnergy.png"%(save_folder))
    
do_energy_range = False
if do_energy_range:
    #Break down energy range
    energy_ranges = [5, 10, 20, 30, 40, 60, 80, 100, 150, 200]
    #energy_ranges = [5, 10, 20, 30, 40, 60, 80, 100, 200]
    for e_index in range(0,len(energy_ranges)-1):
        energy_start = energy_ranges[e_index]
        energy_end = energy_ranges[e_index+1]
        current_mask = np.logical_and(true_energy > energy_start, true_energy < energy_end)
        current_name = "True Energy %i-%i GeV"%(energy_start,energy_end)
        plot_classification_hist(true_isTrack,cnn_predict,mask=current_mask,mask_name=current_name, weights=weights, variable="CNN Classification",units="",bins=50,log=False,save=save,save_folder_name=save_folder)
        plot_classification_hist(true_isTrack,retro_PID_up,mask=current_mask,mask_name=current_name, weights=weights, variable="Retro Classification",units="",bins=50,log=False,save=save,save_folder_name=save_folder)
        ROC(true_isTrack,cnn_predict,reco=retro_PID_up,mask=current_mask,mask_name=current_name,save=save,save_folder_name=save_folder)
        #best_threshold = best_threshold[0]
        #confusion_matrix(true_isTrack, cnn_predict, best_threshold, mask=current_mask, mask_name=current_name, weights=None,save=save, save_folder_name=save_folder)
        #confusion_matrix(true_isTrack, cnn_predict, best_threshold, mask=current_mask, mask_name=current_name, weights=weights,save=save, save_folder_name=save_folder)


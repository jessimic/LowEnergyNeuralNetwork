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
    weights = None
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

cnn_predict = np.array(predict[:,0])
#true
true_energy = np.array(truth[:,0])*efactor
true_x = np.array(truth[:,4])
true_y = np.array(truth[:,5])
true_z = np.array(truth[:,6])
true_CC = np.array(truth[:,11])
true_isTrack = np.array(truth[:,8])
x_origin = np.ones((len(true_x)))*46.290000915527344
y_origin = np.ones((len(true_y)))*-34.880001068115234
true_r = np.sqrt( (true_x - x_origin)**2 + (true_y - y_origin)**2 )

#retro
if reco is not None:
    retro_energy = np.array(reco[:,0])
    retro_zenith = np.array(reco[:,1])
    retro_time = np.array(reco[:,3])
    reco_x = np.array(reco[:,4])
    reco_y = np.array(reco[:,5])
    reco_z = np.array(reco[:,6])
    retro_coszen = np.cos(retro_zenith)
    retro_PID_all =  np.array(reco[:,8])
    retro_PID_up =  np.array(reco[:,9])
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


#weights
if weights is not None:
    weights = np.array(weights[:,8])
    #modify by number of files
    mask_numu = np.array(truth[:,9]) == 14
    mask_nue = np.array(truth[:,9]) == 12
    if sum(mask_numu) > 1:
        weights[mask_numu] = weights[mask_numu]/numu_files
    if sum(mask_nue) > 1:
        weights[mask_nue] = weights[mask_nue])/nue_files

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

#All events
best_threshold = ROC(true_isTrack,cnn_predict,mask=None,mask_name="",save=save,save_folder_name=save_folder)
#confusion_matrix(true_isTrack, cnn_predict, best_threshold, mask=None, mask_name="", weights=None,save=save, save_folder_name=save_folder)
#confusion_matrix(true_isTrack, cnn_predict, best_threshold, mask=None, mask_name="", weights=weights,save=save, save_folder_name=save_folder)
plot_classification_hist(true_isTrack,cnn_predict,mask=None,mask_name="", variable="Classification",units="",bins=50,log=False,save=save,save_folder_name=save_folder)

#plt.figure(figsize=(10,7))
#plt.title("Threshold",fontsize=25)
#plt.xlabel("True Energy (GeV)",fontsize=20)
#plt.plot(energy_range, energy_thres, 'b-')
#plt.savefig("%sThresvsEnergy.png"%(save_folder))

do_energy_auc = True
if do_energy_auc:
# Energy vs AUC
    energy_auc = []
    energy_thres = []
    energy_recall = []
    energy_range = np.arange(5,200, 1.)
    for energy_bin in energy_range:
        current_mask = np.logical_and(true_energy > energy_bin, true_energy < energy_bin + 1)
        
        energy_auc.append(roc_auc_score(true_isTrack[current_mask], cnn_predict[current_mask]))
        
        #best_threshold = ROC(true_isTrack,cnn_predict,mask=current_mask,mask_name="",save=False,save_folder_name=save_folder)
        #energy_thres.append(best_threshold[0])
        
        #predictionCascade = cnn_predict < best_threshold
        #predictionTrack = cnn_predict >= best_threshold
        #energy_recall.append(recall_score(predictionCascade[current_mask], predictionTrack[current_mask]))


    plt.figure(figsize=(10,7))
    plt.title("AUC vs. True Energy",fontsize=25)
    plt.ylabel("AUC",fontsize=20)
    plt.xlabel("True Energy (GeV)",fontsize=20)
    plt.plot(energy_range, energy_auc, 'b-')
    plt.savefig("%sAUCvsEnergy.png"%(save_folder))
    
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
    
do_energy_range = True
if do_energy_range:
    #Break down energy range
    energy_ranges = [5, 10, 20, 30, 40, 60, 80, 100, 150, 200]
    for e_index in range(0,len(energy_ranges)-1):
        energy_start = energy_ranges[e_index]
        energy_end = energy_ranges[e_index+1]
        current_mask = np.logical_and(true_energy > energy_start, true_energy < energy_end)
        current_name = "Energy %i to %i GeV"%(energy_start,energy_end)
        plot_classification_hist(true_isTrack,cnn_predict,mask=current_mask,mask_name=current_name, variable="Classification",units="",bins=50,log=False,save=save,save_folder_name=save_folder)
        best_threshold = ROC(true_isTrack,cnn_predict,mask=current_mask,mask_name=current_name,save=save,save_folder_name=save_folder)
        #best_threshold = best_threshold[0]
        #confusion_matrix(true_isTrack, cnn_predict, best_threshold, mask=current_mask, mask_name=current_name, weights=None,save=save, save_folder_name=save_folder)
        #confusion_matrix(true_isTrack, cnn_predict, best_threshold, mask=current_mask, mask_name=current_name, weights=weights,save=save, save_folder_name=save_folder)


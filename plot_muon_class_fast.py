import numpy as np
import matplotlib.pyplot as plt
import glob
import matplotlib.colors as colors
from scipy.interpolate import UnivariateSpline
from scipy import interpolate
import scipy.stats
import itertools
#import wquantiles as wq
import argparse
import os, sys
from matplotlib import ticker
import matplotlib as mpl
label_size = 15
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_file",type=str,default="prediction_values.hdf5",
                    dest="input_file", help="Name for test only input file")
parser.add_argument("-d", "--path",type=str,default='/mnt/home/micall12/LowEnergyNeuralNetwork/output_plots/',
                    dest="path", help="path to model dir")
parser.add_argument("-m","--model_name",default=None,
                    dest="model_name",help="Name for model directory")
parser.add_argument("-o", "--outdir",default=None,
                    dest="output_dir", help="path of ouput file, if different that the output_plot model dir")
parser.add_argument("--savename", default=None,
                    dest="savename", help="additional dir in the output_dir to save the plots under")
parser.add_argument("--numu", default=None,
                    dest="numu", help="number of numu files to use")
parser.add_argument("--nue", default=None,
                    dest="nue", help="number of nue files to use")
parser.add_argument("--nutau", default=None,
                    dest="nutau", help="number of nutau files to use")
parser.add_argument("--muon", default=None,
                    dest="muon", help="number of muon files to use")
parser.add_argument("--no_nutau",default=False,action='store_true',
                    dest="no_nutau",help="remove nutau events")
parser.add_argument("--i3",default=False,action='store_true',
                    dest="i3",help="flag if inputting i3 files (not hdf5)")
parser.add_argument("--given_threshold",default=None,
                    dest="given_threshold",help="Define cut value (0-1) to apply cut for confusion matrix, if not use, finds closest to true neutrino at 80%")
args = parser.parse_args()

filename = args.input_file
path = args.path
model_name = args.model_name
i3 = args.i3
if i3:
    files = path + filename
    full_path = sorted(glob.glob(files))
    print("Using %i i3 files"%len(full_path))
else:
    full_path = path + "/" + model_name + "/" + filename
    print("Using file %s"%full_path)

if args.output_dir is None:
    outdir = path + "/" + model_name + "/"
else:
    outdir = args.output_dir + "/"
if args.savename is not None:
    outdir = outdir + "/" + args.savename + "/"

given_threshold = args.given_threshold
if given_threshold:
    given_threshold = float(given_threshold)


save = True
save_folder = outdir
print("Saving to %s"%save_folder)
if os.path.isdir(save_folder) != True:
        os.mkdir(save_folder)

numu_files = args.numu
nue_files = args.nue
nutau_files = args.nutau
muon_files = args.muon

if i3:
    #Find (and edit) number of files
    numu_file_list = list(filter(lambda x: "pass2.14" in x, full_path))
    nue_file_list = list(filter(lambda x: "pass2.12" in x, full_path))
    muon_file_list = list(filter(lambda x: "pass2.13" in x, full_path))
    nutau_file_list = list(filter(lambda x: "pass2.16" in x, full_path))
    if numu_files is None:
        numu_files = len(numu_file_list)
        print("Including all %i NuMu files"%numu_files)
    else:
        numu_files = int(numu_files)
        print("Cutting NuMu files to include first %i files, from %s to %s"%(numu_files,numu_file_list[0],numu_file_list[numu_files-1]))
    if nue_files is None:
        nue_files = len(nue_file_list)
        print("Including all %i NuE files"%nue_files)
    else:
        nue_files = int(nue_files)
        print("Cutting NuE files to include LAST %i files, from %s to %s"%(nue_files,nue_file_list[nue_files-1],nue_file_list[-1]))
    if nutau_files is None:
        nutau_files = len(nutau_file_list)
        print("Including all %i NuTau files"%nutau_files)
    else:
        nutau_files = int(nuta_files)
        print("Cutting NuTau files to include first %i files, from %s to %s"%(nutau_files,nutau_file_list[0],nutau_file_list[nutau_files-1]))
    if muon_files is None:
        muon_files = len(muon_file_list)
        print("Including all %i Muon files"%muon_files)
    else:
        muon_files = int(muon_files)
        print("Cutting Muon files to include LAST %i files, from %s to %s"%(muon_files,muon_file_list[muon_files-1],muon_file_list[-1]))

    print("Using %i numu files, %i nue files, %i nutau files, %i muon files"%(numu_files, nue_files, nutau_files, muon_files))
    numu_file_list = numu_file_list[:numu_files]
    nue_file_list = nue_file_list[-nue_files:]
    nutau_file_list = nutau_file_list[:nutau_files]
    muon_file_list = muon_file_list[-muon_files:]
    full_path = np.concatenate((numu_file_list,nue_file_list,nutau_file_list,muon_file_list))


    from read_cnn_i3_files import read_i3_files
    variable_list = ["energy", "prob_track", "zenith", "vertex_x", "vertex_y", "vertex_z", "prob_muon", "prob_muon_v2"] 
    predict, truth, old_reco, info, raw_weights, input_features_DC, input_features_IC= read_i3_files(full_path,variable_list)

    cnn_prob_mu = predict[:,6]


else:
    import h5py
    f = h5py.File(full_path, "r")
    list(f.keys())
    truth = f["Y_test_use"][:]
    predict = f["Y_predicted"][:]
    try:
        info = f["additional_info"][:]
    except:
        info = None
    try:
        raw_weights = f["weights_test"][:]
    except:
        raw_weights = None
    f.close()
    del f

    cnn_prob_mu = np.array(predict[:,:,0][-1])
    
    numu_files =97
    nue_files = 91
    muon_files = 1999
    nutau_files = 45

#Seperate by PID and reweight
cnn_prob_nu = 1-cnn_prob_mu

true_PID = truth[:,9]

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

weights = raw_weights[:,8]
if weights is not None:
    if sum(true_isNuMu) > 1:
        #print("NuMu:",sum(true_isNuMu),sum(weights[true_isNuMu]))
        weights[true_isNuMu] = weights[true_isNuMu]/numu_files
        #print(sum(weights[true_isNuMu]))
    if sum(true_isNuE) > 1:
        #print("NuE:",sum(true_isNuE),sum(weights[true_isNuE]))
        weights[true_isNuE] = weights[true_isNuE]/nue_files
        #print(sum(weights[true_isNuE]))
    if sum(true_isMuon) > 1:
        #print("Muon:",sum(true_isMuon),sum(weights[true_isMuon]))
        weights[true_isMuon] = weights[true_isMuon]/muon_files
        #print(sum(weights[true_isMuon]))
    if sum(nutau_mask_test) > 1:
        #print("NuTau:",sum(true_isNuTau),sum(weights[true_isNuTau]))
        weights[true_isNuTau] = weights[true_isNuTau]/nutau_files
        #print(sum(weights[true_isNuTau]))

if args.no_nutau:
    print("Removing nutau events from testing sample!!!")
    not_NuTau = np.invert(true_isNuTau)
    cnn_prob_mu = cnn_prob_mu[not_NuTau]
    cnn_prob_nu = cnn_prob_nu[not_NuTau]
    weights = weights[not_NuTau]
    true_isMuon = true_isMuon[not_NuTau]
    true_isNu = true_isNu[not_NuTau]
    true_isNuMu = true_isNuMu[not_NuTau]
    true_isNuE = true_isNuE[not_NuTau]

true_all = np.ones(len(weights),dtype=bool)

from PlottingFunctionsClassification import plot_classification_hist
from PlottingFunctionsClassification import ROC
from PlottingFunctionsClassification import my_confusion_matrix

plot_classification_hist(true_isNu,cnn_prob_nu,mask=true_all,
                        mask_name="No Cuts", units="",bins=50,
                        weights=weights, log=False,save=save,
                        save_folder_name=save_folder,
                        name_prob1 = "Neutrino", name_prob0 = "Muon")

threshold1, threshold0, auc = ROC(true_isNu,cnn_prob_nu,
                                mask=None,mask_name="No Cuts",
                                save=save,save_folder_name=save_folder,
                                variable="Probability Neutrino")
#Confusion Matrix
total = sum(weights[true_isNuMu])
if given_threshold is None:
    try_cuts = np.arange(0.01,1.00,0.01)
    fraction_numu = []
    for mu_cut in try_cuts:
        cut_attempt = cnn_prob_mu <= mu_cut
        cut_mask = np.logical_and(true_isNuMu, cut_attempt)
        fraction_numu.append(sum(weights[cut_mask])/total)
#Find closest to 80.61% to match L6 Teseting
    eighty_array = np.ones(len(fraction_numu),dtype=float)*0.8061
    nearest_to_80 = abs(fraction_numu - eighty_array)
    best_index = nearest_to_80.argmin()
    best_mu_cut = try_cuts[best_index]
#print(fraction_num)
#print(nearest_to_80)
    print("best index %i, cut value %f"%(best_index,best_mu_cut))
else:
    best_mu_cut = given_threshold
    print("given cut value %f"%(best_mu_cut))

cnn_binary_class = cnn_prob_mu <= best_mu_cut
percent_save = my_confusion_matrix(true_isNu, cnn_binary_class, weights,
                    mask=None,title="CNN Muon Cut",
                    save=save,save_folder_name=save_folder)


print("AUC: %.3f"%auc)
#save percent order: (CNN Muon, True Neutrino), (CNN Muon, True Muon), (CNN Neutrino, True Neutrino), (CNN Neutrino, True Muon)
print("CNN Neutrino, True Neutrino: %.2f"%percent_save[2])
print("CNN Neutrino, True Muon: %.2f"%percent_save[3])

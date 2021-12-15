import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib.colors as colors
from scipy.interpolate import UnivariateSpline
from scipy import interpolate
import scipy.stats
import itertools
#import wquantiles as wq
from sklearn.metrics import confusion_matrix
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
parser.add_argument("--no_nutau",default=False,action='store_true',
                    dest="no_nutau",help="Remove nutau events")
args = parser.parse_args()

filename = args.input_file
output_plots = args.path
model_name = args.model_name
full_path = output_plots + "/" + model_name + "/" + filename
if args.output_dir is None:
    outdir = output_plots + "/" + model_name + "/"
else:
    outdir = args.output_dir + "/"
if args.savename is not None:
    outdir = outdir + "/" + args.savename + "/"

print("Using file %s"%full_path)

save = True
save_folder = outdir
print("Saving to %s"%save_folder)
if os.path.isdir(save_folder) != True:
        os.mkdir(save_folder)

f = h5py.File(full_path, "r")
list(f.keys())
truth1 = f["Y_test_use"][:]
predict1 = f["Y_predicted"][:]
try:
    info1 = f["additional_info"][:]
except:
    info1 = None
try:
    raw_weights1 = f["weights_test"][:]
except:
    raw_weights1 = None
f.close()
del f

cnn_prob_mu1 = np.array(predict1[:,:,0][-1])
cnn_prob_nu1 = 1-cnn_prob_mu1

true_PID1 = truth1[:,9]

muon_mask_test1 = (true_PID1) == 13
true_isMuon1 = np.array(muon_mask_test1,dtype=bool)
numu_mask_test1 = (true_PID1) == 14
true_isNuMu1 = np.array(numu_mask_test1,dtype=bool)
nue_mask_test1 = (true_PID1) == 12
true_isNuE1 = np.array(nue_mask_test1,dtype=bool)
nutau_mask_test1 = (true_PID1) == 16
true_isNuTau1 = np.array(nutau_mask_test1,dtype=bool)
nu_mask1 = np.logical_or(np.logical_or(numu_mask_test1, nue_mask_test1), nutau_mask_test1)
true_isNu1 = np.array(nu_mask1,dtype=bool)

weights1 = raw_weights1[:,8]

numu_files1 =97
nue_files1 = 91
muon_files1 = 1999
nutau_files1 = 45
if weights1 is not None:
    if sum(true_isNuMu1) > 1:
        #print("NuMu:",sum(true_isNuMu1),sum(weights1[true_isNuMu1]))
        weights1[true_isNuMu1] = weights1[true_isNuMu1]/numu_files1
        #print(sum(weights1[true_isNuMu1]))
    if sum(true_isNuE1) > 1:
        #print("NuE:",sum(true_isNuE1),sum(weights1[true_isNuE1]))
        weights1[true_isNuE1] = weights1[true_isNuE1]/nue_files1
        #print(sum(weights1[true_isNuE1]))
    if sum(true_isMuon1) > 1:
        #print("Muon:",sum(true_isMuon1),sum(weights1[true_isMuon1]))
        weights1[true_isMuon1] = weights1[true_isMuon1]/muon_files1
        #print(sum(weights1[true_isMuon1]))
    if sum(nutau_mask_test1) > 1:
        #print("NuTau:",sum(true_isNuTau1),sum(weights1[true_isNuTau1]))
        weights1[true_isNuTau1] = weights1[true_isNuTau1]/nutau_files1
        #print(sum(weights1[true_isNuTau1]))

if args.no_nutau:
    print("Removing nutau events from testing sample!!!")
    not_NuTau1 = np.invert(true_isNuTau1)
    cnn_prob_mu1 = cnn_prob_mu1[not_NuTau1]
    cnn_prob_nu1 = cnn_prob_nu1[not_NuTau1]
    weights1 = weights1[not_NuTau1]
    true_isMuon1 = true_isMuon1[not_NuTau1]
    true_isNu1 = true_isNu1[not_NuTau1]
    true_isNuMu1 = true_isNuMu1[not_NuTau1]
    true_isNuE1 = true_isNuE1[not_NuTau1]

weights_squared1 = weights1*weights1
true_all1 = np.ones(len(weights1),dtype=bool)

from PlottingFunctionsClassification import plot_classification_hist
from PlottingFunctionsClassification import ROC
from PlottingFunctionsClassification import my_confusion_matrix

plot_classification_hist(true_isNu1,cnn_prob_nu1,mask=true_all1,
                        mask_name="No Cuts", units="",bins=50,
                        weights=weights1, log=False,save=save,
                        save_folder_name=save_folder,
                        name_prob1 = "Neutrino", name_prob0 = "Muon")

threshold1, threshold0, auc = ROC(true_isNu1,cnn_prob_nu1,
                                mask=None,mask_name="No Cuts",
                                save=save,save_folder_name=save_folder,
                                variable="Probability Neutrino")
#Confusion Matrix
total = sum(weights1[true_isNuMu1])
try_cuts = np.arange(0.01,1.00,0.01)
fraction_numu = []
for mu_cut in try_cuts:
    cut_attempt = cnn_prob_mu1 <= mu_cut
    cut_mask = np.logical_and(true_isNuMu1, cut_attempt)
    fraction_numu.append(sum(weights1[cut_mask])/total)
#Find closest to 80.61% to match L6 Teseting
eighty_array = np.ones(len(fraction_numu),dtype=float)*0.8061
nearest_to_80 = abs(fraction_numu - eighty_array)
best_index = nearest_to_80.argmin()
best_mu_cut = try_cuts[best_index]
#print(fraction_num)
#print(nearest_to_80)
print(best_index,best_mu_cut)

cnn_binary_class = cnn_prob_mu1 <= best_mu_cut
percent_save = my_confusion_matrix(true_isNu1, cnn_binary_class, weights1,
                    mask=None,title="CNN Muon Cut",
                    save=save,save_folder_name=save_folder)


print("AUC: %.3f"%auc)
#save percent order: (CNN Muon, True Neutrino), (CNN Muon, True Muon), (CNN Neutrino, True Neutrino), (CNN Neutrino, True Muon)
print("CNN Neutrino, True Neutrino: %.2f"%percent_save[2])
print("CNN Neutrino, True Muon: %.2f"%percent_save[3])

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
parser.add_argument("--save_output",default=False,action='store_true',
                    dest="save_output",help="flag if saving i3 file output together into to hdf5")
parser.add_argument("--given_threshold",default=None,
                    dest="given_threshold",help="Define cut value (0-1) to apply cut for confusion matrix, if not use, finds closest to true neutrino at 80%")
parser.add_argument("--split_flavor",default=False,action='store_true',                    dest="split_flavor",help="flag to plot all flavors separately")
parser.add_argument("--muon_index", default=None,
                    dest="muon_index", help="index in hdf5 where muon classifier output is stored")
args = parser.parse_args()

filename = args.input_file
path = args.path
model_name = args.model_name
#save_output_data = args.save_output
i3 = args.i3
if i3:
    files = path + filename
    full_path = sorted(glob.glob(files))
    print("Given %i i3 files"%len(full_path))
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

split_flavor = args.split_flavor
muon_index = args.muon_index
if muon_index is not None:
    muon_index = int(muon_index)

save = True
save_folder = outdir
print("Saving to %s"%save_folder)
if os.path.isdir(save_folder) != True:
        os.mkdir(save_folder)

numu_files = args.numu
nue_files = args.nue
nutau_files = args.nutau
muon_files = args.muon
if numu_files is not None:
    numu_files = int(numu_files)
if nue_files is not None:
    nue_files = int(nue_files)
if nutau_files is not None:
    nutau_files = int(nutau_files)
if muon_files is not None:
    muon_files = int(muon_files)

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
        print("Cutting NuE files to include LAST %i files, from %s to %s"%(nue_files,nue_file_list[-nue_files],nue_file_list[-1]))
    if nutau_files is None:
        nutau_files = len(nutau_file_list)
        print("Including all %i NuTau files"%nutau_files)
    else:
        nutau_files = int(nutau_files)
        print("Cutting NuTau files to include first %i files, from %s to %s"%(nutau_files,nutau_file_list[0],nutau_file_list[nutau_files-1]))
    if muon_files is None:
        muon_files = len(muon_file_list)
        print("Including all %i Muon files"%muon_files)
    else:
        muon_files = int(muon_files)
        print("Cutting Muon files to include LAST %i files, from %s to %s"%(muon_files,muon_file_list[-muon_files],muon_file_list[-1]))

    print("Using %i numu files, %i nue files, %i nutau files, %i muon files"%(numu_files, nue_files, nutau_files, muon_files))
    numu_file_list = numu_file_list[:numu_files]
    nue_file_list = nue_file_list[-nue_files:]
    nutau_file_list = nutau_file_list[:nutau_files]
    muon_file_list = muon_file_list[-muon_files:]
    full_path = np.concatenate((numu_file_list,nue_file_list,nutau_file_list,muon_file_list))


    from read_cnn_i3_files import read_i3_files
    variable_list = ["energy", "prob_track", "zenith", "vertex_x", "vertex_y", "vertex_z", "prob_muon", "nDOM"] 
    predict, truth, old_reco, info, raw_weights, input_features_DC, input_features_IC= read_i3_files(full_path,variable_list)

    cnn_prob_mu = predict[:,6]
    cnn_energy = predict[:,0]
    true_energy = np.array(truth[:,0])


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
    
    true_energy = np.array(truth[:,0])
    if muon_index is None:
        muon_index = 6
    #try:
    #    cnn_prob_mu = np.array(predict[:,:,0][-1])
    #except:
    cnn_energy = np.array(predict[:,0])
    cnn_prob_track = np.array(predict[:,1])
    cnn_prob_mu = np.array(predict[:,muon_index])

    if numu_files is None:
        numu_files = 1518 #294 #97
    if nue_files is None:
        nue_files = 602 #92 #91
    if muon_files is None:
        muon_files = 19391 #600 #1999
    if nutau_files is None:
        nutau_files = 334 #45
    print("Using given numbers %i numu files, %i nue files, %i muon files, %i nutau files for weighting"%(numu_files,nue_files,muon_files,nutau_files))

#Seperate by PID and reweight
cnn_prob_nu = 1-cnn_prob_mu

true_PID = truth[:,9]
true_isTrack = truth[:,8]
true_isTrack = true_isTrack == 1

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
print("Nu:",sum(true_isNu))

assert sum(true_isTrack[true_isMuon]) ==0, "Muon saved as tracks"
assert sum(true_isTrack[true_isNuE]) ==0, "NuE saved as tracks"
assert sum(true_isTrack[true_isNuTau]) ==0, "NuTau saved as tracks"

weights = raw_weights[:,8]
if weights is not None:
    if sum(true_isNuMu) > 1:
        weights[true_isNuMu] = weights[true_isNuMu]/numu_files
        print("NuMu:",sum(true_isNuMu),sum(weights[true_isNuMu]))
    if sum(true_isNuE) > 1:
        weights[true_isNuE] = weights[true_isNuE]/nue_files
        print("NuE:",sum(true_isNuE),sum(weights[true_isNuE]))
    if sum(true_isMuon) > 1:
        weights[true_isMuon] = weights[true_isMuon]/muon_files
        print("Muon:",sum(true_isMuon),sum(weights[true_isMuon]))
    if sum(nutau_mask_test) > 1:
        weights[true_isNuTau] = weights[true_isNuTau]/nutau_files
        print("NuTau:",sum(true_isNuTau),sum(weights[true_isNuTau]))

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
MuNuTau = np.logical_or(true_isMuon, true_isNuTau)
MuNuMu = np.logical_or(true_isMuon, true_isNuMu)
MuNuE = np.logical_or(true_isMuon, true_isNuE)

from PlottingFunctionsClassification import plot_classification_hist
from PlottingFunctionsClassification import ROC
from PlottingFunctionsClassification import my_confusion_matrix

zoom_min = 0.97
zoom_max = 1.0

plot_classification_hist(true_isNu,cnn_prob_nu,mask=true_all,
                        mask_name="No Cuts", units="",bins=50,
                        weights=weights, log=True,save=save,
                        save_folder_name=save_folder,
                        name_prob1 = "Neutrino", name_prob0 = "Muon")

plot_classification_hist(true_isNu,cnn_prob_nu,mask=true_all,
                        mask_name="No Cuts", units="",bins=50,
                        weights=weights, log=True,save=save,
                        save_folder_name=save_folder,
                        savename="CNNMuon_Near1",xmin=zoom_min,xmax=zoom_max,
                        name_prob1 = "Neutrino", name_prob0 = "Muon")

plot_classification_hist(true_isTrack,cnn_prob_track,mask=true_isNu,
                        mask_name="Neutrino", units="",bins=50,
                        weights=weights, log=True,save=save,
                        save_folder_name=save_folder,
                        name_prob1 = "Track", name_prob0 = "Cascade")

if split_flavor:
    if sum(true_isNuMu) > 0:
        plot_classification_hist(true_isNu,cnn_prob_nu,mask=MuNuMu,
                            mask_name="NuMu", units="",bins=50,
                            weights=weights, log=False,save=save,
                            save_folder_name=save_folder,
                            name_prob1 = "Neutrino", name_prob0 = "Muon")

        plot_classification_hist(true_isNu,cnn_prob_nu,mask=MuNuMu,
                            mask_name="NuMu", units="",bins=50,
                            weights=weights, log=False,save=save,
                            save_folder_name=save_folder,
                            savename="CNNMuon_Near1",xmin=zoom_min,xmax=zoom_max,
                            name_prob1 = "Neutrino", name_prob0 = "Muon")

    if sum(true_isNuE) > 0:
        plot_classification_hist(true_isNu,cnn_prob_nu,mask=MuNuE,
                            mask_name="NuE", units="",bins=50,
                            weights=weights, log=False,save=save,
                            save_folder_name=save_folder,
                            name_prob1 = "Neutrino", name_prob0 = "Muon")

    if sum(true_isNuTau) > 0:
        plot_classification_hist(true_isNu,cnn_prob_nu,mask=MuNuTau,
                            mask_name="NuTau", units="",bins=50,
                            weights=weights, log=False,save=save,
                            save_folder_name=save_folder,
                        name_prob1 = "Neutrino", name_prob0 = "Muon")

threshold1, threshold0, auc = ROC(true_isNu,cnn_prob_nu,
                                mask=None,mask_name="No Cuts",
                                save=save,save_folder_name=save_folder,
                                variable="Probability Neutrino")

threshold1_pid, threshold0_pid, auc_pid = ROC(true_isTrack,cnn_prob_track,
                                mask=true_isNu,mask_name="Neutrino",
                                save=save,save_folder_name=save_folder,
                                variable="Probability Track")
#Confusion Matrix
total = sum(weights[true_isNuMu])
total_mu = sum(weights[true_isMuon])
if given_threshold is None:
    #try_cuts = np.arange(0.01,1.00,0.01)
    try_cuts = np.arange(0.001,0.10,0.001)
    #try_cuts = np.arange(0.161,0.180,0.001)
    fraction_numu = []
    fraction_muon = []
    for mu_cut in try_cuts:
        cut_attempt = cnn_prob_mu <= mu_cut
        cut_mask = np.logical_and(true_isNuMu, cut_attempt)
        cut_mask_mu = np.logical_and(true_isMuon, cut_attempt)
        fraction_numu.append(sum(weights[cut_mask])/total)
        fraction_muon.append(sum(weights[cut_mask_mu])/total_mu)
#Find closest to 80.61% to match L6 Teseting
    eighty_array = np.ones(len(fraction_numu),dtype=float)*0.8061
    nearest_to_80 = abs(fraction_numu - eighty_array)
    best_index = nearest_to_80.argmin()
    best_mu_cut = try_cuts[best_index]
    print(try_cuts)
    print(fraction_numu)
    print(fraction_muon)
    #print(nearest_to_80)
    print("best index %i, cut value %f"%(best_index,best_mu_cut))
else:
    best_mu_cut = given_threshold
    print("given cut value %f"%(best_mu_cut))

cnn_binary_mu = cnn_prob_mu <= best_mu_cut
percent_save,percent_error = my_confusion_matrix(true_isNu, cnn_binary_mu, 
                    weights,
                    mask=None,title="CNN Muon Cut",
                    save=save,save_folder_name=save_folder)


print("Muon AUC: %.3f"%auc)
print("PID AUC: %.3f"%auc_pid)
#save percent order: (CNN Muon, True Neutrino), (CNN Muon, True Muon), (CNN Neutrino, True Neutrino), (CNN Neutrino, True Muon)
print("CNN Neutrino, True Neutrino: %.2f +/- %.2f"%(percent_save[2],percent_error[2]))
print("CNN Muon, True Muon: %.2f +/- %.2f"%(percent_save[1],percent_error[1]))

if split_flavor:
    if sum(true_isNuMu) > 0:
        percent_save,percent_error = my_confusion_matrix(true_isNu[MuNuMu],
                    cnn_binary_mu[MuNuMu], weights[MuNuMu],
                    mask=None,title="CNN Muon Cut NuMu",
                    save=save,save_folder_name=save_folder)
        print("CNN Neutrino, True Muon Neutrino: %.2f +/- %.2f"%(percent_save[2],percent_error[2]))

    if sum(true_isNuE) > 0:
        percent_save,percent_error = my_confusion_matrix(true_isNu[MuNuE],
                    cnn_binary_mu[MuNuE], weights[MuNuE],
                    mask=None,title="CNN Muon Cut NuE",
                    save=save,save_folder_name=save_folder)
        print("CNN Neutrino, True Electron Neutrino: %.2f +/- %.2f"%(percent_save[2],percent_error[2]))

    if sum(true_isNuTau) > 0:
        percent_save,percent_error = my_confusion_matrix(true_isNu[MuNuTau],
                    cnn_binary_mu[MuNuTau], weights[MuNuTau],
                    mask=None,title="CNN Muon Cut NuTau",
                    save=save,save_folder_name=save_folder)
        print("CNN Neutrino, True Tau Neutrino: %.2f +/- %.2f"%(percent_save[2],percent_error[2]))

cnn_binary_pid = cnn_prob_track >= 0.7

plot_efficiency = True
if plot_efficiency:
    energy_array = np.arange(5, 101, 1)
    efficiency_mu_array = np.zeros(len(energy_array)-1)
    true_positive_mu_array = np.zeros(len(energy_array)-1)
    true_negative_mu_array = np.zeros(len(energy_array)-1)
    efficiency_pid_array = np.zeros(len(energy_array)-1)
    true_positive_pid_array = np.zeros(len(energy_array)-1)
    true_negative_pid_array = np.zeros(len(energy_array)-1)
    for energy_index in range(0,len(energy_array)-1):
        emin = energy_array[energy_index]
        emax = energy_array[energy_index+1]
        ecut = np.logical_and(cnn_energy > emin, cnn_energy < emax)

        weights_here = weights[ecut]
        true_positive_mu = np.logical_and(true_isNu[ecut], cnn_binary_mu[ecut])
        positive_mu = true_isNu[ecut]
        true_negative_mu = np.logical_and(true_isMuon[ecut], np.logical_not(cnn_binary_mu[ecut]))
        negative_mu = true_isMuon[ecut]
        true_positive_mu_array[energy_index] = sum(weights_here[true_positive_mu])/sum(weights_here[positive_mu])
        true_negative_mu_array[energy_index] = sum(weights_here[true_negative_mu])/sum(weights_here[negative_mu])
        total_mu = (sum(weights_here[true_positive_mu]) + sum(weights_here[true_negative_mu]))/sum(weights_here)
        efficiency_mu_array[energy_index] = total_mu
        
        all_cut = np.logical_and(ecut, true_isNu)
        weights_here = weights[all_cut]
        true_positive_pid = np.logical_and(true_isTrack[all_cut], cnn_binary_pid[all_cut])
        positive_pid = true_isTrack[all_cut]
        true_negative_pid = np.logical_and(np.logical_not(true_isTrack[all_cut]), np.logical_not(cnn_binary_pid[all_cut]))
        negative_pid = np.logical_not(true_isTrack[all_cut])
        true_positive_pid_array[energy_index] = sum(weights_here[true_positive_pid])/sum(weights_here[positive_pid])
        true_negative_pid_array[energy_index] = sum(weights_here[true_negative_pid])/sum(weights_here[negative_pid])
        total_pid = (sum(weights_here[true_positive_pid]) + sum(weights_here[true_negative_pid]))/sum(weights_here)
        efficiency_pid_array[energy_index] = total_pid

plt.figure(figsize=(10,10))
plt.title("Muon Efficiency",fontsize=25)
x = list(itertools.chain(*zip(energy_array[:-1],energy_array[1:])))
true_pos_mu_plot = list(itertools.chain(*zip(true_positive_mu_array,true_positive_mu_array)))
true_neg_mu_plot = list(itertools.chain(*zip(true_negative_mu_array,true_negative_mu_array)))
plt.plot(x,true_pos_mu_plot,'b-',linewidth=2,label="True Neutrino")
plt.plot(x,true_neg_mu_plot,'r-',linewidth=2,label="True Muon")
plt.xlabel("Reconstructed Energy (GeV)", fontsize=20)
plt.ylabel("Fraction Correctly Classified",fontsize=20)
plt.legend(fontsize=20)
plt.savefig("%s/EfficiencyMuon.png"%save_folder)

plt.figure(figsize=(10,10))
plt.title("Track Efficiency on Neutrinos",fontsize=25)
#plt.bar(energy_array[:-1],efficiency_array,width=1)
#plt.plot(energy_array[:-1],efficiency_pid_array,'b.-',markersize=10,linewidth=2)
plt.plot(energy_array[:-1],true_positive_pid_array,'b.-',markersize=10,linewidth=2,label="True Track")
plt.plot(energy_array[:-1],true_negative_pid_array,'r.-',markersize=10,linewidth=2,label="True Cascade")
plt.xlabel("Reconstructed Energy (GeV)", fontsize=20)
plt.ylabel("Fraction Correctly Classified",fontsize=20)
plt.legend(fontsize=20)
plt.savefig("%s/EfficiencyTrack.png"%save_folder)


#if save_output_data and i3:
#    f = h5py.File("%s/prediction_values_%inumu_%inue_%inutau_%imuon.hdf5"%(save_folder,numu_files,nue_files,nutau_files,muon_files), "w")
#    f.create_dataset("Y_predicted", data=predict)
#    f.create_dataset("Y_test_use", data=truth)
#    f.create_dataset("additional_info", data=info)
#    f.create_dataset("weights_test", data=raw_weights)
#    f.close()

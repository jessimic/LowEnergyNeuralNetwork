import h5py
import argparse
import os, sys
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import matplotlib.colors as colors

numu_files = 101.
nue_files = 101.
no_old_reco = False
#syst_set_names = ["140001", "140002", "140000", "140003", "140004"]
#syst_set_names = ["120001", "120002", "120000", "120003", "120004"]
#syst_set_names = ["140000", "140100", "140101", "140102", "140150", "141118"]
#syst_set_names = ["140000", "141118"]
syst_set_names = ["120000", "121118"]
#syst_set_names = ["120000", "120100", "120101", "120102", "120150"]
#syst_set_names = ["140000", "140500", "140501", "140502", "140503"]
basedir="/mnt/home/micall12/LowEnergyNeuralNetwork/output_plots/Test_Level6.5/"
#outdir = basedir + "SystStudies/14_BFROnly/"
outdir = basedir + "SystStudies/12_BFROnly/"
save_folder_name = outdir
if os.path.isdir(save_folder_name) != True:
    os.mkdir(save_folder_name)

true_energy_dict = {}
true_PID_dict = {}
true_coszenith_dict = {}
cnn_energy_dict = {}
cnn_PID_dict = {}
cnn_coszenith_dict = {}
reco_energy_dict = {}
reco_PID_dict = {}
reco_coszenith_dict = {}
treco_energy_dict = {}
treco_PID_dict = {}
treco_coszenith_dict = {}
weights_dict = {}
wreco_dict = {}

cut_index = 4

for index in range(0,len(syst_set_names)):
    
    keyname = syst_set_names[index]
    if keyname is "140000" or keyname is "120000":
        input_file = basedir + keyname + "/prediction_values_%s_100files.hdf5"%keyname
        numu_files = 99.
    else:
        input_file = basedir + keyname + "/prediction_values_%s.hdf5"%keyname
        numu_files = 101.


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
    
    cnn_energy = np.array(predict[:,0])
    try:
        cnn_class = np.array(predict[:,1])
    except:
        pass
    try:
        cnn_zenith = np.array(predict[:,2])
        cnn_coszenith = np.cos(cnn_zenith)
    except:
        pass

#Truth
    true_energy = np.array(truth[:,0])
    em_equiv_energy = np.array(truth[:,14])
    true_x = np.array(truth[:,4])
    true_y = np.array(truth[:,5])
    true_z = np.array(truth[:,6])
    true_CC = np.array(truth[:,11])
    true_isTrack = np.array(truth[:,8])
    x_origin = np.ones((len(true_x)))*46.290000915527344
    y_origin = np.ones((len(true_y)))*-34.880001068115234
    true_r = np.sqrt( (true_x - x_origin)**2 + (true_y - y_origin)**2 )
    true_neutrino = np.array(truth[:,9],dtype=int)
    true_zenith = np.array(truth[:,12])
    true_coszenith = np.cos(np.array(truth[:,12]))
    isNuMu = true_neutrino == 14
    isNuE = true_neutrino == 12
    if sum(isNuMu) == len(true_neutrino):
        all_NuMu = True
    else:
        all_NuMu = False

# Retro
    if reco is not None:
        retro_energy = np.array(reco[:,0])
        retro_zenith = np.array(reco[:,1])
        retro_time = np.array(reco[:,3])
        retro_PID_full = np.array(reco[:,12])
        retro_PID_up = np.array(reco[:,13])
        reco_x = np.array(reco[:,4])
        reco_y = np.array(reco[:,5])
        reco_z = np.array(reco[:,6])
        retro_coszenith = np.cos(retro_zenith)
        reco_r = np.sqrt( (reco_x - x_origin)**2 + (reco_y - y_origin)**2 )
        retro_nan = np.isnan(retro_energy)
        print(sum(retro_nan)/len(retro_nan))

#Additional info
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
        hlc_x = info[:,10]
        hlc_y = info[:,11]
        hlc_z = info[:,12]

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


    check_energy_gt5 = true_energy > 5.
    assert sum(check_energy_gt5)>0, "No events > 5 GeV in true energy, is this transformed?"

#Vertex Position
    x_origin = np.ones((len(true_x)))*46.290000915527344
    y_origin = np.ones((len(true_y)))*-34.880001068115234
    true_r = np.sqrt( (true_x - x_origin)**2 + (true_y - y_origin)**2 )
    if reco is not None:
        reco_r = np.sqrt( (reco_x - x_origin)**2 + (reco_y - y_origin)**2 )
    if info is not None:
        hlc_rho = np.sqrt( (hlc_x - x_origin)**2 + (hlc_y - y_origin)**2 )

    #True Masks
    maskNONE = true_energy > 0.
    assert sum(maskNONE)==len(true_energy), "Some true energy < 0? Check!" 
    maskCC = true_CC == 1
    maskZ = np.logical_and(true_z > -505, true_z < 192)
    maskR = true_r < 90.
    maskDC = np.logical_and(maskZ,maskR)
    maskE = np.logical_and(true_energy > 5., true_energy < 100.)
    maskE2 = np.logical_and(true_energy > 1., true_energy < 200.)

    #CNN Masks
    maskCNNE = np.logical_and(cnn_energy > 5., cnn_energy < 100.)
    maskCNNE2 = np.logical_and(cnn_energy > 1., cnn_energy < 100.)
    maskCNNZenith = cnn_coszenith <= 0.3

    #additional info Masks
    if info is not None:
        maskHits8 = hits8 == 1
        maskNu = prob_nu > 0.4
        maskNoise = noise_class > 0.95
        masknhit = nhit_doms > 2.5
        maskntop = n_top15 < 2.5
        masknouter = n_outer < 7.5
        maskHLCZ = np.logical_and(hlc_z > -500., hlc_z < -200.) 
        maskHLCR = hlc_rho < 300.
        maskHLCVertex = np.logical_and(maskHLCZ, maskHLCR)
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
        maskRetro = np.logical_and(fit_success, np.logical_and(np.logical_and(maskRetroZenith, maskRetroEnergy), maskRetroTime))
        maskANA = np.logical_and(np.logical_and(np.logical_and(maskRecoDC,  maskRetro), maskMC),maskHits8)
        assert sum(maskANA)!=len(maskANA), "No events after ANA mask"
        print(sum(weights[np.logical_and(maskANA,maskCC)]))

    cut_list = [np.logical_and(maskHits8,maskCC), maskCC, np.logical_and(maskHLCVertex, np.logical_and(np.logical_and(maskCNNE2, maskCC),maskHits8)), np.logical_and(maskHLCVertex, np.logical_and(np.logical_and(maskCNNE, maskCC),maskHits8)),np.logical_and(maskCNNZenith,np.logical_and(maskHLCVertex, np.logical_and(np.logical_and(maskCNNE, maskCC),maskHits8)))]

    mask = cut_list[cut_index]
    fit_success_cut = np.array(fit_success,dtype=bool)
    mask_retro = np.logical_and(mask,fit_success_cut)
    
    cnn_energy_dict[keyname] = cnn_energy[mask]
    cnn_PID_dict[keyname] = cnn_class[mask]
    cnn_coszenith_dict[keyname] = cnn_coszenith[mask]
    true_energy_dict[keyname] = true_energy[mask]
    true_PID_dict[keyname] = true_isTrack[mask]
    true_coszenith_dict[keyname] = true_coszenith[mask]
    reco_energy_dict[keyname] = retro_energy[mask_retro]
    reco_PID_dict[keyname] = retro_PID_full[mask_retro]
    reco_coszenith_dict[keyname] = retro_coszenith[mask_retro]
    treco_energy_dict[keyname] = true_energy[mask_retro]
    treco_PID_dict[keyname] = true_isTrack[mask_retro]
    treco_coszenith_dict[keyname] = true_coszenith[mask_retro]
    weights_dict[keyname] = weights[mask]
    wreco_dict[keyname] = weights[mask_retro]


print(true_energy_dict.keys())

save=True
if save ==True:
    print("Saving to %s"%save_folder_name)

cut_names = ["Weighted_Hits8CC", "WeightedCC", "WeightedCNN1100_HLCVertex_Hits8CC", "WeightedCNN5100_HLCVertex_Hits8CC", "WeightedCNN5100_ZenithHLCVertex_Hits8CC"]

from PlottingFunctions import plot_systematic_slices
from PlottingFunctions import plot_compare_resolution

save_base_name = save_folder_name
#for cut_index in [3, 4]:
    #cuts = cut_list[cut_index]
folder_name = cut_names[cut_index]

print("Working on %s"%folder_name)

save_base_name += "/%s/"%folder_name
if os.path.isdir(save_base_name) != True:
    os.mkdir(save_base_name)

#quick_res = (cnn_energy_dict["140000"] - true_energy_dict["140000"])/ true_energy_dict["140000"]
#high = quick_res > 2
#low = quick_res < -1.5
#print((sum(high)+sum(low))/len(quick_res))

save_folder_name = save_base_name + "/Energy/"
if os.path.isdir(save_folder_name) != True:
    os.mkdir(save_folder_name)

#Energy
plot_systematic_slices(true_energy_dict, cnn_energy_dict, syst_set_names,
                        weights_dict = weights_dict, use_fraction=True, \
                        use_old_reco = True, old_reco_dict=reco_energy_dict,
                        old_reco_truth_dict=treco_energy_dict,
                        old_reco_weights_dict=wreco_dict,
                        title="Systematic Set Energy Resolution",
                        save=save,savefolder=save_folder_name)

plot_systematic_slices(true_energy_dict, cnn_energy_dict, syst_set_names,
                        weights_dict = weights_dict, use_fraction=False, \
                        use_old_reco = True, old_reco_dict=reco_energy_dict,
                        old_reco_truth_dict=treco_energy_dict,
                        title="Systematic Set Energy Resolution",
                        old_reco_weights_dict=wreco_dict,
                        save=save,savefolder=save_folder_name)

plot_compare_resolution(true_energy_dict,cnn_energy_dict,syst_set_names,
                        weights_dict=weights_dict, savefolder=save_folder_name,\
                        save=save,bins=100,use_fraction=True, minval=-2,maxval=2)

plot_compare_resolution(treco_energy_dict,reco_energy_dict,syst_set_names,
                        weights_dict=wreco_dict, savefolder=save_folder_name,\
                        save=save,bins=100,use_fraction=True, minval=-2,maxval=2, reco_name="Retro")

save_folder_name = save_base_name + "/Zenith/"
if os.path.isdir(save_folder_name) != True:
    os.mkdir(save_folder_name)

#Zenith
plot_systematic_slices(true_coszenith_dict, cnn_coszenith_dict, syst_set_names,
                        weights_dict = weights_dict, use_fraction=True, \
                        use_old_reco = True, old_reco_dict=reco_coszenith_dict,
                        old_reco_truth_dict=treco_coszenith_dict,
                        old_reco_weights_dict=wreco_dict,
                        title="Systematic Set Cos Zenith Resolution",
                        save=save,savefolder=save_folder_name)

plot_systematic_slices(true_coszenith_dict, cnn_coszenith_dict, syst_set_names,
                        weights_dict = weights_dict, use_fraction=False, \
                        use_old_reco = True, old_reco_dict=reco_coszenith_dict,
                        old_reco_truth_dict=treco_coszenith_dict,
                        old_reco_weights_dict=wreco_dict,
                        title="Systematic Set Cos Zenith Resolution",
                        save=save,savefolder=save_folder_name)

plot_compare_resolution(true_coszenith_dict,cnn_coszenith_dict,syst_set_names,
                        weights_dict=weights_dict, savefolder=save_folder_name,variable="Cos Zenith",\
                        save=save,bins=100,use_fraction=True, minval=-2,maxval=2)

plot_compare_resolution(treco_coszenith_dict,reco_coszenith_dict,syst_set_names,
                        weights_dict=wreco_dict, savefolder=save_folder_name,variable="Cos Zenith",\
                        save=save,bins=100,use_fraction=True, minval=-2,maxval=2, reco_name="Retro")


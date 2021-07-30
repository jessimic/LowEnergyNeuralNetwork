import h5py
import argparse
import os, sys
import numpy as np
import dama as dm

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import matplotlib.colors as colors

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input",type=str,default=None,
                    dest="input_file", help="path and name of the input file")
parser.add_argument("-o", "--outdir",type=str,default='/mnt/home/micall12/LowEnergyNeuralNetwork/output_plots/',
                    dest="output_dir", help="path of ouput file")
parser.add_argument("--numu",type=float,default=1518.,
                    dest="numu", help="number of numu files")
parser.add_argument("--nue",type=float,default=602.,
                    dest="nue", help="number of nue files")
parser.add_argument("--no_old_reco", default=False,action='store_true',
                        dest='no_old_reco',help="no old reco")
parser.add_argument("--nu_type",type=str,default="NuMu",
                    dest="nu_type", help="NuMu or NuE")
args = parser.parse_args()

input_file = args.input_file
save_folder_name = args.output_dir
numu_files = args.numu
nue_files = args.nue
nu_type = args.nu_type

f = h5py.File(input_file, "r")
truth = f["Y_test_use"][:]
predict = f["Y_predicted"][:]
try:
    info = f["additional_info"][:]
except:
    info = None
if args.no_old_reco:
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


cnn_energy = np.array(predict[:,0])*100
cnn_error = np.array(predict[:,1])
#try:
#    cnn_class = np.array(predict[:,1])
#except:
#    pass
try:
    cnn_zenith = np.array(predict[:,2])
    cnn_coszenith = np.cos(cnn_zenith)
except:
    pass
try:
    cnn_x = np.array(predict[:,3])
    cnn_y = np.array(predict[:,4])
    cnn_z = np.array(predict[:,5])
except:
    cnn_x = None
    pass

#Truth
true_energy = np.array(truth[:,0])*100
true_error = abs(true_energy - cnn_energy)
em_equiv_energy = np.array(truth[:,14])
true_x = np.array(truth[:,4])
true_y = np.array(truth[:,5])
true_z = np.array(truth[:,6])
x_origin = np.ones((len(true_x)))*46.290000915527344
y_origin = np.ones((len(true_y)))*-34.880001068115234
true_CC = np.array(truth[:,11])
true_isTrack = np.array(truth[:,8])
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

if cnn_x is not None:
    cnn_r = np.sqrt( (cnn_x - x_origin)**2 + (cnn_y - y_origin)**2 )

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
else:
    weights = np.ones(len(true_energy))
    weights = np.array(weights,dtype=bool)

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

#Plot
from PlottingFunctions import plot_distributions
from PlottingFunctions import plot_2D_prediction
from PlottingFunctions import plot_2D_prediction_fraction
from PlottingFunctions import plot_single_resolution
from PlottingFunctions import plot_bin_slices
from PlottingFunctions import plot_rms_slices
from PlottingFunctionsClassification import ROC
from PlottingFunctionsClassification import plot_classification_hist

save=True
if save ==True:
    print("Saving to %s"%save_folder_name)


#True Masks
maskNONE = true_energy > 0.
assert sum(maskNONE)==len(true_energy), "Some true energy < 0? Check!" 
maskCC = true_CC == 1
maskZ = np.logical_and(true_z > -505, true_z < 192)
maskR90 = true_r < 90.
maskR150 = true_r < 150.
maskR300 = true_r < 300.
maskDC = np.logical_and(maskZ,maskR300)
maskE = np.logical_and(true_energy > 5., true_energy < 100.)
maskE2 = np.logical_and(true_energy > 1., true_energy < 200.)

#CNN Masks
maskCNNE = np.logical_and(cnn_energy > 5., cnn_energy < 100.)
maskCNNE2 = np.logical_and(cnn_energy > 1., cnn_energy < 100.)
#maskCNNZenith = cnn_coszenith <= 0.3
#maskCNNR90 = cnn_r < 90
#maskCNNR150 = cnn_r < 150
#maskCNNR300 = cnn_r < 300
#maskCNNZ = np.logical_and(cnn_z > -500, cnn_z < -200)
#maskCNNVertex = np.logical_and(maskCNNR150, maskCNNZ)

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

no_cuts = np.ones(len(true_energy))
no_cuts = np.array(no_cuts,dtype=bool)

cut_list = [
no_cuts,
#np.logical_and(maskHits8,maskCC), 
#np.logical_and( maskE,np.logical_and(maskHits8,maskCC)), 
#np.logical_and(maskCNNZ, np.logical_and(maskCNNR150 ,np.logical_and(maskCNNZenith, np.logical_and(np.logical_and(maskCNNE, maskCC),maskHits8)))),
#np.logical_and(maskCNNZ, np.logical_and(maskCNNR90 ,np.logical_and(maskCNNZenith, np.logical_and(np.logical_and(maskCNNE, maskCC),maskHits8)))),
#maskHits8,
#np.logical_and(maskCNNZ, np.logical_and(maskCNNR150 ,np.logical_and(maskCNNZenith, np.logical_and(maskCNNE,maskHits8))))
]

cut_names = [
"NoCuts",
#"Weighted_Hits8CC",
#"WeightedTrueE5100_Hits8CC", 
#"WeightedReco5100_ZenithZR150_Hits8CC", 
#"WeightedReco5100_ZenithZR90_Hits8CC", 
#"Weighted_Hits8",
#"WeightedReco5100_ZenithZR150_Hits8"
]

minvals_energy = [1, 5, 5,5,1,5]
maxvals_energy = [200, 100, 100, 100., 200, 100]
binss_energy = [199, 95, 95, 95, 199,95]
syst_bins_energy = [20, 10, 10, 10, 20,10]
minvals_zenith =  [-1, -1,-1,-1,-1,-1,-1]
maxvals_zenith = [1,1,1,1, 0.3, 1, 1,1,0.3,0.3,1,0.3]
binss_zenith = [100, 100, 100, 100, 100, 100, 100, 100,100,100,100,100]
syst_bins_zenith = [20, 20, 20, 20, 12, 20, 20, 20,12,12,20,12]
minvals_error = [0, 0, 0, 0, 0, 0, 0,0,0,0,0,0]
maxvals_error = [100,100,100,100,100,100,100]
binss_error = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
syst_bins_error = [20, 20, 20, 20, 20, 20, 20, 20,20,20,20,20]
sample_names = ["CC", "CC", "CC", "CC","CC&NC","CC&NC"]
plot_main = True
plot_EMequiv = False
plot_switch = False
plot_others = False
plot_vertex = False
plot_PID = False
save_base_name = save_folder_name


for cut_index in [0]: #0,6,8,9 range(1,len(cut_list)):
    cuts = cut_list[cut_index]
    folder_name = cut_names[cut_index]
    reco_name = "Retro"
    sample_name = sample_names[cut_index]
    print("Number of events. CNN: %i"%(sum(cuts)))
    
    print("Working on %s"%folder_name)

    save_cut_name = save_base_name + "/%s/"%folder_name
    if os.path.isdir(save_cut_name) != True:
        os.mkdir(save_cut_name)

    true_energy_val = true_energy[cuts]
    true_weights = weights[cuts]
    minval_energy = minvals_energy[cut_index]
    maxval_energy = maxvals_energy[cut_index]
    true_R = true_r[cuts]
    true_Z = true_z[cuts]
    #cnn_R = cnn_r[cuts]
    #cnn_Z = cnn_z[cuts]

    for variable in ["error"]:

        save_folder_name = save_cut_name + "/%s/"%variable
        if os.path.isdir(save_folder_name) != True:
            os.mkdir(save_folder_name)

        retro_val = None
        retro_true_val = None
        retro_true_R = None
        retro_true_Z = None
        retro_weights = None
        if variable == "energy":
            plot_name = "Energy"
            plot_units = "(GeV)"
            minval = minvals_energy[cut_index]
            maxval = maxvals_energy[cut_index]
            syst_bin = syst_bins_energy[cut_index]
            bins = binss_energy[cut_index]
            cnn_val = np.copy(cnn_energy[cuts])
            true_val = np.copy(true_energy[cuts])
        elif variable == "zenith":
            plot_name = "Cosine Zenith"
            plot_units = ""
            minval = minvals_zenith[cut_index]
            maxval = maxvals_zenith[cut_index]
            syst_bin = syst_bins_zenith[cut_index]
            bins = binss_zenith[cut_index]
            cnn_val = np.copy(cnn_coszenith[cuts])
            true_val = np.copy(true_coszenith[cuts])
        elif variable == "error":
            plot_name = "Absolute Error"
            plot_units = "(GeV)"
            minval = minvals_error[cut_index]
            maxval = maxvals_error[cut_index]
            syst_bin = syst_bins_error[cut_index]
            bins = binss_error[cut_index]
            cnn_val = np.copy(cnn_error[cuts])
            true_val = np.copy(true_error[cuts])
        else:
            print("ONLY work with 3 variables (energy, zenith, error) currently")
    
        #print("With cuts, CNN E range is %f - %f, CNN Cos Zen range is %f - %f"%(min(cnn_energy[cuts]),max(cnn_energy[cuts]),min(cnn_coszenith[cuts]),max(cnn_coszenith[cuts])))

        #print(plot_name, plot_units, minval, maxval, syst_bin, bins)
        #print(max(true_val), max(cnn_val), max(retro_val))
        #print(true_val[:10], cnn_val[:10], retro_val[:10])
        if nu_type == "NuMu" or nu_type == "numu":
            dist_title = r'$\nu_\mu$ '
        elif nu_type == "NuE" or nu_type == "nue":
            dist_title = r'for $\nu_e$ '
        else:
            dist_title += nu_type
        dist_title += sample_name
    

        if plot_main:
            plot_distributions(true_val, cnn_val, 
                                old_reco=retro_val,old_reco_weights=retro_weights,\
                                save=save, savefolder=save_folder_name, weights=true_weights,\
                                reco_name = reco_name, variable=plot_name, units= plot_units,
                                minval=minval,maxval=maxval,bins=bins)
            plot_distributions(true_val, cnn_val,
                                old_reco=retro_val,old_reco_weights=retro_weights,\
                                save=save, savefolder=save_folder_name, weights=true_weights,\
                                reco_name = reco_name, variable=plot_name, units= plot_units,
                                minval=min(true_val), maxval=max(true_val))
        if plot_others:
            plot_distributions(true_r[cuts], reco_r[cuts],\
                                save=save, savefolder=save_folder_name, weights=true_weights,\
                                cnn_name = reco_name,
                                variable="Radial Vertex", units= "(m)",log=True)
            plot_distributions(true_z[cuts], reco_z[cuts],\
                                save=save, savefolder=save_folder_name,
                                weights=true_weights,cnn_name = reco_name,
                                variable="Z Vertex", units= "(m)",log=True)
        if plot_main:
            switch = False 
            plot_2D_prediction(true_val, cnn_val,weights=true_weights,\
                                save=save, savefolder=save_folder_name,
                                bins=bins, switch_axis=switch,
                                variable=plot_name, units=plot_units,
                                reco_name="CNN",flavor=nu_type,sample=sample_name)
            plot_2D_prediction(true_val, cnn_val,weights=true_weights,\
                                save=save, savefolder=save_folder_name,
                                bins=bins,switch_axis=switch,\
                                minval=minval, maxval=maxval, 
                                cut_truth=True, axis_square=True,\
                                variable=plot_name, units=plot_units,\
                                reco_name="CNN",flavor=nu_type,sample=sample_name)
            if retro_val is not None:
                plot_2D_prediction(retro_true_val, retro_val, weights=retro_weights,
                                save=save, savefolder=save_folder_name,
                                bins=bins,switch_axis=switch,\
                                variable=plot_name, units=plot_units,
                                reco_name=reco_name,flavor=nu_type,sample=sample_name)
                plot_2D_prediction(retro_true_val, retro_val, weights=retro_weights,
                                save=save, savefolder=save_folder_name,
                                bins=bins,switch_axis=switch,\
                                minval=minval, maxval=maxval,
                                cut_truth=True, axis_square=True,\
                                variable=plot_name, units=plot_units,
                                reco_name=reco_name,flavor=nu_type,sample=sample_name)
        if plot_EMequiv:
            plot_2D_prediction(em_equiv_energy[cuts], cnn_val,weights=true_weights,\
                                save=save, savefolder=save_folder_name,
                                bins=bins,switch_axis=switch,\
                                minval=minval, maxval=maxval,
                                cut_truth=True, axis_square=True,\
                                variable=plot_name, units=plot_units,
                                reco_name="CNN",variable_type="EM Equiv",
                                flavor=nu_type,sample=sample_name)
            if retro_val is not None:
                plot_2D_prediction(em_equiv_energy[cuts], retro_val, weights=true_weights,
                                save=save, savefolder=save_folder_name,
                                bins=bins,switch_axis=switch,\
                                minval=minval, maxval=maxval,
                                cut_truth=True, axis_square=True,\
                                variable=plot_name, units=plot_units, 
                                reco_name=reco_name, variable_type = "EM Equiv",
                                flavor=nu_type,sample=sample_name)
        if plot_switch:
            switch = True
            plot_2D_prediction(true_val, cnn_val,weights=true_weights,\
                                save=save, savefolder=save_folder_name,
                                bins=bins, switch_axis=switch,
                                variable=plot_name, units=plot_units,
                                reco_name="CNN",flavor=nu_type,sample=sample_name)
            plot_2D_prediction(true_val, cnn_val,weights=true_weights,\
                                save=save, savefolder=save_folder_name,
                                bins=bins,switch_axis=switch,\
                                minval=minval, maxval=maxval,
                                cut_truth=True, axis_square=True,\
                                variable=plot_name, units=plot_units,
                                reco_name="CNN",flavor=nu_type,sample=sample_name)
            if retro_val is not None:
                plot_2D_prediction(retro_true_val, retro_val, weights=retro_weights,
                                save=save, savefolder=save_folder_name,
                                bins=bins,switch_axis=switch,\
                                variable=plot_name, units=plot_units,
                                reco_name=reco_name,flavor=nu_type,sample=sample_name)
                plot_2D_prediction(retro_true_val, retro_val, weights=retro_weights,
                                save=save, savefolder=save_folder_name,
                                bins=bins,switch_axis=switch,\
                                minval=minval, maxval=maxval,
                                cut_truth=True, axis_square=True,\
                                variable=plot_name, units=plot_units,
                                reco_name=reco_name,flavor=nu_type,sample=sample_name)
        if plot_main:   
            if retro_val is not None:
                plot_single_resolution(true_val, cnn_val, weights=true_weights,\
                           use_old_reco = True, old_reco = retro_val,\
                           old_reco_truth=retro_true_val,
                            old_reco_weights = retro_weights,old_reco_name=reco_name,
                           minaxis=-maxval, maxaxis=maxval, bins=bins,\
                           save=save, savefolder=save_folder_name,\
                           variable=plot_name, units=plot_units,
                           flavor=nu_type,sample=sample_name)
                
            else:
                plot_single_resolution(true_val, cnn_val, weights=true_weights,\
                           use_old_reco = False,\
                           minaxis=-maxval, maxaxis=maxval, bins=bins,\
                           save=save, savefolder=save_folder_name,\
                           variable=plot_name, units=plot_units,
                           flavor=nu_type,sample=sample_name)
        
        
        if plot_vertex:
                plot_bin_slices(true_val, cnn_val, energy_truth=true_R,
                            weights=true_weights,  
                            old_reco = retro_val,old_reco_truth=retro_true_val,
                            reco_energy_truth=retro_true_R,
                            old_reco_weights=retro_weights,use_fraction = True,
                            bins=syst_bin, min_val=0, max_val=300,\
                            save=save, savefolder=save_folder_name,\
                            variable=plot_name, units="(m)",
                            reco_name=reco_name, xvariable="True Radius",
                            flavor=nu_type,sample=sample_name)
                plot_bin_slices(true_val, cnn_val, energy_truth=true_Z,
                            weights=true_weights,  
                            old_reco = retro_val,old_reco_truth=retro_true_val,
                            reco_energy_truth=retro_true_Z,
                            old_reco_weights=retro_weights,use_fraction = True,
                            bins=syst_bin, min_val=-550, max_val=-150,\
                            save=save, savefolder=save_folder_name,\
                            variable=plot_name, units="(m)",
                            reco_name=reco_name, xvariable="True Z",
                            flavor=nu_type,sample=sample_name)
                plot_bin_slices(true_val, cnn_val, energy_truth=cnn_R,
                            weights=true_weights,  
                            old_reco = retro_val,old_reco_truth=retro_true_val,
                            reco_energy_truth=retro_R,
                            old_reco_weights=retro_weights,use_fraction = True,
                            bins=syst_bin, min_val=0, max_val=300,\
                            save=save, savefolder=save_folder_name,\
                            variable=plot_name, units="(m)",
                            reco_name=reco_name, xvariable="Reconstructed Radius",
                            flavor=nu_type,sample=sample_name)
                plot_bin_slices(true_val, cnn_val, energy_truth=cnn_Z,
                            weights=true_weights,  
                            old_reco = retro_val,old_reco_truth=retro_true_val,
                            reco_energy_truth=retro_Z,
                            old_reco_weights=retro_weights,use_fraction = True,
                            bins=syst_bin, min_val=-550, max_val=-150,\
                            save=save, savefolder=save_folder_name,\
                            variable=plot_name, units="(m)",
                            reco_name=reco_name, xvariable="Reconstructed Z",
                            flavor=nu_type,sample=sample_name)

        
        if variable == "energy":
            if plot_others:
                plot_2D_prediction_fraction(true_val, cnn_val,weights=true_weights,\
                                save=save, savefolder=save_folder_name,bins=bins,\
                                xminval=minval, xmaxval=maxval, yminval=-2,ymaxval=2.,\
                                variable=plot_name, units=plot_units, reco_name="CNN")
                plot_2D_prediction_fraction(retro_true_val, retro_val, weights=retro_weights,
                                save=save, savefolder=save_folder_name,bins=bins,\
                                xminval=minval, xmaxval=maxval, yminval=-2,ymaxval=2.,\
                                variable=plot_name, units=plot_units, reco_name=reco_name)
                plot_2D_prediction_fraction(true_val, cnn_val,weights=true_weights,\
                                save=save, savefolder=save_folder_name,bins=bins,\
                                variable=plot_name, units=plot_units, reco_name="CNN")
                plot_2D_prediction_fraction(retro_true_val, retro_val, weights=retro_weights,
                                save=save, savefolder=save_folder_name,bins=bins,\
                                variable=plot_name, units=plot_units, reco_name=reco_name)
            
            if plot_main:           
                if nu_type == "NuMu" or nu_type == "numu":
                    dist_title = r'$\nu_\mu$ '
                elif nu_type == "NuE" or nu_type == "nue":
                    dist_title = r'for $\nu_e$ '
                else:
                    dist_title += nu_type
                dist_title += sample_name
                #tretro_energy_val, retro_weights
                plot_distributions(true_val,log=True, 
                                    save=save, savefolder=save_folder_name, 
                                    weights=true_weights,reco_name = reco_name,
                                    variable=plot_name, units= plot_units,
                                    bins=bins,
                                    title="Testing Energy Distribution for %s"%dist_title)
                #true_val, true_weights
                plot_distributions(true_val,log=True,minval=1,maxval=300, 
                                    save=save, savefolder=save_folder_name, 
                                    weights=true_weights,reco_name = reco_name,
                                    variable=plot_name, units= plot_units,
                                    bins=bins,
                                    title="Testing Energy Distribution for %s"%dist_title)

                if retro_val is not None:
                    plot_single_resolution(true_val, cnn_val, weights=true_weights,\
                           use_old_reco = True, old_reco = retro_val,\
                           old_reco_truth=retro_true_val, old_reco_weights=retro_weights,
                           minaxis=-2, maxaxis=2, bins=bins,old_reco_name=reco_name,\
                           save=save, savefolder=save_folder_name,use_fraction=True,\
                           variable=plot_name, units=plot_units,
                           flavor=nu_type,sample=sample_name)
                else:
                    plot_single_resolution(true_val, cnn_val, weights=true_weights,\
                           use_old_reco = False, minaxis=-2, maxaxis=2, bins=bins,\
                           save=save, savefolder=save_folder_name,use_fraction=True,\
                           variable=plot_name, units=plot_units,
                           flavor=nu_type,sample=sample_name)

            
                plot_bin_slices(true_val, cnn_val, weights=true_weights,  
                            old_reco = retro_val,old_reco_truth=retro_true_val,
                            old_reco_weights=retro_weights,use_fraction = True,
                            bins=syst_bin, min_val=minval, max_val=maxval,\
                            save=save, savefolder=save_folder_name,\
                            variable=plot_name, units=plot_units, reco_name=reco_name,
                            flavor=nu_type,sample=sample_name)
                plot_rms_slices(true_val, cnn_val, weights=true_weights, 
                            old_reco_truth=retro_true_val, 
                            old_reco = retro_val, old_reco_weights = retro_weights,\
                            use_fraction = True, bins=syst_bin,
                            min_val=minval, max_val=maxval,\
                            save=save, savefolder=save_folder_name,\
                            variable=plot_name, units=plot_units, reco_name=reco_name,
                            flavor=nu_type,sample=sample_name)
                plot_bin_slices(true_val, cnn_val, weights=None,  
                            old_reco = retro_val,old_reco_truth=retro_true_val,\
                            use_fraction = True, bins=syst_bin, 
                            min_val=minval, max_val=maxval,\
                            save=save, savefolder=save_folder_name,\
                            variable=plot_name, units=plot_units, reco_name=reco_name,
                            flavor=nu_type,sample=sample_name)
                plot_bin_slices(true_val, cnn_val, weights=true_weights,  
                            old_reco = retro_val,old_reco_truth=retro_true_val,
                            old_reco_weights=retro_weights, vs_predict = True,\
                            use_fraction = True, bins=syst_bin,
                            min_val=minval, max_val=maxval,\
                            save=save, savefolder=save_folder_name,\
                            variable=plot_name, units=plot_units, reco_name=reco_name,
                            flavor=nu_type,sample=sample_name)

            
            

        
        if variable == "zenith":
            if retro_val is not None:
                plot_single_resolution(true_val, cnn_val, weights=true_weights,\
                           use_old_reco = True, old_reco_truth=retro_true_val,
                           old_reco = retro_val,\
                           old_reco_weights = retro_weights,
                           minaxis=-2, maxaxis=2, bins=bins,\
                           save=save, savefolder=save_folder_name,\
                           variable=plot_name, units=plot_units, old_reco_name=reco_name)
            else:
                plot_single_resolution(true_val, cnn_val, weights=true_weights,\
                           use_old_reco = False,
                           minaxis=-2, maxaxis=2, bins=bins,\
                           save=save, savefolder=save_folder_name,\
                           variable=plot_name, units=plot_units)
            plot_bin_slices(true_val, cnn_val, weights=true_weights,  
                            old_reco_truth=retro_true_val,
                            old_reco = retro_val,old_reco_weights=retro_weights,\
                            use_fraction = False, bins=syst_bin, min_val=minval, max_val=maxval,\
                            save=save, savefolder=save_folder_name,\
                            variable=plot_name, units=plot_units, reco_name=reco_name)
            plot_distributions(true_val,
                                    save=save, savefolder=save_folder_name, weights=true_weights,\
                                    reco_name = reco_name, variable=plot_name, units= plot_units,
                                    bins=bins)
            plot_distributions(true_val,minval=-1,maxval=1., 
                                    save=save, savefolder=save_folder_name, weights=true_weights,\
                                    reco_name = reco_name, variable=plot_name, units= plot_units,
                                    bins=bins)
            """
            plot_bin_slices(true_val, cnn_val, weights=true_weights,  
                            old_reco = retro_val,vs_predict = True,\
                            use_fraction = False, bins=syst_bin, min_val=minval, max_val=maxval,\
                            save=save, savefolder=save_folder_name,\
                            variable=plot_name, units=plot_units, reco_name=reco_name)
            """
            plot_bin_slices(true_val, cnn_val, weights=true_weights, \
                            old_reco = retro_val,old_reco_truth=retro_true_val,
                            energy_truth= true_energy_val,\
                            reco_energy_truth=tretro_energy_val,
                            old_reco_weights=retro_weights,
                            use_fraction = False, bins=syst_bin, \
                            min_val=minval_energy, max_val=maxval_energy,\
                            save=save, savefolder=save_folder_name,\
                            variable="True Neutrino Energy", units=plot_units, reco_name=reco_name)
            plot_rms_slices(true_val, cnn_val, weights=true_weights,  
                            old_reco = retro_val, old_reco_truth=retro_true_val,
                            energy_truth= true_energy_val,\
                            reco_energy_truth=tretro_energy_val,
                            old_reco_weights=retro_weights,
                            use_fraction = False, bins=syst_bin,
                            min_val=minval_energy, max_val=maxval_energy,\
                            save=save, savefolder=save_folder_name,\
                            variable="True Neutrino Energy", units=plot_units, reco_name=reco_name)


    if plot_PID:
        #reco=reco_class
        ROC(true_isTrack,cnn_class,mask=cuts,mask_name="",save=save,save_folder_name=save_folder_name)
        plot_classification_hist(true_isTrack,cnn_class,mask=cuts,mask_name="", variable="Probability Track",units="",bins=50,log=False,save=save,save_folder_name=save_folder_name)


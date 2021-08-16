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

plot_containment = False
plot_resolution = False
plot_energy =False
plot_main = False
print_rates = False
plot_rates = False
plot_after_threshold = False
plot_2d_vertex = False
plot_1dhists = True
plot_2d_hists = False
plot_rcut_muonrate = False
do_energy_cut_plots = False
check_nhits = False
plot_cut_effect = False

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
cnn_prob_mu = np.array(predict[:,-1])
cnn_zenith = np.array(predict[:,1])
cnn_prob_track = np.array(predict[:,2])
cnn_x = np.array(predict[:,3])
cnn_y = np.array(predict[:,4])
cnn_z = np.array(predict[:,5])
cnn_coszen = np.cos(cnn_zenith)
cnn_energy = np.array(predict[:,0])

true_energy = np.array(truth[:,0])*efactor
true_zenith = np.array(truth[:,1])
true_azimuth = np.array(truth[:,2])
true_coszenith = np.cos(true_zenith)
true_x = np.array(truth[:,4])
true_y = np.array(truth[:,5])
true_z = np.array(truth[:,6])
true_CC = np.array(truth[:,11])
true_isCC = true_CC == 1
true_track = np.array(truth[:,8])
true_isTrack = np.array(true_track,dtype=bool)
true_track_length = np.array(truth[:,7])
nx = np.sin(true_zenith)*np.cos(true_azimuth)
ny = np.sin(true_zenith)*np.sin(true_azimuth)
nz = np.cos(true_zenith)
true_xend = true_x + true_track_length*nx
true_yend = true_y + true_track_length*ny
true_zend = true_z + true_track_length*nz

noise_class = info[:,4]
nhit_doms = info[:,5]
coin_muon = info[:,0]
ntop = info[:,6]
nouter = info[:,7]

x_origin = np.ones((len(true_x)))*46.290000915527344
y_origin = np.ones((len(true_y)))*-34.880001068115234
true_r = np.sqrt( (true_x - x_origin)**2 + (true_y - y_origin)**2 )
cnn_r = np.sqrt( (cnn_x - x_origin)**2 + (cnn_y - y_origin)**2 )
true_rend = np.sqrt( (true_xend - x_origin)**2 + (true_yend - y_origin)**2 )

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
r_cut_value = 165
z_min_value = -495
z_max_value = -225
no_cut = true_energy > 0
upgoing = cnn_coszen < 0.3
e_min_cut = 5
zenith_cut = upgoing
r_cut = cnn_r < r_cut_value
z_cut = np.logical_and(cnn_z > z_min_value, cnn_z < z_max_value)
vertex_cut = np.logical_and(r_cut, z_cut)
CC_cut = true_isCC 
cnn_mu_cut = .45
cnn_nu = cnn_prob_mu < cnn_mu_cut
cnn_prob_nu = np.ones(len(cnn_prob_mu)) - cnn_prob_mu
e_large_cut = np.logical_and(cnn_energy > e_min_cut, cnn_energy < 200)
e_small_cut = np.logical_and(cnn_energy > e_min_cut, cnn_energy < 200)

noise_cut = noise_class > 0.95
nhits_cut = nhit_doms >= 3
ntop_cut = ntop < 3
nouter_cut = nouter < 8

from PlottingFunctionsClassification import ROC
from PlottingFunctionsClassification import plot_classification_hist
from PlottingFunctionsClassification import confusion_matrix
from PlottingFunctions import plot_2D_prediction
from PlottingFunctions import plot_bin_slices
from PlottingFunctions import plot_distributions

mask_here = upgoing & noise_cut
mask_name_here="QuickTest"
if no_weights:
    mask_name_here += "_NoWeights"

save_folder += "/%s/"%mask_name_here
print("Working on %s"%save_folder)
if os.path.isdir(save_folder) != True:
    os.mkdir(save_folder)

if plot_containment:
    from plot_1d_slices import plot_1d_binned_slices
    a_mask = true_isNuMu & true_isCC
    plot_1d_binned_slices(true_energy[a_mask], cnn_energy[a_mask],
                    reco1_weight=weights[a_mask],xarray1=true_rend[a_mask],
                    plot_resolution = True,use_fraction = True,
                    xmin=0, xmax=800,bins=50,xline=260,xline_name="IC19",
                    save=save, savefolder=save_folder,
                    x_name="R End", x_units="(m)",reco1_name="CNN")
    plot_1d_binned_slices(true_energy[a_mask], cnn_energy[a_mask],
                    reco1_weight=weights[a_mask],xarray1=true_zend[a_mask],
                    plot_resolution = True,use_fraction = True,
                    xmin=-800, xmax=600,bins=50,
                    xline=[-500,500],xline_name="IceCube Volume",
                    save=save, savefolder=save_folder,
                    x_name="Z End", x_units="(m)",reco1_name="CNN")
    resolution = (cnn_energy - true_energy)/ true_energy
    minval = 1
    maxval = 300
    bins = 100

    xvar="True Energy (GeV)"
    yvar="Energy Resolution (R-T)/T"
    save_title="Resolution Vs True Energy"
    plt.figure(figsize=(10,10))
    plt.title("%s"%save_title,fontsize=25)
    plt.hist2d(true_energy[a_mask],resolution[a_mask],weights=weights[a_mask],bins=bins,cmap='viridis_r',range=[[1,500],[-1,1]],cmin=1e-12)
    cbar = plt.colorbar()
    plt.ylabel("%s"%yvar,fontsize=20)
    plt.xlabel("%s"%xvar,fontsize=20)
    plt.savefig("%s%s2DHist.png"%(save_folder,save_title.replace(" ","")),bbox_inches='tight')
    plt.close()

    xvar="CNN Energy (GeV)"
    yvar="Energy Resolution (R-T)/T"
    save_title="Resolution Vs CNN Energy"
    plt.figure(figsize=(10,10))
    plt.title("%s"%save_title,fontsize=25)
    plt.hist2d(cnn_energy[a_mask],resolution[a_mask],weights=weights[a_mask],bins=bins,cmap='viridis_r',range=[[1,300],[-1,1]],cmin=1e-12) #norm=colors.LogNorm())
    cbar = plt.colorbar()
    plt.ylabel("%s"%yvar,fontsize=20)
    plt.xlabel("%s"%xvar,fontsize=20)
    plt.savefig("%s%s2DHist.png"%(save_folder,save_title.replace(" ","")),bbox_inches='tight')
    plt.close()

    plot_2D_prediction(true_energy[a_mask], cnn_energy[a_mask],
                        weights=weights[a_mask],
                        save=save, savefolder=save_folder,
                        bins=bins,minval=1,maxval=300,
                        variable="Energy", units="(GeV)",
                        reco_name="CNN",variable_type="True",
                        flavor="NuMu",axis_square=True,
                        sample="CC",
                        no_contours=False)

    rinside_cut = true_rend < 260
    zinside_cut = np.logical_and(true_zend > -500, true_zend < 500)
    r_start = true_r < 260
    a_mask = true_isNuMu & true_isCC & rinside_cut & zinside_cut & z_cut & r_start

    xvar="True Energy (GeV)"
    yvar="Energy Resolution (R-T)/T"
    save_title="Fully Contained Resolution Vs True Energy"
    plt.figure(figsize=(10,10))
    plt.title("%s"%save_title,fontsize=25)
    plt.hist2d(true_energy[a_mask],resolution[a_mask],weights=weights[a_mask],bins=bins,cmap='viridis_r',range=[[1,500],[-1,1]],cmin=1e-12) #,cmin=1, norm=colors.LogNorm())
    cbar = plt.colorbar()
    plt.ylabel("%s"%yvar,fontsize=20)
    plt.xlabel("%s"%xvar,fontsize=20)
    plt.savefig("%s%s2DHist.png"%(save_folder,save_title.replace(" ","")),bbox_inches='tight')
    plt.close()

    plot_1d_binned_slices(true_energy[a_mask], cnn_energy[a_mask],
                    reco1_weight=weights[a_mask],xarray1=true_rend[a_mask],
                    plot_resolution = True,use_fraction = True,
                    xmin=0, xmax=260,bins=20,
                    save=save, savefolder=save_folder,
                    x_name="Fully Contained R", x_units="(m)",reco1_name="CNN")
    plot_2D_prediction(true_energy[a_mask], cnn_energy[a_mask],
                        weights=weights[a_mask],
                        save=save, savefolder=save_folder,
                        bins=bins,minval=1,maxval=300,
                        variable="Energy", units="(GeV)",
                        reco_name="CNN",variable_type="True",
                        flavor="NuMu",save_name="Fully Contained",
                        sample="CC",axis_square=True,
                        no_contours=False)
    plot_1d_binned_slices(true_energy[a_mask], cnn_energy[a_mask],
                    reco1_weight=weights[a_mask],xarray1=true_zend[a_mask],
                    plot_resolution = True,use_fraction = True,
                    xmin=-500, xmax=300,bins=30,
                    save=save, savefolder=save_folder,
                    x_name="Fully Contained Z", x_units="(m)",reco1_name="CNN")

    routside_cut = true_rend > 260
    zoutside_cut = np.logical_or(true_zend < -500, true_zend > 500)
    outside_cut = np.logical_or(routside_cut, zoutside_cut)
    a_mask = true_isNuMu & true_isCC & outside_cut & z_cut & r_start
    xvar="True Energy (GeV)"
    yvar="Energy Resolution (R-T)/T"
    save_title="End Outside Resolution Vs True Energy"
    plt.figure(figsize=(10,10))
    plt.title("%s"%save_title,fontsize=25)
    plt.hist2d(true_energy[a_mask],resolution[a_mask],weights=weights[a_mask],bins=bins,cmap='viridis_r',range=[[1,500],[-1,1]],cmin=1e-12) #, norm=colors.LogNorm())
    cbar = plt.colorbar()
    plt.ylabel("%s"%yvar,fontsize=20)
    plt.xlabel("%s"%xvar,fontsize=20)
    plt.savefig("%s%s2DHist.png"%(save_folder,save_title.replace(" ","")),bbox_inches='tight')
    plt.close()
    plot_1d_binned_slices(true_energy[a_mask], cnn_energy[a_mask],
                    reco1_weight=weights[a_mask],xarray1=true_rend[a_mask],
                    plot_resolution = True,use_fraction = True,
                    xmin=0, xmax=800,bins=50,xline=260,xline_name="IC19",
                    save=save, savefolder=save_folder,
                    x_name="End Outside R", x_units="(m)",reco1_name="CNN")
    plot_2D_prediction(true_energy[a_mask], cnn_energy[a_mask],
                        weights=weights[a_mask],
                        save=save, savefolder=save_folder,
                        bins=bins,minval=1,maxval=300,
                        variable="Energy", units="(GeV)",
                        reco_name="CNN",variable_type="True",
                        flavor="NuMu",save_name="End Outside",
                        sample="CC",axis_square=True,
                        no_contours=False)
    plot_1d_binned_slices(true_energy[a_mask], cnn_energy[a_mask],
                    reco1_weight=weights[a_mask],xarray1=true_zend[a_mask],
                    plot_resolution = True,use_fraction = True,
                    xmin=-800, xmax=600,bins=50,xline=[-500,500],xline_name="IceCube Volume",
                    save=save, savefolder=save_folder,
                    x_name="End Outside Z", x_units="(m)",reco1_name="CNN")

if plot_resolution:

    const_cut = cnn_nu & nhits_cut & noise_cut & CC_cut
    flavor_mask = [true_isNu, true_isNuMu, true_isNuE, true_isNuTau]
    flavor_names = ["Nu", "NuMu", "NuE", "NuTau"]
    
    #Energy
    this_var_mask = const_cut & z_cut & r_cut & zenith_cut
    minval = 1
    maxval = 300
    bins = 100
    syst_bin = 20
    cut_line = [100, 200]
    variable_name = "Energy"
    units = "(GeV)"
    for i in range(0,4):
        a_mask = this_var_mask & flavor_mask[i]
        flavor_name = flavor_names[i]
        plot_2D_prediction(true_energy[a_mask], cnn_energy[a_mask],
                            weights=weights[a_mask],
                            save=save, savefolder=save_folder,
                            bins=bins,minval=minval,maxval=maxval,
                            switch_axis = True,
                            variable=variable_name, units=units,
                            reco_name="CNN",variable_type="True",
                            flavor=flavor_name,
                            sample="CC",xline=cut_line,
                            no_contours=False)
        plot_2D_prediction(true_energy[a_mask], cnn_energy[a_mask],
                            weights=weights[a_mask],
                            save=save, savefolder=save_folder,
                            bins=bins,minval=minval,maxval=maxval,
                            switch_axis = True,axis_square=True,
                            variable=variable_name, units=units,
                            reco_name="CNN",variable_type="True",
                            flavor=flavor_name,
                            sample="CC",xline=cut_line,
                            no_contours=False)
        plot_2D_prediction(true_energy[a_mask], cnn_energy[a_mask],
                            weights=weights[a_mask],
                            save=save, savefolder=save_folder,
                            bins=bins,minval=minval,maxval=maxval,
                            switch_axis = False,axis_square=True,
                            variable=variable_name, units=units,
                            reco_name="CNN",variable_type="True",
                            flavor=flavor_name,
                            sample="CC",xline=cut_line,
                            no_contours=False)
        plot_bin_slices(true_energy[a_mask], cnn_energy[a_mask],
                        weights=weights[a_mask],
                        vs_predict = True,use_fraction = True,
                        specific_bins=[1,3,10,15,20,25,30,35,40,50,60,70,80,100,120,140,160,200,250],
                        min_val=minval, max_val=220,
                        save=save, savefolder=save_folder,\
                        variable=variable_name, units=units,
                        flavor=flavor_name,sample="CC")
        plot_distributions(true_energy[a_mask], cnn_energy[a_mask],
                          weights=weights[a_mask],\
                          save=save, savefolder=save_folder, 
                          reco_name = "CNN", variable=variable_name,
                          units= "",flavor=flavor_name,sample="CC",
                          minval=minval,maxval=maxval,bins=bins,xlog=True) 

    
    #Zenith
    this_var_mask = const_cut & z_cut & r_cut & e_large_cut
    minval = -1.1
    maxval = 1.1
    bins = 100
    syst_bin = 20
    variable_name = "Cosine Zenith"
    units = ""
    cut_line = 0.3
    for i in range(0,4):
        a_mask = flavor_mask[i] & this_var_mask
        flavor_name = flavor_names[i]
        plot_2D_prediction(true_coszenith[a_mask], cnn_coszen[a_mask],
                            weights=weights[a_mask],
                            save=save, savefolder=save_folder,
                            bins=bins,minval=minval,maxval=maxval,
                            switch_axis = True,
                            variable=variable_name, units=units,
                            reco_name="CNN",variable_type="True",
                            flavor=flavor_name,
                            sample="CC",xline=cut_line,
                            no_contours=False)
        plot_2D_prediction(true_coszenith[a_mask], cnn_coszen[a_mask],
                            weights=weights[a_mask],
                            save=save, savefolder=save_folder,
                            bins=bins,minval=minval,maxval=maxval,
                            switch_axis = True,axis_square=True,
                            variable=variable_name, units=units,
                            reco_name="CNN",variable_type="True",
                            flavor=flavor_name,
                            sample="CC",xline=cut_line,
                            no_contours=False)
        plot_bin_slices(true_coszenith[a_mask], cnn_coszen[a_mask],
                        weights=weights[a_mask],
                        vs_predict = True,use_fraction = False,
                        bins=syst_bin,
                        min_val=minval, max_val=maxval,
                        save=save, savefolder=save_folder,
                        variable=variable_name, units=units,
                        flavor=flavor_name,sample="CC")
        plot_distributions(true_coszenith[a_mask], cnn_coszen[a_mask],
                          weights=weights[a_mask],\
                          save=save, savefolder=save_folder, 
                          reco_name = "CNN", variable=variable_name,
                          units= "",flavor=flavor_name,sample="CC",
                          minval=minval,maxval=maxval,bins=bins) 

    
    #R^2 Position
    this_var_mask = const_cut & z_cut & e_large_cut & zenith_cut
    minval = 0
    maxval = 200
    bins = 100
    syst_bin = 20
    variable_name = "R^2 Position"
    units = "(m^2)"
    cut_line = 165*165
    for i in range(0,4):
        a_mask = this_var_mask & flavor_mask[i]
        flavor_name = flavor_names[i]
        plot_2D_prediction(true_r[a_mask]*true_r[a_mask],
                            cnn_r[a_mask]*cnn_r[a_mask],
                            weights=weights[a_mask],
                            save=save, savefolder=save_folder,
                            bins=bins,minval=minval,maxval=maxval,
                            switch_axis = True,
                            variable=variable_name, units=units,
                            reco_name="CNN",variable_type="True",
                            flavor=flavor_name,
                            sample="CC",xline=cut_line,
                            no_contours=False)
        plot_2D_prediction(true_r[a_mask]*true_r[a_mask],
                            cnn_r[a_mask]*cnn_r[a_mask],
                            weights=weights[a_mask],
                            save=save, savefolder=save_folder,
                            bins=bins,minval=minval*minval,
                            maxval=maxval*maxval,
                            switch_axis = True,axis_square=True,
                            variable=variable_name, units=units,
                            reco_name="CNN",variable_type="True",
                            flavor=flavor_name,
                            sample="CC",xline=cut_line,
                            no_contours=False)
        plot_2D_prediction(true_r[a_mask]*true_r[a_mask],
                            cnn_r[a_mask]*cnn_r[a_mask],
                            weights=weights[a_mask],
                            save=save, savefolder=save_folder,
                            bins=bins,minval=minval,maxval=maxval,
                            variable=variable_name, units=units,
                            reco_name="CNN",variable_type="True",
                            flavor=flavor_name,
                            sample="CC",yline=cut_line,
                            no_contours=False)
        plot_2D_prediction(true_r[a_mask]*true_r[a_mask],
                            cnn_r[a_mask]*cnn_r[a_mask],
                            weights=weights[a_mask],
                            save=save, savefolder=save_folder,
                            bins=bins,minval=minval*minval,
                            maxval=maxval*maxval,
                            axis_square=True,
                            variable=variable_name, units=units,
                            reco_name="CNN",variable_type="True",
                            flavor=flavor_name,
                            sample="CC",yline=cut_line,
                            no_contours=False)
        plot_bin_slices(true_r[a_mask], cnn_r[a_mask],
                        weights=weights[a_mask],
                        vs_predict = True,use_fraction = False,
                        specific_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 180, 200, 220],
                        min_val=minval, max_val=200,
                        save=save, savefolder=save_folder,
                        variable="R Position", units="(m)",
                        flavor=flavor_name,sample="CC")
        plot_bin_slices(true_r[a_mask], cnn_r[a_mask],
                        weights=weights[a_mask], energy_truth=true_energy[a_mask],
                        use_fraction = False,bins=20,
                        min_val=1, max_val=200,
                        save=save, savefolder=save_folder,
                        variable="R Position", units="(m)",
                        flavor=flavor_name,sample="CC")
        plot_distributions(true_r[a_mask]*true_r[a_mask],
                          cnn_coszen[a_mask]*cnn_coszen[a_mask],
                          weights=weights[a_mask],\
                          save=save, savefolder=save_folder, 
                          reco_name = "CNN", variable=variable_name,
                          units= "",flavor=flavor_name,sample="CC",
                          minval=minval*minval,maxval=maxval*maxval,bins=bins) 


    
    #Z Position
    this_var_mask = const_cut & e_large_cut & r_cut & zenith_cut
    minval = -600
    maxval = -100
    bins = 100
    syst_bin = 20
    cut_line = [-495, -225]
    variable_name = "Z Position"
    units = "(m)"
    for i in range(0,4):
        a_mask = this_var_mask & flavor_mask[i]
        flavor_name = flavor_names[i]
        plot_2D_prediction(true_z[a_mask], cnn_z[a_mask],
                            weights=weights[a_mask],
                            save=save, savefolder=save_folder,
                            bins=bins,minval=minval,maxval=maxval,
                            switch_axis = True,
                            variable=variable_name, units=units,
                            reco_name="CNN",variable_type="True",
                            flavor=flavor_name,
                            sample="CC",xline=cut_line,
                            no_contours=False)
        plot_2D_prediction(true_z[a_mask], cnn_z[a_mask],
                            weights=weights[a_mask],
                            save=save, savefolder=save_folder,
                            bins=bins,minval=minval,maxval=maxval,
                            switch_axis = True,axis_square=True,
                            variable=variable_name, units=units,
                            reco_name="CNN",variable_type="True",
                            flavor=flavor_name,
                            sample="CC",xline=cut_line,
                            no_contours=False)
        plot_bin_slices(true_z[a_mask], cnn_z[a_mask],
                        weights=weights[a_mask],
                        vs_predict = True,use_fraction = False,
                        specific_bins=[-600,-550,-530,-510,-490,-470,-450,-430,-410,-390,-370,-350,-330,-310,-290,-270,-250,-230,-200],
                        min_val=minval, max_val=maxval,
                        save=save, savefolder=save_folder,
                        variable=variable_name, units=units,
                        flavor=flavor_name,sample="CC")
        plot_distributions(true_z[a_mask], cnn_z[a_mask],
                          weights=weights[a_mask],\
                          save=save, savefolder=save_folder, 
                          reco_name = "CNN", variable="Z Position",
                          units= "",flavor=flavor_name,sample="CC",
                          minval=minval,maxval=maxval,bins=bins) 
    

    all_cuts = nhits_cut & noise_cut & e_large_cut & r_cut & zenith_cut & z_cut
    a_mask_name_here = "All Cuts"
    plot_classification_hist(true_isNu,cnn_prob_nu,mask=all_cuts, mask_name=a_mask_name_here, units="",weights=weights,bins=50,log=True,save=save,save_folder_name=save_folder,name_prob1 = "Neutrino", name_prob0 = "Muon")

    ROC(true_isNu,cnn_prob_nu,mask=all_cuts,mask_name=a_mask_name_here,save=save,save_folder_name=save_folder,variable="Probability Neutrino")

    all_cuts = cnn_nu & nhits_cut & noise_cut & e_large_cut & r_cut & zenith_cut & z_cut

    plot_classification_hist(true_isTrack,cnn_prob_track,mask=all_cuts,mask_name=a_mask_name_here, units="",weights=weights,bins=50,log=False,save=save,save_folder_name=save_folder,name_prob1 = "Track", name_prob0 = "Cascade")

    ROC(true_isTrack,cnn_prob_track,mask=all_cuts,mask_name=a_mask_name_here,save=save,save_folder_name=save_folder,variable="Probability Track")

    quick_mask = all_cuts
    print("All Cuts:\t %.2e\t %.2e\t %.2e\t %.2e\t %.2e\t"%(sum(weights[true_isMuon&quick_mask]),sum(weights[true_isNuE&quick_mask]),sum(weights[true_isNuMu&quick_mask]),sum(weights[true_isNuTau&quick_mask]),sum(weights[true_isNu&quick_mask])))

print("Rates:\t Mu\t NuE\t NuMu\t NuTau\t Nu\n")
print("PreCut:\t %.2e\t %.2e\t %.2e\t %.2e\t %.2e\t"%(sum(weights[true_isMuon]),sum(weights[true_isNuE]),sum(weights[true_isNuMu]),sum(weights[true_isNuTau]),sum(weights[true_isNu])))
quick_mask = upgoing
print("Upgoing:\t %.2e\t %.2e\t %.2e\t %.2e\t %.2e\t"%(sum(weights[true_isMuon&quick_mask]),sum(weights[true_isNuE&quick_mask]),sum(weights[true_isNuMu&quick_mask]),sum(weights[true_isNuTau&quick_mask]),sum(weights[true_isNu&quick_mask])))
quick_mask = noise_cut
print("Noise:\t %.2e\t %.2e\t %.2e\t %.2e\t %.2e\t"%(sum(weights[true_isMuon&quick_mask]),sum(weights[true_isNuE&quick_mask]),sum(weights[true_isNuMu&quick_mask]),sum(weights[true_isNuTau&quick_mask]),sum(weights[true_isNu&quick_mask])))
quick_mask=nhits_cut
print("nHits3:\t %.2e\t %.2e\t %.2e\t %.2e\t %.2e\t"%(sum(weights[true_isMuon&quick_mask]),sum(weights[true_isNuE&quick_mask]),sum(weights[true_isNuMu&quick_mask]),sum(weights[true_isNuTau&quick_mask]),sum(weights[true_isNu&quick_mask])))
quick_mask=ntop_cut
print("nTop3:\t %.2e\t %.2e\t %.2e\t %.2e\t %.2e\t"%(sum(weights[true_isMuon&quick_mask]),sum(weights[true_isNuE&quick_mask]),sum(weights[true_isNuMu&quick_mask]),sum(weights[true_isNuTau&quick_mask]),sum(weights[true_isNu&quick_mask])))
quick_mask=nouter_cut
print("nOuter8:\t %.2e\t %.2e\t %.2e\t %.2e\t %.2e\t"%(sum(weights[true_isMuon&quick_mask]),sum(weights[true_isNuE&quick_mask]),sum(weights[true_isNuMu&quick_mask]),sum(weights[true_isNuTau&quick_mask]),sum(weights[true_isNu&quick_mask])))

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
plt.yscale('log')
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
plt.yscale('log')
plt.title("DOMs Hit in Top 15 Layers of IceCube",fontsize=25)
plt.legend()
plt.savefig("%sNTopHist.png"%(save_folder),bbox_inches='tight')
plt.close()

plt.figure(figsize=(10,7))
maxval = 20
bins=20
plt.hist(nhit_doms[true_isMuon],range=[minval,maxval],bins=bins,weights=weights[true_isMuon],label=r'$\mu$',alpha=0.5)
plt.hist(nhit_doms[true_isNuMu],range=[minval,maxval],bins=bins,weights=weights[true_isNuMu],label=r'$\nu_\mu$',alpha=0.5)
plt.hist(nhit_doms[true_isNuE],range=[minval,maxval],bins=bins,weights=weights[true_isNuE],label=r'$\nu_e$',alpha=0.5)
plt.hist(nhit_doms[true_isNuTau],range=[minval,maxval],bins=bins,weights=weights[true_isNuTau],label=r'$\nu_\tau$',alpha=0.5)
#plt.yscale('log')
plt.ylabel("Rate (Hz)",fontsize=20)
plt.xlabel("Number DOMs Hit",fontsize=20)
plt.title("Number of DOMs with Direct Pulses",fontsize=25)
plt.legend()
plt.savefig("%sNHitsHist.png"%(save_folder),bbox_inches='tight')
plt.close()

weights = weights[mask_here]
true_isMuon = true_isMuon[mask_here]
true_isNuMu = true_isNuMu[mask_here]
true_isNuE = true_isNuE[mask_here]
true_isNuTau = true_isNuTau[mask_here]
true_isNu = true_isNu[mask_here]
true_isCC = true_isCC[mask_here]
true_energy = true_energy[mask_here]
true_coszenith = true_coszenith[mask_here]
cnn_prob_mu = cnn_prob_mu[mask_here]
cnn_prob_track = cnn_prob_track[mask_here]
cnn_nu = cnn_nu[mask_here]
truth_PID = truth[:,9][mask_here]
true_isTrack = true_isTrack[mask_here]
true_r = true_r[mask_here]
true_z = true_z[mask_here]
cnn_r = cnn_r[mask_here]
cnn_z = cnn_z[mask_here]
nhit_doms = nhit_doms[mask_here]
cnn_coszen = cnn_coszen[mask_here]
cnn_energy = cnn_energy[mask_here]
no_cuts = true_energy > 0
cnn_prob_nu = np.ones(len(cnn_prob_mu)) - cnn_prob_mu

print("%s Cut:\t %.2e\t %.2e\t %.2e\t %.2e\t %.2e\t"%(mask_name_here,sum(weights[true_isMuon]),sum(weights[true_isNuE]),sum(weights[true_isNuMu]),sum(weights[true_isNuTau]),sum(weights[true_isNu])))




if plot_energy:
    plt.figure(figsize=(10,7))
    emin = 0
    emax = 500
    bins = 10**(np.arange(0,2.7,0.1))
    plt.hist(true_energy,range=[emin,emax],bins=bins,weights=weights)
    plt.xlabel("True Energy (GeV)")
    plt.title("True Energy Distribution")
    plt.ylabel("Rate (Hz)")
    plt.yscale("log")
    plt.xscale("log")
    plt.savefig("%sEnergyHist.png"%(save_folder),bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10,7))
    bins = 50 #10**(np.linspace(0,5e-6,50))
    plt.hist(weights[true_isMuon],bins=bins)
    plt.xlabel("Weights (Hz)")
    #plt.xscale('log')
    #plt.yscale('log')
    plt.title("True Muon Weights Distribution")
    plt.savefig("%sWeightsHist.png"%(save_folder),bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10,10))
    plt.title("Weights vs Energy for True Muon",fontsize=25)
    ybins = 10**(np.linspace(0,5e-6,50))
    xbins = (np.linspace(np.min(cnn_energy[true_isMuon]),np.max(cnn_energy[true_isMuon]),50))
    plt.hist2d(cnn_energy[true_isMuon],weights[true_isMuon],bins=bins,cmap='viridis_r',cmin=1)
    cbat = plt.colorbar()
    #plt.yscale('log')
    plt.ylabel("Weights (Hz)",fontsize=20)
    plt.xlabel("CNN Energy (GeV)",fontsize=20)
    plt.savefig("%sWeightsEnergy2DHist.png"%(save_folder),bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10,10))
    plt.title("Weights vs Cos Zenith for True Muon",fontsize=25)
    plt.hist2d(cnn_coszen[true_isMuon],weights[true_isMuon],bins=bins,cmap='viridis_r',cmin=1)
    cbat = plt.colorbar()
    #plt.yscale('log')
    plt.ylabel("Weights (Hz)",fontsize=20)
    plt.xlabel("CNN Cos Zenith",fontsize=20)
    plt.savefig("%sWeightsCosZen2DHist.png"%(save_folder),bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10,10))
    plt.title("Weights vs Z for True Muon",fontsize=25)
    plt.hist2d(true_z[true_isMuon],weights[true_isMuon],bins=bins,cmap='viridis_r',cmin=1)
    cbat = plt.colorbar()
    #plt.yscale('log')
    plt.ylabel("Weights (Hz)",fontsize=20)
    plt.xlabel("True Z (m)",fontsize=20)
    plt.savefig("%sWeightsZ2DHist.png"%(save_folder),bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10,7))
    bins = 10**(np.arange(0,2.7,0.1))
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

    a_mask_here = true_isNuMu&true_isCC
    a_flavor_here = "NuMu"
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
                                bins=100,minval=0,maxval=200,axis_square=True,
                                variable="Energy", units="(GeV)",
                                reco_name="CNN",flavor=a_flavor_here,sample="CC")
    plot_2D_prediction(true_energy[a_mask_here],
                                retro_energy[a_mask_here],
                                weights=weights[a_mask_here],
                                save=save, savefolder=save_folder,
                                bins=100,minval=0,maxval=200,axis_square=True,
                                variable="Energy", units="(GeV)",
                                reco_name="Retro",flavor=a_flavor_here,sample="CC")
    plot_bin_slices(true_energy[a_mask_here],
                            cnn_energy[a_mask_here], 
                            old_reco = retro_energy[a_mask_here],
                            weights=weights[a_mask_here],
                            vs_predict = True,\
                            use_fraction = True, bins=20,
                            min_val=1, max_val=300,\
                            save=save, savefolder=save_folder,\
                            variable="Energy", units="(GeV)",
                            flavor=a_flavor_here,sample="CC")
    a_mask_here = true_isNuE&true_isCC
    a_flavor_here = "NuE"
    plot_2D_prediction(true_energy[a_mask_here],
                                cnn_energy[a_mask_here],
                                weights=weights[a_mask_here],
                                save=save, savefolder=save_folder,
                                bins=100,minval=0,maxval=200,axis_square=True,
                                variable="Energy", units="(GeV)",
                                reco_name="CNN",flavor=a_flavor_here,sample="CC")
    plot_2D_prediction(true_energy[a_mask_here],
                                retro_energy[a_mask_here],
                                weights=weights[a_mask_here],
                                save=save, savefolder=save_folder,
                                bins=100,minval=0,maxval=200,axis_square=True,
                                variable="Energy", units="(GeV)",
                                reco_name="Retro",flavor=a_flavor_here,sample="CC")
    a_mask_here = true_isNuTau&true_isCC
    a_flavor_here = "NuTau"
    plot_2D_prediction(true_energy[a_mask_here],
                                cnn_energy[a_mask_here],
                                weights=weights[a_mask_here],
                                save=save, savefolder=save_folder,
                                bins=100,minval=0,maxval=200,axis_square=True,
                                variable="Energy", units="(GeV)",
                                reco_name="CNN",flavor=a_flavor_here,sample="CC")
    plot_2D_prediction(true_energy[a_mask_here],
                                retro_energy[a_mask_here],
                                weights=weights[a_mask_here],
                                save=save, savefolder=save_folder,
                                bins=100,minval=0,maxval=200,axis_square=True,
                                variable="Energy", units="(GeV)",
                                reco_name="Retro",flavor=a_flavor_here,sample="CC")
    a_mask_here = true_isNu&true_isCC
    a_flavor_here = "Nu"
    plot_2D_prediction(true_energy[a_mask_here],
                                cnn_energy[a_mask_here],
                                weights=weights[a_mask_here],
                                save=save, savefolder=save_folder,
                                bins=100,minval=0,maxval=200,axis_square=True,
                                variable="Energy", units="(GeV)",
                                reco_name="CNN",flavor=a_flavor_here,sample="CC")
    plot_2D_prediction(true_energy[a_mask_here],
                                retro_energy[a_mask_here],
                                weights=weights[a_mask_here],
                                save=save, savefolder=save_folder,
                                bins=100,minval=0,maxval=200,axis_square=True,
                                variable="Energy", units="(GeV)",
                                reco_name="Retro",flavor=a_flavor_here,sample="CC")

if plot_main:
    plt.figure(figsize=(10,7))
    plt.hist(cnn_prob_mu[true_isMuon],bins=50,weights=weights[true_isMuon],label=r'$\mu$',alpha=0.5)
    plt.hist(cnn_prob_mu[true_isNuMu],bins=50,weights=weights[true_isNuMu],label=r'$\nu_\mu$',alpha=0.5)
    plt.hist(cnn_prob_mu[true_isNuE],bins=50,weights=weights[true_isNuE],label=r'$\nu_e$',alpha=0.5)
    plt.hist(cnn_prob_mu[true_isNuTau],bins=50,weights=weights[true_isNuTau],label=r'$\nu_\tau$',alpha=0.5)
#plt.yscale('log')
    plt.ylabel("Rate (Hz)",fontsize=20)
    plt.xlabel("Muon Probability",fontsize=20)
    plt.title("Muon Probability Distribution",fontsize=25)
    plt.legend()
    plt.savefig("%sProbMuonParticleHist.png"%(save_folder),bbox_inches='tight')
    plt.close()


    #ROC(true_isMuon,cnn_prob_mu,mask=no_cuts,mask_name=mask_name_here,save=save,save_folder_name=save_folder)

    #plot_classification_hist(true_isMuon,cnn_prob_mu,mask=no_cuts,mask_name=mask_name_here, units="",weights=weights,bins=50,log=False,save=save,save_folder_name=save_folder,name_prob0 = "Neutrino", name_prob1 = "Muon")

    plot_classification_hist(true_isNu,cnn_prob_nu,mask=no_cuts, mask_name=mask_name_here, units="",weights=weights,bins=50,log=False,save=save,save_folder_name=save_folder,name_prob1 = "Neutrino", name_prob0 = "Muon")

    ROC(true_isNu,cnn_prob_nu,mask=no_cuts,mask_name=mask_name_here,save=save,save_folder_name=save_folder,variable="Probability Neutrino")

    plot_classification_hist(true_isTrack,cnn_prob_track,mask=no_cuts,mask_name=mask_name_here, units="",weights=weights,bins=50,log=False,save=save,save_folder_name=save_folder,name_prob1 = "Track", name_prob0 = "Cascade")

    ROC(true_isTrack,cnn_prob_track,mask=no_cuts,mask_name=mask_name_here,save=save,save_folder_name=save_folder,variable="Probability Track")

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

if check_nhits:


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

    nhit_cuts = np.arange(0,11,1)
    nhit_muon_rates = []
    nhit_numu_rates = []
    nhit_nue_rates = []
    nhit_nutau_rates = []
    nhit_nu_rates = []
    nhit_muon = []
    nhit_numu = []
    nhit_nue = []
    nhit_nutau = []
    nhit_nu = []
    print(len(nhit_doms), len(weights),len(truth_PID))
    for nhit_cut in nhit_cuts:
        
        predictionNu = nhit_doms >= nhit_cut
        predictionMu = nhit_doms < nhit_cut

        save_weights = weights[predictionNu]
        true_PID = truth_PID[predictionNu]

        muon_mask = true_PID == 13
        true_isMuon = np.array(muon_mask,dtype=bool)
        nue_mask = true_PID == 12
        true_isNuE = np.array(nue_mask,dtype=bool)
        numu_mask = true_PID == 14
        true_isNuMu = np.array(numu_mask,dtype=bool)
        nutau_mask = true_PID == 16
        true_isNuTau = np.array(nutau_mask,dtype=bool)

        
        nhit_muon.append(sum(true_isMuon))
        nhit_nue.append(sum(true_isNuE))
        nhit_numu.append(sum(true_isNuMu))
        nhit_nutau.append(sum(true_isNuTau))
        nhit_nu.append(nhit_nue[-1] + nhit_numu[-1] + nhit_nutau[-1])
        
        nhit_muon_rates.append(sum(save_weights[true_isMuon]))
        nhit_nue_rates.append(sum(save_weights[true_isNuE]))
        nhit_numu_rates.append(sum(save_weights[true_isNuMu]))
        nhit_nutau_rates.append(save_weights[true_isNuTau])
        nhit_nu_rates.append(nhit_nue_rates[-1] + nhit_numu_rates[-1] + nhit_nutau_rates[-1])
    
    fig,ax = plt.subplots(2,1,figsize=(10,10))
    plt.suptitle("Rate SANTA nhit DOMs",fontsize=25)
    plt.plot(nhit_cuts,nhit_muon_rates,color='orange',label=r'$\mu$')
    plt.plot(nhit_cuts,nhit_nue_rates,color='g',label=r'$\nu_e$')
    plt.plot(nhit_cuts,nhit_numu_rates,color='b',label=r'$\nu_\mu$')
    plt.plot(nhit_cuts,nhit_nutau_rates,color='purple',label=r'$\nu_\tau$')
    plt.set_xlabel("n hit DOMs cut",fontsize=20)
    plt.set_ylabel("Rate (Hz)",fontsize=20)
    plt.legend(fontsize=20)
    plt.savefig("%snhitRates.png"%(save_folder),bbox_inches='tight')
    plt.close()
    
    
    fig,ax = plt.subplots(2,1,figsize=(10,10))
    plt.suptitle("SANTA nhit DOMs",fontsize=25)
    ax[0].plot(nhit_cuts,nhit_muon,color='orange',label=r'$\mu$')
    ax[0].plot(nhit_cuts,nhit_nue,color='g',label=r'$\nu_e$')
    ax[0].plot(nhit_cuts,nhit_numu,color='b',label=r'$\nu_\mu$')
    ax[0].plot(nhit_cuts,nhit_nutau,color='purple',label=r'$\nu_\tau$')
    ax[0].set_xlabel("n hit DOMs cut",fontsize=20)
    ax[0].set_ylabel("Number Events",fontsize=20)
    ax[0].legend(fontsize=20)
    
    ax[1].plot(nhit_cuts,np.true_divide(nhit_muon/nhit_muon[0]),color='orange',label=r'$\mu$')
    ax[1].plot(nhit_cuts,np.true_divide(nhit_nue/nhit_nue[0]),color='g',label=r'$\nu_e$')
    ax[1].plot(nhit_cuts,np.true_divide(nhit_numu/nhit_nue[0]),color='b',label=r'$\nu_\mu$')
    ax[1].plot(nhit_cuts,np.true_divide(nhit_nutau/nhit_nutau[0]),color='purple',label=r'$\nu_\tau$')
    ax[1].xlabel("n hit DOMs cut",fontsize=20)
    ax[1].ylabel("Fraction Events Remaining",fontsize=20)
    ax[1].legend(fontsize=20)
    
    plt.savefig("%snhitRates.png"%(save_folder),bbox_inches='tight')
    plt.close()


step=0.01
cut_values = np.arange(step,1+step,step)
muon_rates = []
numu_rates = []
nue_rates = []
nutau_rates = []
nu_rates = []
for muon_cut in cut_values:
    
    muon_rate, nue_rate, numu_rate, nutau_rate = return_rates(truth_PID, cnn_prob_mu, weights, muon_cut)

    muon_rates.append(muon_rate)
    nue_rates.append(nue_rate)
    numu_rates.append(numu_rate)
    nutau_rates.append(nutau_rate)
    nu_rates.append(nue_rate + numu_rate + nutau_rate)

if plot_rcut_muonrate:
    target_muon_rate = 0.000034
    r36_cuts = np.arange(90,200,5)
    #Static masks
    e_small_cut = np.logical_and(cnn_energy > e_min_cut, cnn_energy < 100)
    e_large_cut = np.logical_and(cnn_energy > e_min_cut, cnn_energy < 200)
    z_cut = np.logical_and(cnn_z > z_min_value, cnn_z < z_max_value)
    static_mask_small = np.logical_and(e_small_cut, z_cut)
    static_mask_large = np.logical_and(e_large_cut, z_cut)

    mu_Lthreshold = []
    nu_Lrates_save = []
    numu_Lrates_save = []
    mu_Lrates_save = []
    mu_Sthreshold = []
    nu_Srates_save = []
    numu_Srates_save = []
    mu_Srates_save = []
    for r36_check in r36_cuts:
        #r36 mask
        r36_mask = cnn_r < r36_check
        small_mask = np.logical_and(static_mask_small, r36_mask)
        large_mask = np.logical_and(static_mask_large, r36_mask)
        
        #find muon_cut for specific r36
        muon_Lrates = []
        nu_Lrates = []
        numu_Lrates = []
        muon_Srates = []
        numu_Srates = []
        nu_Srates = []
        difference_to_target_L = []
        difference_to_target_S = []
        for muon_cut in cut_values:

            muon_rate, nue_rate, numu_rate, nutau_rate = return_rates(truth_PID[large_mask], cnn_prob_mu[large_mask], weights[large_mask], muon_cut)
            muon_Lrates.append(muon_rate)
            nu_Lrates.append(nue_rate + numu_rate + nutau_rate)
            numu_Lrates.append(numu_rate)
            difference_to_target_L.append(np.abs(muon_rate - target_muon_rate))
            
            muon_rate, nue_rate, numu_rate, nutau_rate = return_rates(truth_PID[small_mask], cnn_prob_mu[small_mask], weights[small_mask], muon_cut)
            muon_Srates.append(muon_rate)
            nu_Srates.append(nue_rate + numu_rate + nutau_rate)
            numu_Srates.append(numu_rate)
            difference_to_target_S.append(np.abs(muon_rate - target_muon_rate))
        
        
        closest_target_Lindex = difference_to_target_L.index(min(difference_to_target_L))
        mu_Lthreshold.append(cut_values[closest_target_Lindex])
        nu_Lrates_save.append(nu_Lrates[closest_target_Lindex])
        numu_Lrates_save.append(numu_Lrates[closest_target_Lindex])
        mu_Lrates_save.append(muon_Lrates[closest_target_Lindex])
        closest_target_Sindex = difference_to_target_S.index(min(difference_to_target_S))
        mu_Sthreshold.append(cut_values[closest_target_Sindex])
        nu_Srates_save.append(nu_Srates[closest_target_Sindex])
        numu_Srates_save.append(numu_Srates[closest_target_Sindex])
        mu_Srates_save.append(muon_Srates[closest_target_Sindex])
    
    print(mu_Lthreshold,mu_Sthreshold)
    print(mu_Lrates_save,mu_Srates_save)
    plt.figure(figsize=(10,7))
    plt.title("Neutrino Rate vs. R36 Cut (const Mu rate = .034mHz)",fontsize=25)
    plt.plot(r36_cuts,nu_Srates_save,color='g',label=r'$\nu$ Energy [3, 100]')
    plt.plot(r36_cuts,numu_Srates_save,color='g',linestyle="dashed",label=r'$\nu_\mu$ Energy [3, 100]')
    plt.plot(r36_cuts,nu_Lrates_save,color='b',label=r'$\nu$ Energy [3, 200]')
    plt.plot(r36_cuts,numu_Lrates_save,color='b',linestyle="dashed",label=r'$\nu_\mu$ Energy [3, 200]')
    plt.axhline(0.000991,color='k',label=r'oscNext $\nu$ rate')
    plt.axhline(0.000619,color='k',linestyle="dashed",label=r'oscNext $\nu$ rate')
    plt.xlabel("R36 cut values (< X m)",fontsize=20)
    plt.ylabel("Remaining Neutrino Rate (Hz)",fontsize=20)
    plt.legend(fontsize=20)
    plt.savefig("%sNuVsR36Rates.png"%(save_folder),bbox_inches='tight')
    plt.close()

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
    cnn_nu = cnn_prob_mu <= cnn_mu_cut
    cnn_mu = cnn_prob_mu > cnn_mu_cut
    correct_mu = np.logical_and(cnn_mu, true_isMuon)
    wrong_mu = np.logical_and(cnn_mu, true_isNu)
    correct_nu = np.logical_and(cnn_nu, true_isNu)
    wrong_nu = np.logical_and(cnn_nu, true_isMuon)

    print("Rates After Cut at %f:\t Mu\t NuE\t NuMu\t NuTau\t Nu\n"%cnn_mu_cut)
    r_cut = cnn_r < r_cut_value
    r_cut_out = cnn_r > r_cut_value
    z_cut = np.logical_and(cnn_z > z_min_value, cnn_z < z_max_value)
    z_cut_out = np.logical_or(cnn_z < z_min_value, cnn_z > z_max_value)
    vertex_cut = np.logical_and(r_cut, z_cut)
    vertex_cut_out = np.logical_or(r_cut_out, z_cut_out)
    e_small_cut = np.logical_and(cnn_energy > e_min_cut, cnn_energy < 100)
    e_large_cut = np.logical_and(cnn_energy > e_min_cut, cnn_energy < 200)
    print("%s Cut:\t %.2e\t %.2e\t %.2e\t %.2e\t %.2e\t"%("E < 100",sum(weights[true_isMuon&e_small_cut]),sum(weights[true_isNuE&e_small_cut]),sum(weights[true_isNuMu&e_small_cut]),sum(weights[true_isNuTau&e_small_cut]),sum(weights[true_isNu&e_small_cut])))
    print("%s Cut:\t %.2e\t %.2e\t %.2e\t %.2e\t %.2e\t"%("E < 200",sum(weights[true_isMuon&e_large_cut]),sum(weights[true_isNuE&e_large_cut]),sum(weights[true_isNuMu&e_large_cut]),sum(weights[true_isNuTau&e_large_cut]),sum(weights[true_isNu&e_large_cut])))
    print("%s Cut:\t %.2e\t %.2e\t %.2e\t %.2e\t %.2e\t"%("R <%i"%r_cut_value,sum(weights[true_isMuon&r_cut]),sum(weights[true_isNuE&r_cut]),sum(weights[true_isNuMu&r_cut]),sum(weights[true_isNuTau&r_cut]),sum(weights[true_isNu&r_cut])))
    print("%s Cut:\t %.2e\t %.2e\t %.2e\t %.2e\t %.2e\t"%("Z in [%i, %i]"%(z_min_value,z_max_value),sum(weights[true_isMuon&z_cut]),sum(weights[true_isNuE&z_cut]),sum(weights[true_isNuMu&z_cut]),sum(weights[true_isNuTau&z_cut]),sum(weights[true_isNu&z_cut])))
    print("%s Cut:\t %.2e\t %.2e\t %.2e\t %.2e\t %.2e\t"%("R&Z",sum(weights[true_isMuon&vertex_cut]),sum(weights[true_isNuE&vertex_cut]),sum(weights[true_isNuMu&vertex_cut]),sum(weights[true_isNuTau&vertex_cut]),sum(weights[true_isNu&vertex_cut])))
    print("%s Cut:\t %.2e\t %.2e\t %.2e\t %.2e\t %.2e\t"%("Muon at %.2f"%cnn_mu_cut,sum(weights[true_isMuon&cnn_nu]),sum(weights[true_isNuE&cnn_nu]),sum(weights[true_isNuMu&cnn_nu]),sum(weights[true_isNuTau&cnn_nu]),sum(weights[true_isNu&cnn_nu])))
    print("%s Cut:\t %.2e\t %.2e\t %.2e\t %.2e\t %.2e\t"%("Muon & R & Z",sum(weights[true_isMuon&cnn_nu&vertex_cut]),sum(weights[true_isNuE&cnn_nu&vertex_cut]),sum(weights[true_isNuMu&cnn_nu&vertex_cut]),sum(weights[true_isNuTau&cnn_nu&vertex_cut]),sum(weights[true_isNu&cnn_nu&vertex_cut])))
    print("%s Cut:\t %.2e\t %.2e\t %.2e\t %.2e\t %.2e\t"%("Muon & R & Z & E100",sum(weights[true_isMuon&cnn_nu&vertex_cut&e_small_cut]),sum(weights[true_isNuE&cnn_nu&vertex_cut&e_small_cut]),sum(weights[true_isNuMu&cnn_nu&vertex_cut&e_small_cut]),sum(weights[true_isNuTau&cnn_nu&vertex_cut&e_small_cut]),sum(weights[true_isNu&cnn_nu&vertex_cut&e_small_cut])))
    print("%s Cut:\t %.2e\t %.2e\t %.2e\t %.2e\t %.2e\t"%("Muon & R & Z & E200",sum(weights[true_isMuon&cnn_nu&vertex_cut&e_large_cut]),sum(weights[true_isNuE&cnn_nu&vertex_cut&e_large_cut]),sum(weights[true_isNuMu&cnn_nu&vertex_cut&e_large_cut]),sum(weights[true_isNuTau&cnn_nu&vertex_cut&e_large_cut]),sum(weights[true_isNu&cnn_nu&vertex_cut&e_large_cut])))

    if plot_cut_effect:
        # R PLOT
        fig,ax = plt.subplots(2,1,figsize=(10,10))
        fig.suptitle("Reconstructed Radius After Cut",fontsize=25)
        ax[0].hist(cnn_r[true_isNu],bins=50,weights=weights[true_isNu],range=[0,300],label=r'Full Distribution',alpha=0.5)
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
        
        ax[1].hist(cnn_r[true_isMuon],bins=50,weights=weights[true_isMuon],range=[0,300],label=r'Full Distribution',alpha=0.5)
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
        ax[0].hist(cnn_r[true_isNu]*cnn_r[true_isNu],bins=50,weights=weights[true_isNu],range=[a_min,a_max],label=r'Full Distribution',alpha=0.5)
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
        
        ax[1].hist(cnn_r[true_isMuon]*cnn_r[true_isMuon],bins=50,weights=weights[true_isMuon],range=[a_min,a_max],label=r'Full Distribution',alpha=0.5)
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
        ax[0].hist(cnn_coszen[true_isNu],bins=50,weights=weights[true_isNu],range=[a_min,a_max],label=r'Full Distribution',alpha=0.5)
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
        
        ax[1].hist(cnn_coszen[true_isMuon],bins=50,weights=weights[true_isMuon],range=[a_min,a_max],label=r'Full Distribution',alpha=0.5)
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
        bins = 10**(np.arange(0,2.3,0.1))
        fig,ax = plt.subplots(2,1,figsize=(10,10))
        fig.suptitle("Reconstructed Energy After Cut",fontsize=25)
        ax[0].hist(cnn_energy[true_isNu],bins=bins,weights=weights[true_isNu],range=[a_min,a_max],label=r'Full Distribution',alpha=0.5)
        ax[0].hist(cnn_energy[true_isNu&cnn_nu],bins=bins,weights=weights[cnn_nu&true_isNu],range=[a_min,a_max],label=r'After Muon Cut',alpha=0.5)
        ax[0].hist(cnn_energy[true_isNu&vertex_cut],bins=bins,weights=weights[true_isNu&vertex_cut],range=[a_min,a_max],label=r'After Vertex Cut',alpha=0.5)
        ax[0].hist(cnn_energy[true_isNu&cnn_nu&vertex_cut],bins=bins,weights=weights[true_isNu&cnn_nu&vertex_cut],range=[a_min,a_max],label=r'After Vertex + Muon Cut',alpha=0.5)
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
        
        ax[1].hist(cnn_energy[true_isMuon],bins=bins,weights=weights[true_isMuon],range=[a_min,a_max],label=r'Full Distribution',alpha=0.5)
        ax[1].hist(cnn_energy[true_isMuon&cnn_nu],bins=bins,weights=weights[cnn_nu&true_isMuon],range=[a_min,a_max],label=r'After Muon Cut',alpha=0.5)
        ax[1].hist(cnn_energy[true_isMuon&vertex_cut],bins=bins,weights=weights[true_isMuon&vertex_cut],range=[a_min,a_max],label=r'After Vertex Cut',alpha=0.5)
        ax[1].hist(cnn_energy[true_isMuon&cnn_nu&vertex_cut],bins=bins,weights=weights[true_isMuon&cnn_nu&vertex_cut],range=[a_min,a_max],label=r'After Vertex + Muon Cut',alpha=0.5)
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
        ax[0].hist(cnn_z[true_isNu],bins=50,weights=weights[true_isNu],range=[a_min,a_max],label=r'Full Distribution',alpha=0.5)
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
        
        ax[1].hist(cnn_z[true_isMuon],bins=50,weights=weights[true_isMuon],range=[a_min,a_max],label=r'Full Distribution',alpha=0.5)
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
        a_mask = true_isNu & cnn_nu & e_large_cut & vertex_cut
        a_title = "True Neutrino: All Cuts Energy %i-200"%e_min_cut
        #bins= 10**(np.arange(0,2.7,0.1))
        plot_distributions(true_energy[a_mask], cnn_energy[a_mask],
                              weights=weights[a_mask],\
                              save=save, savefolder=save_folder, 
                              variable="Reconstructed Energy",
                              units= "(GeV)",title=a_title,
                              minval=0,maxval=500,bins=100,xlog=True) 
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
        
        a_mask = true_isMuon & cnn_nu
        a_title = "True Muon: Only Muon Cut"
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
       
        nue_numu_mask = np.logical_or(true_isNuMu, true_isNuE)
        a_mask = nue_numu_mask & cnn_nu & e_large_cut & vertex_cut
        a_title = "True NuMu NuE: All Cuts Energy %i-200"%e_min_cut
        #bins= 10**(np.arange(0,2.7,0.1))
        plot_distributions(true_energy[a_mask], cnn_energy[a_mask],
                              weights=weights[a_mask],\
                              save=save, savefolder=save_folder, 
                              variable="Reconstructed Energy",
                              units= "(GeV)",title=a_title,
                              minval=0,maxval=500,bins=100,xlog=True) 

    print(sum(weights[correct_mu]),sum(weights[wrong_mu]),sum(weights[wrong_nu]),sum(weights[correct_nu]))

    confusion_matrix(true_isMuon, cnn_prob_mu, 0.45, mask=no_cuts, mask_name=mask_name_here, weights=weights, save=True, save_folder_name=save_folder, name_prob1 = "Muon", name_prob0 = "Neutrino")


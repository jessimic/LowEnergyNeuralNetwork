################################
# Plotting truth vs PEGLEG reconstruction
#   Generates 3 plot distributions for energy and zenith
#   plot_single_resolution
#   plot_energy_slices
#   plot_distributions
#################################

import numpy as np
import h5py
import os, sys
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy import interpolate
import scipy.stats

import matplotlib
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)

font = {'family' : 'normal',
        'weight' : 'normal',
                'size'   : 22}

matplotlib.rc('font', **font)

numu = False
nue = True
if numu:
    input_file = "/mnt/scratch/micall12/training_files/Level5p_IC86.2013_genie_numu.014640.all.all.lt200_vertexDC_file00.hdf5"
    save_folder_name = "/mnt/home/micall12/DNN_LE/output_plots/pegleg_resolution/numu_CC/"
if nue:
    input_file = "/mnt/scratch/micall12/training_files/Level5p_IC86.2013_geneie_nue.01240.all.all.lt200_vertexDC_file00.hdf5"
    save_folder_name = "/mnt/home/micall12/DNN_LE/output_plots/pegleg_resolution/nue_CC/"
fractional=False

f = h5py.File(input_file, 'r')
f = h5py.File(input_file, 'r')
labels = f['labels'][:]
reco = f['reco_labels'][:]
f.close()
del f

Y_labels_all = np.array(labels)
reco_labels_all = np.array(reco)

Y_energy = np.array(Y_labels_all[:,0])
Y_coszenith = np.array(np.cos(Y_labels_all[:,1]))
reco_energy = np.array(reco_labels_all[:,0])
reco_coszenith = np.array(np.cos(reco_labels_all[:,1]))

#Mask energy
min_energy = 5.
max_energy = 100.
Y_energy = np.array(Y_energy)
Y_coszenith = np.array(Y_coszenith)
reco_energy = np.array(reco_energy)
reco_coszenith = np.array(reco_coszenith)
Y_mask = np.logical_and(Y_energy>min_energy,Y_energy<max_energy)


# isCC = index 11
mask_CC = Y_labels_all[:,11]==1

# multiple masks
energy_cc = np.logical_and(Y_mask,mask_CC)

from PlottingFunctions import plot_single_resolution
from PlottingFunctions import plot_energy_slices
from PlottingFunctions import plot_distributions

plot_single_resolution(Y_energy[energy_cc],reco_energy[energy_cc],bins=100, use_fraction=fractional,\
                        save=True,savefolder=save_folder_name,\
                        variable="Energy", units = "GeV")

plot_single_resolution(Y_coszenith[energy_cc],reco_coszenith[energy_cc],bins=100, use_fraction=fractional,\
                        save=True,savefolder=save_folder_name,\
                        variable="CosZenith", units = "")

plot_energy_slices(Y_energy[energy_cc], reco_energy[energy_cc], \
                       use_fraction = fractional, \
                       bins=10,min_val=5.,max_val=100.,\
                       save=True,savefolder=save_folder_name)

plot_energy_slices(Y_coszenith[energy_cc], reco_coszenith[energy_cc], \
                       use_fraction = fractional, 
                       bins=10,min_val=-1.,max_val=1.,\
                       save=True,savefolder=save_folder_name,\
                        variable="CosZenith",units="")

plot_distributions(Y_energy[energy_cc], reco_energy[energy_cc],\
                    save=True,savefolder=save_folder_name,\
                    variable="Energy",units="GeV")

plot_distributions(Y_coszenith[energy_cc], reco_coszenith[energy_cc],\
                    save=True,savefolder=save_folder_name,\
                    variable="CozZenith",units="")

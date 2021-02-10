#########################
# Version of CNN on 12 May 2020
# 
# Evaluates net for given model and plots
# Takes in ONE file to Test on, can compare to old reco
# Runs Energy, Zenith, Track length (1 variable energy or zenith, 2 = energy then zenith, 3 = EZT)
#   Inputs:
#       -i input_file:  name of ONE file 
#       -d path:        path to input files
#       -o ouput_dir:   path to output_plots directory
#       -n name:        name for folder in output_plots that has the model you want to load
#       -e epochs:      epoch number of the model you want to load
#       --variables:    Number of variables to train for (1 = energy or zenith, 2 = EZ, 3 = EZT)
#       --first_variable: Which variable to train for, energy or zenith (for num_var = 1 only)
#       --compare_reco: boolean flag, true means you want to compare to a old reco (pegleg, retro, etc.)
#       -t test:        Name of reco to compare against, with "oscnext" used for no reco to compare with
####################################

import numpy as np
import h5py
import time
import os, sys
import random
from collections import OrderedDict
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_file",type=str,default=None,
                    dest="input_file", help="names for test only input file")
parser.add_argument("-d", "--path",type=str,default='/data/icecube/jmicallef/processed_CNN_files/',
                    dest="path", help="path to input files")
parser.add_argument("-o", "--output_dir",type=str,default='/home/users/jmicallef/LowEnergyNeuralNetwork/',
                    dest="output_dir", help="path to output_plots directory, do not end in /")
parser.add_argument("-n", "--name",type=str,default=None,
                    dest="name", help="name for output directory and where model file located")
parser.add_argument("--variables", type=int,default=2,
                    dest="output_variables", help="1 for [energy], 2 for [energy, zenith], 3 for [energy, zenith, track]")
parser.add_argument("--mask_zenith", default=False,action='store_true',
                        dest='mask_zenith',help="mask zenith for up and down going")
parser.add_argument("--z_values", type=str,default=None,
                        dest='z_values',help="Options are gt0 or lt0")
parser.add_argument("--emax",type=float,default=100.,
                        dest='emax',help="MAX ENERGY TO PLOT")
parser.add_argument("--efactor",type=float,default=100.,
                        dest='efactor',help="ENERGY TO MULTIPLY BY!")
parser.add_argument("--test",type=str,default="PegLeg",
                    dest='reco_name',help="Name of reco in file")
args = parser.parse_args()

test_file = args.path + args.input_file
output_variables = args.output_variables
filename = args.name

min_energy = 5
energy_factor = args.efactor
max_energy = args.emax
mask_zenith = args.mask_zenith
z_values = args.z_values

save = True
save_folder_name = "%soutput_plots/%s/"%(args.output_dir,filename)
if save==True:
    if os.path.isdir(save_folder_name) != True:
        os.mkdir(save_folder_name)

reco_name = args.reco_name
save_folder_name += "%s_plots/"%reco_name
if save==True:
    if os.path.isdir(save_folder_name) != True:  
        os.mkdir(save_folder_name)
  
#Load in test data
print("Testing on %s"%test_file)
f = h5py.File(test_file, 'r')
Y_test_use = f['Y_test'][:]
reco_test_use = f['reco_test'][:]
f.close
del f
print(Y_test_use.shape,reco_test_use.shape)

#mask_energy_train = np.logical_and(np.array(Y_test_use[:,0])>min_energy/max_energy,np.array(Y_test_use[:,0])<1.0)
#Y_test_use = np.array(Y_test_use)[mask_energy_train]
#X_test_DC_use = np.array(X_test_DC_use)[mask_energy_train]
#X_test_IC_use = np.array(X_test_IC_use)[mask_energy_train]
#if compare_reco:
#    reco_test_use = np.array(reco_test_use)[mask_energy_train]
print("TRANSFORMING ZENITH TO COS(ZENITH)")
reco_test_use[:,1] = np.cos(reco_test_use[:,1])

if mask_zenith:
    print("MANUALLY GETTING RID OF HALF THE EVENTS (UPGOING/DOWNGOING ONLY)")
    if z_values == "gt0":
        maxvals = [max_energy, 1., 0.]
        minvals = [min_energy, 0., 0.]
        mask_zenith = np.array(Y_test_use[:,1])>0.0
    if z_values == "lt0":
        maxvals = [max_energy, 0., 0.]
        minvals = [min_energy, -1., 0.]
        mask_zenith = np.array(Y_test_use[:,1])<0.0
    Y_test_use = Y_test_use[mask_zenith]
    reco_test_use = reco_test_use[mask_zenith]


### MAKE THE PLOTS ###
from PlottingFunctions import plot_single_resolution
from PlottingFunctions import plot_2D_prediction
from PlottingFunctions import plot_2D_prediction_fraction
from PlottingFunctions import plot_bin_slices
from PlottingFunctions import plot_distributions
from PlottingFunctions import plot_length_energy

plots_names = ["Energy", "CosZenith", "Track"]
plots_units = ["(GeV)", "", "(m)"]
maxabs_factors = [energy_factor, 1., 200.]
maxvals = [max_energy, 1., 0.]
minvals = [min_energy, -1., 0.]
use_fractions = [True, False, True]
bins_array = [100,100,100]
if output_variables == 3: 
    maxvals = [max_energy, 1., max(Y_test_use[:,2])*maxabs_factor[2]]

for num in range(0,output_variables):

    NN_index = num
    true_index = num
    name_index = num
    plot_name = plots_names[name_index]
    plot_units = plots_units[name_index]
    maxabs_factor = maxabs_factors[name_index]
    maxval = maxvals[name_index]
    minval = minvals[name_index]
    use_frac = use_fractions[name_index]
    bins = bins_array[name_index]
    print("Plotting %s at position %i in true test output and %i in NN test output"%(plot_name, true_index,NN_index))
    
    plot_2D_prediction(Y_test_use[:,true_index]*maxabs_factor,\
                        reco_test_use[:,NN_index],\
                        save,save_folder_name,bins=bins,\
                        minval=minval,maxval=maxval,cut_truth=True,axis_square=True,\
                        variable=plot_name,units=plot_units, reco_name=reco_name)
    plot_2D_prediction(Y_test_use[:,true_index]*maxabs_factor, reco_test_use[:,NN_index],\
                        save,save_folder_name,bins=bins,\
                        minval=None,maxval=None,\
                        variable=plot_name,units=plot_units,  reco_name=reco_name)
    plot_single_resolution(Y_test_use[:,true_index]*maxabs_factor,\
                    reco_test_use[:,NN_index],\
                   minaxis=-maxval,maxaxis=maxval,bins=bins,
                   save=save,savefolder=save_folder_name,\
                   variable=plot_name,units=plot_units)
    plot_bin_slices(Y_test_use[:,true_index]*maxabs_factor, reco_test_use[:,NN_index],\
                    use_fraction = False,\
                    bins=10,min_val=minval,max_val=maxval,\
                    save=True,savefolder=save_folder_name,\
                    variable=plot_name,units=plot_units)
   
    if num ==0:
        plot_2D_prediction_fraction(Y_test_use[:,true_index]*maxabs_factor,\
                        reco_test_use[:,NN_index],\
                        save,save_folder_name,bins=bins,\
                        minval=0,maxval=2,\
                        variable=plot_name,units=plot_units)
        plot_single_resolution(Y_test_use[:,true_index]*maxabs_factor,\
                   reco_test_use[:,NN_index]*maxabs_factor,\
                   use_fraction=True,\
                   minaxis=-2.,maxaxis=2.,
                   save=save,savefolder=save_folder_name,\
                   variable=plot_name,units=plot_units)
        plot_bin_slices(Y_test_use[:,true_index]*maxabs_factor, reco_test_use[:,NN_index],\
                    use_fraction = use_frac,\
                    bins=10,min_val=minval,max_val=maxval,\
                    save=True,savefolder=save_folder_name,\
                    variable=plot_name,units=plot_units)
        #plot_length_energy(Y_test_use, reco_test_use[:,NN_index]*maxabs_factor,\
        #                    use_fraction=True,ebins=20,tbins=20,\
        #                    emin=minvals[first_var_index], emax=maxvals[first_var_index],\
        #                    tmin=0.,tmax=450.,tfactor=maxabs_factors[2],\
        #                    savefolder=save_folder_name)
    if num ==1:
        plot_bin_slices(Y_test_use[:,true_index], reco_test_use[:,NN_index], \
                       energy_truth=Y_test_use[:,0]*max_energy, \
                       use_fraction = False, \
                       bins=10,min_val=min_energy,max_val=max_energy,\
                       save=True,savefolder=save_folder_name,\
                       variable=plot_name,units=plot_units)

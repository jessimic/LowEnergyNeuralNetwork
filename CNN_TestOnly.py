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
parser.add_argument("-e","--epoch", type=int,default=None,
                    dest="epoch", help="which model number (number of epochs) to load in")
parser.add_argument("--variables", type=int,default=1,
                    dest="output_variables", help="1 for [energy], 2 for [energy, zenith], 3 for [energy, zenith, track]")
parser.add_argument("--first_variable", type=str,default="energy",
                    dest="first_variable", help = "name for first variable (energy, zenith only two supported)")
parser.add_argument("--compare_reco", default=False,action='store_true',
                        dest='compare_reco',help="use flag to compare to old reco vs. NN")
parser.add_argument("-t","--test", type=str,default="oscnext",
                        dest='test',help="name of reco")
parser.add_argument("--mask_zenith", default=False,action='store_true',
                        dest='mask_zenith',help="mask zenith for up and down going")
parser.add_argument("--z_values", type=str,default=None,
                        dest='z_values',help="Options are gt0 or lt0")
parser.add_argument("--emax",type=float,default=100.,
                        dest='emax',help="Max energy for use for plotting")
parser.add_argument("--efactor",type=float,default=100.,
                        dest='efactor',help="ENERGY FACTOR TO MULTIPLY BY!")
args = parser.parse_args()

test_file = args.path + args.input_file
output_variables = args.output_variables
filename = args.name
epoch = args.epoch
compare_reco = args.compare_reco
print("Comparing reco?", compare_reco)

dropout = 0.2
learning_rate = 1e-3
DC_drop_value = dropout
IC_drop_value =dropout
connected_drop_value = dropout
min_energy = 5
max_energy = args.emax
energy_factor = args.efactor

mask_zenith = args.mask_zenith
z_values = args.z_values

save = True
save_folder_name = "%soutput_plots/%s/"%(args.output_dir,filename)
if save==True:
    if os.path.isdir(save_folder_name) != True:
        os.mkdir(save_folder_name)
load_model_name = "%s%s_%iepochs_model.hdf5"%(save_folder_name,filename,args.epoch) 
use_old_weights = True

if args.first_variable == "Zenith" or args.first_variable == "zenith" or args.first_variable == "Z" or args.first_variable == "z":
    first_var = "zenith"
    first_var_index = 1
    print("Assuming Zenith is the only variable to test for")
    assert output_variables==1,"DOES NOT SUPPORT ZENITH FIRST + additional variables"
elif args.first_variable == "energy" or args.first_variable == "energy" or args.first_variable == "e" or args.first_variable == "E":
    first_var = "energy"
    first_var_index = 0
    print("testing with energy as the first index")
else:
    first_var = "energy"
    first_var_index = 0
    print("only supports energy and zenith right now! Please choose one of those. Defaulting to energy")
    print("testing with energy as the first index")

reco_name = args.test
save_folder_name += "/%s_%sepochs/"%(reco_name.replace(" ",""),epoch)
if os.path.isdir(save_folder_name) != True:
    os.mkdir(save_folder_name)
    
#Load in test data
print("Testing on %s"%test_file)
f = h5py.File(test_file, 'r')
Y_test_use = f['Y_test'][:]
X_test_DC_use = f['X_test_DC'][:]
X_test_IC_use = f['X_test_IC'][:]
if compare_reco:
    reco_test_use = f['reco_test'][:]
try:
    weights = f["weights_test"][:]
except:
    weights = None
    print("File does not have weights, not using...")
    pass
f.close
del f
print(X_test_DC_use.shape,X_test_IC_use.shape,Y_test_use.shape)

#mask_energy_train = np.logical_and(np.array(Y_test_use[:,0])>min_energy/max_energy,np.array(Y_test_use[:,0])<1.0)
#Y_test_use = np.array(Y_test_use)[mask_energy_train]
#X_test_DC_use = np.array(X_test_DC_use)[mask_energy_train]
#X_test_IC_use = np.array(X_test_IC_use)[mask_energy_train]
#if compare_reco:
#    reco_test_use = np.array(reco_test_use)[mask_energy_train]
if compare_reco:
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
    X_test_DC_use = X_test_DC_use[mask_zenith]
    X_test_IC_use = X_test_IC_use[mask_zenith]
    if compare_reco:
        reco_test_use = reco_test_use[mask_zenith]

#Make network and load model
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

from cnn_model import make_network
model_DC = make_network(X_test_DC_use,X_test_IC_use,output_variables,DC_drop_value,IC_drop_value,connected_drop_value)
model_DC.load_weights(load_model_name)
print("Loading model %s"%load_model_name)

# WRITE OWN LOSS FOR MORE THAN ONE REGRESSION OUTPUT
from keras.losses import mean_squared_error
from keras.losses import mean_absolute_percentage_error

if first_var == "zenith":
    def ZenithLoss(y_truth,y_predicted):
        #return logcosh(y_truth[:,1],y_predicted[:,1])
        return mean_squared_error(y_truth[:,1],y_predicted[:,0])

    def CustomLoss(y_truth,y_predicted):
            zenith_loss = ZenithLoss(y_truth,y_predicted)
            return zenith_loss

    model_DC.compile(loss=ZenithLoss,
                optimizer=Adam(lr=learning_rate),
                metrics=[ZenithLoss])
    
    print("zenith first")


else: 
    def EnergyLoss(y_truth,y_predicted):
        return mean_absolute_percentage_error(y_truth[:,0],y_predicted[:,0])

    def ZenithLoss(y_truth,y_predicted):
        return mean_squared_error(y_truth[:,1],y_predicted[:,1])

    def TrackLoss(y_truth,y_predicted):
        return mean_squared_logarithmic_error(y_truth[:,2],y_predicted[:,2])

    if output_variables == 3:
        def CustomLoss(y_truth,y_predicted):
            energy_loss = EnergyLoss(y_truth,y_predicted)
            zenith_loss = ZenithLoss(y_truth,y_predicted)
            track_loss = TrackLoss(y_truth,y_predicted)
            return energy_loss + zenith_loss + track_loss

        model_DC.compile(loss=CustomLoss,
                  optimizer=Adam(lr=learning_rate),
                  metrics=[EnergyLoss,ZenithLoss,TrackLoss])

    elif output_variables == 2:
        def CustomLoss(y_truth,y_predicted):
            energy_loss = EnergyLoss(y_truth,y_predicted)
            zenith_loss = ZenithLoss(y_truth,y_predicted)
            return energy_loss + zenith_loss

        model_DC.compile(loss=CustomLoss,
                  optimizer=Adam(lr=learning_rate),
                  metrics=[EnergyLoss,ZenithLoss])
    else:
        def CustomLoss(y_truth,y_predicted):
            energy_loss = EnergyLoss(y_truth,y_predicted)
            return energy_loss

        model_DC.compile(loss=EnergyLoss,
                    optimizer=Adam(lr=learning_rate),
                    metrics=[EnergyLoss])

# Run prediction
#Y_test_compare = Y_test_use[:,first_var_index]
#score = model_DC.evaluate([X_test_DC_use,X_test_IC_use], Y_test_compare, batch_size=256)
#print("Evaluate:",score)
t0 = time.time()
Y_test_predicted = model_DC.predict([X_test_DC_use,X_test_IC_use])
t1 = time.time()
print("This took me %f seconds for %i events"%(((t1-t0)),Y_test_predicted.shape[0]))
#print(X_test_DC_use.shape,X_test_IC_use.shape,Y_test_predicted.shape,Y_test_use.shape)

print("Saving output file: %s/prediction_values.hdf5"%save_folder_name)
f = h5py.File("%s/prediction_values.hdf5"%save_folder_name, "w")
f.create_dataset("Y_test_use", data=Y_test_use)
f.create_dataset("Y_predicted", data=Y_test_predicted)
if compare_reco:
    f.create_dataset("reco_test", data=reco_test_use)
if weights is not None:
    f.create_dataset("weights_test", data=weights)
f.close()

### MAKE THE PLOTS ###
from PlottingFunctions import plot_single_resolution
from PlottingFunctions import plot_2D_prediction
from PlottingFunctions import plot_2D_prediction_fraction
from PlottingFunctions import plot_bin_slices
from PlottingFunctions import plot_distributions
from PlottingFunctions import plot_length_energy

plots_names = ["Energy", "Cosine Zenith", "Track"]
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
    if first_var == "energy":
        true_index = num
        name_index = num
    if first_var == "zenith":
        true_index = first_var_index
        name_index = first_var_index
    plot_name = plots_names[name_index]
    plot_units = plots_units[name_index]
    maxabs_factor = maxabs_factors[name_index]
    maxval = maxvals[name_index]
    minval = minvals[name_index]
    use_frac = use_fractions[name_index]
    bins = bins_array[name_index]
    print("Plotting %s at position %i in true test output and %i in NN test output"%(plot_name, true_index,NN_index))
    
    plot_2D_prediction(Y_test_use[:,true_index]*maxabs_factor,\
                        Y_test_predicted[:,NN_index]*maxabs_factor,\
                        save,save_folder_name,bins=bins,\
                        minval=minval,maxval=maxval,cut_truth=True,axis_square=True,\
                        variable=plot_name,units=plot_units)
    plot_2D_prediction(Y_test_use[:,true_index]*maxabs_factor, Y_test_predicted[:,NN_index]*maxabs_factor,\
                        save,save_folder_name,bins=bins,\
                        minval=None,maxval=None,\
                        variable=plot_name,units=plot_units)
    plot_single_resolution(Y_test_use[:,true_index]*maxabs_factor,\
                   Y_test_predicted[:,NN_index]*maxabs_factor,\
                   minaxis=-maxval,maxaxis=maxval,bins=bins,
                   save=save,savefolder=save_folder_name,\
                   variable=plot_name,units=plot_units)
    plot_bin_slices(Y_test_use[:,true_index]*maxabs_factor, Y_test_predicted[:,NN_index]*maxabs_factor,\
                    use_fraction = False,\
                    bins=10,min_val=minval,max_val=maxval,\
                    save=True,savefolder=save_folder_name,\
                    variable=plot_name,units=plot_units)
   
    if compare_reco:
        plot_single_resolution(Y_test_use[:,true_index]*maxabs_factor,\
                   Y_test_predicted[:,NN_index]*maxabs_factor,\
                   use_old_reco = True, old_reco = reco_test_use[:,true_index],\
                   minaxis=-maxval,maxaxis=maxval,bins=bins,
                   save=save,savefolder=save_folder_name,\
                   variable=plot_name,units=plot_units, reco_name=reco_name)
        plot_single_resolution(Y_test_use[:,true_index]*maxabs_factor,\
                   Y_test_predicted[:,NN_index]*maxabs_factor,\
                   use_old_reco = True, old_reco = reco_test_use[:,true_index],\
                   use_fraction=True,bins=bins,maxaxis=2.,minaxis=-2.,\
                   save=save,savefolder=save_folder_name,\
                   variable=plot_name,units=plot_units)
        plot_bin_slices(Y_test_use[:,true_index]*maxabs_factor, Y_test_predicted[:,NN_index]*maxabs_factor,\
                    old_reco = reco_test_use[:,true_index],\
                    use_fraction = use_frac,\
                    bins=10,min_val=minval,max_val=maxval,\
                    save=True,savefolder=save_folder_name,\
                    variable=plot_name,units=plot_units, epochs = epoch,reco_name=reco_name)
    if first_var == "energy" and num ==0:
        plot_2D_prediction_fraction(Y_test_use[:,true_index]*maxabs_factor,\
                        Y_test_predicted[:,NN_index]*maxabs_factor,\
                        save,save_folder_name,bins=bins,axis_square=True,\
                        minval=-2.,maxval=2,\
                        variable=plot_name,units=plot_units)
        plot_bin_slices(Y_test_use[:,true_index]*maxabs_factor, Y_test_predicted[:,NN_index]*maxabs_factor,\
                    use_fraction = use_frac,\
                    bins=10,min_val=minval,max_val=maxval,\
                    save=True,savefolder=save_folder_name,\
                    variable=plot_name,units=plot_units)
        #plot_length_energy(Y_test_use, Y_test_predicted[:,NN_index]*maxabs_factor,\
        #                    use_fraction=True,ebins=20,tbins=20,\
        #                    emin=minvals[first_var_index], emax=maxvals[first_var_index],\
        #                    tmin=0.,tmax=450.,tfactor=maxabs_factors[2],\
        #                    savefolder=save_folder_name)
    if num > 0 or first_var == "zenith":
        plot_bin_slices(Y_test_use[:,true_index], Y_test_predicted[:,NN_index], \
                       energy_truth=Y_test_use[:,0]*max_energy, \
                       use_fraction = False, \
                       bins=10,min_val=min_energy,max_val=max_energy,\
                       save=True,savefolder=save_folder_name,\
                       variable=plot_name,units=plot_units, epochs=epoch)
        if compare_reco:
            plot_bin_slices(Y_test_use[:,true_index], Y_test_predicted[:,NN_index], \
                       energy_truth=Y_test_use[:,0]*max_energy, \
                       old_reco = reco_test_use[:,true_index],\
                       use_fraction = False, \
                       bins=10,min_val=min_energy,max_val=max_energy,\
                       save=True,savefolder=save_folder_name,\
                       variable=plot_name,units=plot_units, epochs = epoch,reco_name=reco_name)

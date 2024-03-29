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
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_file",type=str,default=None,
                    dest="input_file", help="names for test only input file")
parser.add_argument("-d", "--path",type=str,default='/data/icecube/jmicallef/processed_CNN_files/',
                    dest="path", help="path to input files")
parser.add_argument("-o", "--output_dir",type=str,default='/mnt/home/micall12/LowEnergyNeuralNetwork/output_plots/',
                    dest="output_dir", help="path to save output plots")
parser.add_argument("-n", "--name",type=str,default="prediction_values",
                    dest="name", help="name for output directory and where model file located")
parser.add_argument("-t","--test", default=None,
                        dest='test',help="name of subfoler for plots")
parser.add_argument("--model_dir", type=str,default="/mnt/home/micall12/LowEnergyNeuralNetwork/output_plots/",
                        dest='model_dir',help="name of reco")
parser.add_argument("--variable_list",nargs='+',default=[],
                    dest="variable_list", help="names of variables that were predicted: energy, zenith, class, muon, vertex, ending")
parser.add_argument("--epoch_list",nargs='+',default=[None,None,None,None,None],
                    dest="epoch_list", help="epochs to pull models from")
parser.add_argument("--modelname_list",nargs='+',default=[None,None,None,None,None],
                    dest="modelname_list", help="name output folder where model is stored, if NONE, assumes it is in the modeldir directly")
parser.add_argument("--factor_list",nargs='+',default=[100,1,1,1,1],
                    dest="factor_list", help="factor to multiply output by")
parser.add_argument("--small_network",default=False,action='store_true',
                    dest="small_network",help="Use smaller network model (cnn_model_simple.py)")
parser.add_argument("--dense_nodes", type=int,default=300,
                    dest="dense_nodes",help="Number of nodes in dense layer, only works for small network")
parser.add_argument("--conv_nodes", type=int,default=100,
                    dest="conv_nodes",help="Number of nodes in conv layers, only works for small network")
parser.add_argument("--save_inputs", default=False,action='store_true',
                    dest="save_inputs", help="saving input features of the cnn")
args = parser.parse_args()

test_file = args.path + args.input_file
filename = args.name
save_inputs = args.save_inputs

dropout = 0.2
DC_drop_value = dropout
IC_drop_value =dropout
connected_drop_value = dropout

small_network = args.small_network
dense_nodes = args.dense_nodes
conv_nodes = args.conv_nodes

variable_list = args.variable_list
epoch_list = args.epoch_list
modelname_list = args.modelname_list
factor_list = np.array(args.factor_list,dtype=float)

accepted_names = ["energy", "zenith", "class", "vertex", "muon", "error", "nDOM", "ending"]
for var in variable_list:
    assert var in accepted_names, "Variable must be one of the accepted names, check parse arg help for variable for more info"
if "error" in variable_list:
    assert len(variable_list) < 3, "Only supports error on single variable at the moment"

model_name_list = []
num_variables = len(variable_list)
for variable_index in range(num_variables):
    if epoch_list[variable_index] is None:
        model_name = args.model_dir + "/" + modelname_list[variable_index] + ".hdf5"
    else:
        model_name = "%s/%s/%s_%sepochs_model.hdf5"%(args.model_dir,modelname_list[variable_index], modelname_list[variable_index],epoch_list[variable_index])
    model_name_list.append(model_name)
    print("Predicting: %s,\nOutput transformation scale factor: %.2f.,\nUsing model: %s"%(variable_list[variable_index], factor_list[variable_index], model_name))

save = True
save_folder_name = "%s"%(args.output_dir)
if args.test is not None:
    save_folder_name += "/%s/"%args.test
if save==True:
    if os.path.isdir(save_folder_name) != True:
        os.mkdir(save_folder_name)

def cnn_test(features_DC, features_IC, load_model_name, output_variables=1,DC_drop=0.2,IC_drop=0.2,connected_drop=0.2,dense_nodes=300,conv_nodes=100,model_type="energy"):
    if small_network:
        from cnn_model_simple import make_network
        model_DC = make_network(features_DC,features_IC,1,DC_drop,IC_drop,connected_drop,conv_nodes=conv_nodes,dense_nodes=dense_nodes)
    else:
        if model_type == "class" or model_type == "muon":
            from cnn_model_classification import make_network
        elif model_type == "error":
            from cnn_model_losserror import make_network
        else:
            from cnn_model import make_network
        model_DC = make_network(features_DC,features_IC, output_variables, DC_drop, IC_drop,connected_drop)
    model_DC.load_weights(load_model_name)

    Y_test_predicted = model_DC.predict([features_DC,features_IC])

    return Y_test_predicted

#Load in test data
print("Testing on %s"%test_file)
f = h5py.File(test_file, 'r')
Y_test_use = f['Y_test'][:]
X_test_DC_use = f['X_test_DC'][:]
X_test_IC_use = f['X_test_IC'][:]
try:
    reco_test_use = f['reco_test'][:]
except:
    reco_test_use = None
try:
    weights = f["weights_test"][:]
except:
    weights = None
    print("File does not have weights, not using...")
    pass
f.close
del f
print(X_test_DC_use.shape,X_test_IC_use.shape,Y_test_use.shape)

total_variables = num_variables
if "vertex" in variable_list:
    total_variables += 2
if "ending" in variable_list:
    total_variables += 2
if "error" in variable_list:
    total_variables += 1
cnn_predictions=np.zeros((Y_test_use.shape[0],total_variables))
output_index = 0
for network in range(num_variables):
    factor = factor_list[network]
    if variable_list[network] == "vertex" or variable_list[network] == "ending":
        output_var = 3
    else:
        output_var = 1
    if "error" in variable_list[network]:
        output_var = output_var*2

    if variable_list[network] == "nDOM":
        t0 = time.time() 
        charge_DC = X_test_DC_use[:,:,:,0] > 0
        charge_IC = X_test_IC_use[:,:,:,0] > 0
        DC_flat = np.reshape(charge_DC,[X_test_DC_use.shape[0],480])
        IC_flat = np.reshape(charge_IC,[X_test_IC_use.shape[0],1140])
        DOMs_hit_DC = np.sum(DC_flat,axis=-1)
        DOMs_hit_IC = np.sum(IC_flat,axis=-1)
        DOMs_hit = DOMs_hit_DC + DOMs_hit_IC
        cnn_predictions[:,output_index] = DOMs_hit
        output_index += 1
        t1 = time.time()
        print("Time to calculate CNN %s on %i events: %f seconds"%(variable_list[network],X_test_DC_use.shape[0],t1-t0))
    else:
        t0 = time.time()
        cnn_predict = cnn_test(X_test_DC_use, X_test_IC_use, model_name_list[network],model_type=variable_list[network], output_variables=output_var,DC_drop=DC_drop_value,IC_drop=IC_drop_value,connected_drop=connected_drop_value,dense_nodes=dense_nodes,conv_nodes=conv_nodes)
        if factor is not None:
            factor = int(factor)
            if output_var == 1:
                cnn_predictions[:,output_index] = cnn_predict[:,0]*factor
                output_index += 1
            if output_var == 2: #case for error on ONE variable
                cnn_predictions[:,output_index] = cnn_predict[:,0]*factor
                output_index += 1
                cnn_predictions[:,output_index] = cnn_predict[:,1]
                output_index += 1
            if output_var == 3:
                for i in range(0,3):
                    cnn_predictions[:,output_index] = cnn_predict[:,i]*factor
                    output_index += 1
            #untransform truth
            if variable_list[network] == "energy":
                Y_test_use[:,0] = Y_test_use[:,0]*factor
        t1 = time.time()
        print("Time to run CNN Predict %s on %i events: %f seconds"%(variable_list[network],X_test_DC_use.shape[0],t1-t0))
    


print("Saving output file: %s/%s.hdf5"%(save_folder_name,filename))
f = h5py.File("%s/%s.hdf5"%(save_folder_name,filename), "w")
f.create_dataset("Y_test_use", data=Y_test_use)
f.create_dataset("Y_predicted", data=cnn_predictions)
if save_inputs:
    f.create_dataset("X_test_DC", data=X_test_DC_use)
    f.create_dataset("X_test_IC", data=X_test_IC_use)
if reco_test_use is not None:
    f.create_dataset("reco_test", data=reco_test_use)
if weights is not None:
    f.create_dataset("weights_test", data=weights)
f.close()

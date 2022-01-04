#########################
# Version of CNN on 8 Nov 2019
# 
# Runs net and plots
# Takes in multiple files to train on (large dataset)
# Runs Energy and Zenith only (must do both atm)
####################################

import numpy as np
import h5py
import time
import math
import os, sys
import argparse
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from handle_data import CutMask
from handle_data import VertexMask

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_files",type=str,default=None,
                    dest="input_files", help="name and path for input files")
parser.add_argument("-d", "--path",type=str,default='/mnt/research/IceCube/jmicallef/DNN_files/',
                    dest="path", help="path to input files")
parser.add_argument("-o", "--output_dir",type=str,default='/mnt/home/micall12/LowEnergyNeuralNetwork/',
                    dest="output_dir", help="path to output_plots directory, do not end in /")
parser.add_argument("-n", "--name",type=str,default='numu_flat_EZ_5_100_CC_cleaned',
                    dest="name", help="name for output directory and model file name")
parser.add_argument("-e","--epochs", type=int,default=30,
                    dest="epochs", help="number of epochs for neural net")
parser.add_argument("--start", type=int,default=0,
                    dest="start_epoch", help="epoch number to start at")
parser.add_argument("--variables", type=int,default=1,
                    dest="train_variables", help="1 for [energy], 2 for [energy, zenith], 3 for [energy, zenith, track]")
parser.add_argument("--model",default=None,
                    dest="model",help="Name of file with model weights to load--will start from here if file given")
parser.add_argument("--first_variable", type=str,default="energy",
                    dest="first_variable", help = "name for first variable (energy, zenith, or class are the only supported)")
parser.add_argument("--dropout", type=float,default=0.2,
                    dest="dropout",help="dropout value as float")
parser.add_argument("--lr", type=float,default=0.001,
                    dest="learning_rate",help="learning rate as a FLOAT")
parser.add_argument("--lr_drop", type=float,default=0.1,
                    dest="lr_drop",help="factor to drop learning rate by each lr_epoch")
parser.add_argument("--lr_epoch", type=int,default=None,
                    dest="lr_epoch",help="step size for number of epochs LR changes")
parser.add_argument("--error",default=False,action='store_true',
                    dest="do_error",help="Add flag if want to train for error (only single variable supported, not for classification)")
parser.add_argument("--cut_downgoing",default=False,action='store_true',
                    dest="cut_downgoing",help="Add flag if you want to only train on upgoing events (cosine zenith < 0.3)")
parser.add_argument("--logE",default=False,action='store_true',
                    dest="logE",help="Add flag if want to train for energy in log scale")
parser.add_argument("--small_network",default=False,action='store_true',
                    dest="small_network",help="Use smaller network model (cnn_model_simple.py)")
parser.add_argument("--layer_network",default=False,action='store_true',
                    dest="layer_network",help="Use network model with layer normalization instead of batch (cnn_model_layer.py)")
parser.add_argument("--instance_network",default=False,action='store_true',
                    dest="instance_network",help="Use network model with instance normalization instead of batch (cnn_model_layer.py)")
parser.add_argument("--dense_nodes", type=int,default=300,
                    dest="dense_nodes",help="Number of nodes in dense layer, only works for small network")
parser.add_argument("--conv_nodes", type=int,default=100,
                    dest="conv_nodes",help="Number of nodes in conv layers, only works for small network")
parser.add_argument("--chop_energy",default=False,action='store_true',
                    dest="chop_energy",help="Cut out low energy, specified by the ecut value")
parser.add_argument("--vertex_cut",default=False,action='store_true',
                    dest="vertex_cut",help="Add starting vertex cut")
parser.add_argument("--ecut", type=int,default=3,
                    dest="ecut",help="Energy cut value, in GeV")
parser.add_argument("--cut_nDOM",default=False,action='store_true',
                    dest="cut_nDOM",help="Cut based on theshold of nDOMs in CNN strings")
parser.add_argument("--nDOM", type=int,default=4,
                    dest="nDOM",help="Number of DOMs for threshold, will cut any number < value, so 4 means it will cut events with < 4 hit DOMs")
args = parser.parse_args()

# Settings from args
input_files = args.input_files
path = args.path
num_epochs = args.epochs
filename = args.name

epochs_step_drop = args.lr_epoch
if epochs_step_drop is None or epochs_step_drop==0:
    no_change = 0
    lr_drop = 1
    epochs_step_drop = 1
    # set up so pow(lr_drop,(epoch+1)/epochs_step_drop*nochange) = 1
    print("YOU DIDN'T GIVE A STEP SIZE TO DROP LR, WILL NOT CHANGE LR")
else:
    no_change = 1
    lr_drop = args.lr_drop

initial_lr = args.learning_rate

train_variables = args.train_variables
batch_size = 256
dropout = args.dropout
DC_drop_value = dropout
IC_drop_value =dropout
connected_drop_value = dropout

small_network = args.small_network
dense_nodes = args.dense_nodes
conv_nodes = args.conv_nodes
instance_network = args.instance_network
layer_network = args.layer_network
check_only_one_special_network = [small_network,instance_network,layer_network]
assert sum(check_only_one_special_network) <= 1, "too many special networks, choose between small, instance, OR layered"

start_epoch = args.start_epoch
do_error = args.do_error
cut_downgoing = args.cut_downgoing
logE = args.logE

chop_energy = args.chop_energy
ecut = args.ecut
vertex_cut = args.vertex_cut
cut_nDOM = args.cut_nDOM
threshold_nDOM = args.nDOM

old_model_given = args.model

if args.first_variable == "Zenith" or args.first_variable == "zenith" or args.first_variable == "Z" or args.first_variable == "z":
    first_var = "zenith"
    print("Assuming Zenith is the only variable to train for")
    assert train_variables==1,"DOES NOT SUPPORT ZENITH FIRST + additional variables"
elif args.first_variable == "energy" or args.first_variable == "Energy" or args.first_variable == "e" or args.first_variable == "E":
    first_var = "energy"
    print("training with energy as the first index")
elif args.first_variable == "class" or args.first_variable == "classification" or args.first_variable == "Class" or args.first_variable == "Classification" or args.first_variable == "C" or args.first_variable == "c" or args.first_variable == "PID" or args.first_variable == "pid":
    first_var = "class"
elif args.first_variable == "muon" or args.first_variable == "Muon":
    first_var = "muon"
else:
    first_var = None

if do_error:
    assert first_var != "class", "Cannot do error with current classificaiton setup"
    assert train_variables < 2, "Cannot do error with current classificaiton setup"
    print("Training for error")

assert first_var is not None, "Only supports energy and zenith and classification right now! Please choose one of those."

save = True
save_folder_name = "%s/output_plots/%s/"%(args.output_dir,filename)
if save==True:
    if os.path.isdir(save_folder_name) != True:
        os.mkdir(save_folder_name)
make_header_saveloss = False
if os.path.isfile("%ssaveloss_currentepoch.txt"%(save_folder_name)) != True:
    make_header_saveloss = True
    
        
use_old_reco = False

files_with_paths = path + input_files
file_names = sorted(glob.glob(files_with_paths))

print("\nFiles Used \nTraining %i files that look like %s \nStarting with model: %s \nSaving output to: %s"%(len(file_names),file_names[0],old_model_given,save_folder_name))

print("\nNetwork Parameters \nbatch_size: %i \ndropout: %f \nstarting learning rate: %f \ndense nodes: %i \nconv nodes: %i"%(batch_size,dropout,initial_lr,dense_nodes, conv_nodes))
if small_network:
    print("number conv layers: 5")

#print("Starting at epoch: %s \nTraining until: %s epochs \nTraining on %s variables \nUsing Network Config in %s"%(start_epoch,start_epoch+num_epochs,train_variables,network))
print("Starting at epoch: %s \nTraining until: %s epochs \nTraining on %s variables with %s first"%(start_epoch,start_epoch+num_epochs,train_variables, first_var))

if chop_energy:
    print("CUTTING OUT ENERGY ABOVE %.2f"%ecut)

afile = file_names[0]
f = h5py.File(afile, 'r')
X_train_DC = f['X_train_DC'][:]
X_train_IC = f['X_train_IC'][:]
f.close()
del f
print("Train Data DC", X_train_DC.shape)
print("Train Data IC", X_train_IC.shape)

# LOAD MODEL
if small_network:
    from cnn_model_simple import make_network
    model_DC = make_network(X_train_DC,X_train_IC,1,DC_drop_value,IC_drop_value,connected_drop_value,conv_nodes=conv_nodes,dense_nodes=dense_nodes)
elif layer_network:
    from cnn_model_layer import make_network
    model_DC = make_network(X_train_DC,X_train_IC,1,DC_drop_value,IC_drop_value,connected_drop_value)
elif instance_network:
    from cnn_model_instance import make_network
    model_DC = make_network(X_train_DC,X_train_IC,1,DC_drop_value,IC_drop_value,connected_drop_value)
else:
    if first_var == "class" or first_var == "muon":
        from cnn_model_classification import make_network
    else:
        if do_error:
            from cnn_model_losserror import make_network
        else:
            from cnn_model import make_network
    model_DC = make_network(X_train_DC,X_train_IC,train_variables,DC_drop_value,IC_drop_value,connected_drop_value)

# WRITE OWN LOSS FOR MORE THAN ONE REGRESSION OUTPUT
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.losses import mean_squared_error
from keras.losses import mean_squared_logarithmic_error
from keras.losses import logcosh
from keras.losses import mean_absolute_percentage_error
from keras.losses import BinaryCrossentropy

if first_var == "energy":
    def EnergyLoss(y_truth,y_predicted):
        return mean_absolute_percentage_error(y_truth[:,0],y_predicted[:,0])

    def ZenithLoss(y_truth,y_predicted):
        return mean_squared_error(y_truth[:,1],y_predicted[:,1])

    def ErrorLoss(y_truth,y_predicted):
        return mean_squared_error(tf.stop_gradient(tf.math.abs(y_truth[:,0]-y_predicted[:,0])),y_predicted[:,1])

if first_var == "zenith":
    def ZenithLoss(y_truth,y_predicted):
        return mean_squared_error(y_truth[:,1],y_predicted[:,0])

    def ErrorLoss(y_truth,y_predicted):
        return mean_squared_error(tf.stop_gradient(tf.math.abs(y_truth[:,1]-y_predicted[:,0])),y_predicted[:,1])

def TrackLoss(y_truth,y_predicted):
    return mean_squared_error(y_truth[:,2],y_predicted[:,2])

if train_variables == 3:
    def CustomLoss(y_truth,y_predicted):
        energy_loss = EnergyLoss(y_truth,y_predicted)
        zenith_loss = ZenithLoss(y_truth,y_predicted)
        track_loss = TrackLoss(y_truth,y_predicted)
        return energy_loss + zenith_loss + track_loss

elif train_variables == 2:
    def CustomLoss(y_truth,y_predicted):
        energy_loss = EnergyLoss(y_truth,y_predicted)
        zenith_loss = ZenithLoss(y_truth,y_predicted)
        return energy_loss + zenith_loss

else:
    if first_var == "energy":
        def CustomLoss(y_truth,y_predicted):
            energy_loss = EnergyLoss(y_truth,y_predicted)
            return energy_loss
    if first_var == "zenith":
        def CustomLoss(y_truth,y_predicted):
            zenith_loss = ZenithLoss(y_truth,y_predicted)
            return zenith_loss
#    if first_var == "class":
#        def CustomLoss(y_truth,y_predicted):
#            binary_class = ClassificationLoss(y_truth,y_predicted)
#            return binary_class

# Run neural network and record time ##
end_epoch = start_epoch + num_epochs
t0 = time.time()
for epoch in range(start_epoch,end_epoch):
   
    learning_rate = initial_lr * math.pow(lr_drop, math.floor((1+epoch)/epochs_step_drop)*no_change) 
    
        
    ## NEED TO HAVE OUTPUT DATA ALREADY TRANSFORMED!!! ##
    #print("True Epoch %i/%i"%(epoch+1,num_epochs))
    t0_loading = time.time()
    # Get new dataset
    input_file = file_names[epoch%len(file_names)]
    print("Now using file %s with lr %.1E"%(input_file,learning_rate))
    f = h5py.File(input_file, 'r')
    Y_train = f['Y_train'][:]
    X_train_DC = f['X_train_DC'][:]
    X_train_IC = f['X_train_IC'][:]
    X_validate_DC = f['X_validate_DC'][:]
    X_validate_IC = f['X_validate_IC'][:]
    Y_validate = f['Y_validate'][:]
    f.close()
    del f

    if cut_nDOM:
        charge_DC = X_train_DC[:,:,:,0] > 0
        charge_IC = X_train_IC[:,:,:,0] > 0
        DC_flat = np.reshape(charge_DC,[X_train_DC.shape[0],480])
        IC_flat = np.reshape(charge_IC,[X_train_IC.shape[0],1140])
        DOMs_hit_DC = np.sum(DC_flat,axis=-1)
        DOMs_hit_IC = np.sum(IC_flat,axis=-1)
        DOMs_hit = DOMs_hit_DC + DOMs_hit_IC
        check_min_nDOM = DOMs_hit >= threshold_nDOM
        Y_train = Y_train[check_min_nDOM]
        X_train_DC = X_train_DC[check_min_nDOM]
        X_train_IC = X_train_IC[check_min_nDOM]
        print("Cut %i training events due to nDOM >= %i"%(len(check_min_nDOM)-sum(check_min_nDOM),threshold_nDOM))
        
        charge_DC = X_validate_DC[:,:,:,0] > 0
        charge_IC = X_validate_IC[:,:,:,0] > 0
        DC_flat = np.reshape(charge_DC,[X_validate_DC.shape[0],480])
        IC_flat = np.reshape(charge_IC,[X_validate_IC.shape[0],1140])
        DOMs_hit_DC = np.sum(DC_flat,axis=-1)
        DOMs_hit_IC = np.sum(IC_flat,axis=-1)
        DOMs_hit = DOMs_hit_DC + DOMs_hit_IC
        check_min_nDOM = DOMs_hit >= threshold_nDOM
        Y_validate = Y_validate[check_min_nDOM]
        X_validate_DC = X_validate_DC[check_min_nDOM]
        X_validate_IC = X_validate_IC[check_min_nDOM]
        print("Cut %i validation events due to nDOM >= %i"%(len(check_min_nDOM)-sum(check_min_nDOM),threshold_nDOM))
  
    if cut_downgoing:
        print("Cutting downgoing events, only keeping cosine zenith < 0.3")
        check_coszenith = np.logical_or(Y_train[:,1] > 1, Y_train[:,1] < -1)
        assert sum(check_coszenith)==0, "check truth label for train, not in cosine potentially (numbers > 1 or < -1"
        check_coszenith = np.logical_or(Y_validate[:,1] > 1, Y_validate[:,1] < -1)
        assert sum(check_coszenith)==0, "check truth label for validate, not in cosine potentially (numbers > 1 or < -1"
        mask_train = Y_train[:,1] < 0.3
        mask_validate = Y_validate[:,1] < 0.3
        Y_train = Y_train[mask_train]
        X_train_DC = X_train_DC[mask_train]
        X_train_IC = X_train_IC[mask_train]
        Y_validate = Y_validate[mask_validate]
        X_validate_DC = X_validate_DC[mask_validate]
        X_validate_IC = X_validate_IC[mask_validate]

    if logE:
        print("Training and validating on LOG E")
        Y_train[:,0] = np.log10(Y_train[:,0])
        Y_validate[:,0] = np.log10(Y_validate[:,0])
    
    if chop_energy:
        cut_train = Y_train[:,0] < ecut/100.
        cut_validate = Y_validate[:,0] < ecut/100.
        Y_train = Y_train[cut_train]
        X_train_DC = X_train_DC[cut_train]
        X_train_IC = X_train_IC[cut_train]
        Y_validate = Y_validate[cut_validate]
        X_validate_DC = X_validate_DC[cut_validate]
        X_validate_IC = X_validate_IC[cut_validate]

    if vertex_cut:
        vertex_mask_train = VertexMask(Y_train,azimuth_index=7,track_index=2,max_track=200.)
        cut_train = vertex_mask_train["start_IC19"]
        vertex_mask_val = VertexMask(Y_validate,azimuth_index=7,track_index=2,max_track=200.)
        cut_validate = vertex_mask_val["start_IC19"]
        print("Removing %i events from train, %i events from validate"%((len(cut_train)-sum(cut_train)),(len(cut_validate)-sum(cut_validate))))
        Y_train = Y_train[cut_train]
        X_train_DC = X_train_DC[cut_train]
        X_train_IC = X_train_IC[cut_train]
        Y_validate = Y_validate[cut_validate]
        X_validate_DC = X_validate_DC[cut_validate]
        X_validate_IC = X_validate_IC[cut_validate]
    
    #Set up binary crosentropy calculation for classificaiton
    if first_var == "muon":
        muon_mask_train = (Y_train[:,9]) == 13
        Y_train = np.array(muon_mask_train,dtype=int)
        muon_mask_val = (Y_validate[:,9]) == 13
        Y_validate = np.array(muon_mask_val,dtype=int)
    if first_var == "class":
        Y_train = np.array(Y_train[:,8],dtype=int)
        Y_validate = np.array(Y_validate[:,8],dtype=int)

    # Compile model
    if train_variables == 1:
        if first_var == "energy":
            if do_error:
                model_DC.compile(loss=CustomLoss,
                    optimizer=Adam(lr=learning_rate),
                    metrics=[EnergyLoss,ErrorLoss])
            else:
                model_DC.compile(loss=CustomLoss,
                    optimizer=Adam(lr=learning_rate),
                    metrics=[EnergyLoss])
        if first_var == "zenith":
            if do_error:
                model_DC.compile(loss=CustomLoss,
                    optimizer=Adam(lr=learning_rate),
                    metrics=[ZenithLoss,ErrorLoss])
            else:
                model_DC.compile(loss=CustomLoss,
                    optimizer=Adam(lr=learning_rate),
                    metrics=[ZenithLoss])
        if first_var == "class" or first_var == "muon":
            model_DC.compile(loss=BinaryCrossentropy(),
            optimizer=Adam(lr=learning_rate))
    elif train_variables == 2:
        model_DC.compile(loss=CustomLoss,
              optimizer=Adam(lr=learning_rate),
              metrics=[EnergyLoss,ZenithLoss])
    elif train_variables == 3:
        model_DC.compile(loss=CustomLoss,
              optimizer=Adam(lr=learning_rate),
              metrics=[EnergyLoss,ZenithLoss,TrackLoss])
    else:
        print("Only supports 1, 2, or 3 labels (energy, zenith, track). Not compiling. This will fail")
    
	# Use old weights
    if epoch > 0 and not old_model_given:
        last_model = '%scurrent_model_while_running.hdf5'%save_folder_name
        model_DC.load_weights(last_model)
    elif old_model_given:
        print("Using given model %s"%old_model_given)
        model_DC.load_weights(old_model_given)
        old_model_given = None
    else:
        print("Training set: %i, Validation set: %i"%(len(Y_train),len(Y_validate)))
        print(epoch,end_epoch)
    
    #Run one epoch with dataset
    t0_epoch = time.time()
    network_history = model_DC.fit([X_train_DC, X_train_IC], Y_train,
                            validation_data= ([X_validate_DC, X_validate_IC], Y_validate),
                            batch_size=batch_size,
                            initial_epoch= epoch,
                            epochs=epoch+1, #goes from intial to epochs, so need it to be greater than initial
                            callbacks = [ModelCheckpoint('%scurrent_model_while_running.hdf5'%save_folder_name)],
                            verbose=1)
    t1_epoch = time.time()
    t1_loading = time.time()
    dt_epoch = (t1_epoch - t0_epoch)/60.
    dt_loading = (t1_loading - t0_loading)/60.
    
    #Set up file that saves losses once
    if make_header_saveloss and epoch==0:
        afile = open("%ssaveloss_currentepoch.txt"%(save_folder_name),"a")
        afile.write("Epoch" + '\t' + "Time Epoch" + '\t' + "Time Train" + '\t')
        for key in network_history.history.keys():
            afile.write(str(key) + '\t')
        afile.write('\n')
        afile.close()

    # Save loss
    afile = open("%ssaveloss_currentepoch.txt"%(save_folder_name),"a")
    afile.write(str(epoch+1) + '\t' + str(dt_loading) + '\t' + str(dt_epoch) + '\t')
    for key in network_history.history.keys():
        afile.write(str(network_history.history[key][0]) + '\t')
    afile.write('\n')    
    afile.close()

    print(epoch,len(file_names),epoch%len(file_names),len(file_names)-1)
    if epoch%len(file_names) == (len(file_names)-1) or (epoch%5==0):
        model_save_name = "%s%s_%iepochs_model.hdf5"%(save_folder_name,filename,epoch+1)
        model_DC.save(model_save_name)
        print("Saved model to %s"%model_save_name)
    
    
t1 = time.time()
print("This took me %f minutes"%((t1-t0)/60.))

model_DC.save("%s%s_model_final.hdf5"%(save_folder_name,filename))



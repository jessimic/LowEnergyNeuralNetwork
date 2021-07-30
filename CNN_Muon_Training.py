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

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_file",type=str,default=None,
                    dest="input_file", help="name for ONE AND ONLY ONE input file")
parser.add_argument("-d", "--path",type=str,default='/mnt/research/IceCube/jmicallef/DNN_files/',
                    dest="path", help="path to input file")
parser.add_argument("-o", "--output_dir",type=str,default='/mnt/home/micall12/LowEnergyNeuralNetwork/',
                    dest="output_dir", help="path to output_plots directory, do not end in /")
parser.add_argument("-n", "--name",type=str,default='numu_flat_EZ_5_100_CC_cleaned',
                    dest="name", help="name for output directory and model file name")
parser.add_argument("-e","--epochs", type=int,default=30,
                    dest="epochs", help="number of epochs for neural net")
parser.add_argument("--start", type=int,default=0,
                    dest="start_epoch", help="epoch number to start at")
parser.add_argument("--model",default=None,
                    dest="model",help="Name of file with model weights to load--will start from here if file given")
parser.add_argument("--lr", type=float,default=0.001,
                    dest="learning_rate",help="learning rate as a FLOAT")
parser.add_argument("--lr_drop", type=float,default=0.1,
                    dest="lr_drop",help="factor to drop learning rate by each lr_epoch")
parser.add_argument("--lr_epoch", type=int,default=None,
                    dest="lr_epoch",help="step size for number of epochs LR changes")
parser.add_argument("--save_every", type=int,default=5,
                    dest="save_every",help="step size for number how many epochs to save at")
parser.add_argument("--cut_downgoing",default=False,action='store_true',
                    dest="cut_downgoing",help="Add flag if you want to only train on upgoing events (cosine zenith < 0.3)")
args = parser.parse_args()

# Settings from args
input_file = args.input_file
path = args.path
num_epochs = args.epochs
filename = args.name
save_every = args.save_every

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

batch_size = 256
dropout = 0.2
DC_drop_value = dropout
IC_drop_value =dropout
connected_drop_value = dropout
start_epoch = args.start_epoch
cut_downgoing = args.cut_downgoing

old_model_given = args.model

save = True
save_folder_name = "%s/output_plots/%s/"%(args.output_dir,filename)
if save==True:
    if os.path.isdir(save_folder_name) != True:
        os.mkdir(save_folder_name)
make_header_saveloss = False
if os.path.isfile("%ssaveloss_currentepoch.txt"%(save_folder_name)) != True:
    make_header_saveloss = True
    
file_name = path + input_file

print("\nFiles Used \nTraining on file %s \nStarting with model: %s \nSaving output to: %s"%(file_name,old_model_given,save_folder_name))
print("\nNetwork Parameters \nbatch_size: %i \ndropout: %f \nstarting learning rate: %f"%(batch_size,dropout,initial_lr))
print("Starting at epoch: %s \nTraining until: %s epochs"%(start_epoch,start_epoch+num_epochs))

t0 = time.time()
f = h5py.File(file_name, 'r')
Y_train_file = f['Y_train'][:]
X_train_DC = f['X_train_DC'][:]
X_train_IC = f['X_train_IC'][:]
X_validate_DC = f['X_validate_DC'][:]
X_validate_IC = f['X_validate_IC'][:]
Y_validate_file = f['Y_validate'][:]
f.close()
del f
print("Train Data DC", X_train_DC.shape)
print("Train Data IC", X_train_IC.shape)
if cut_downgoing:
    print("Cutting downgoing events, only keeping cosine zenith < 0.3")
    check_coszenith = np.logical_or(Y_train_file[:,1] > 1, Y_train_file[:,1] < -1)
    assert sum(check_coszenith)==0, "check truth label for train, not in cosine potentially (numbers > 1 or < -1"
    check_coszenith = np.logical_or(Y_validate_file[:,1] > 1, Y_validate_file[:,1] < -1)
    assert sum(check_coszenith)==0, "check truth label for validate, not in cosine potentially (numbers > 1 or < -1"
    mask_train = Y_train_file[:,1] < 0.3
    mask_validate = Y_validate_file[:,1] < 0.3
    Y_train = Y_train_file[mask_train]
    X_train_DC = X_train_DC[mask_train]
    X_train_IC = X_train_IC[mask_train]
    Y_validate = Y_validate_file[mask_validate]
    X_validate_DC = X_validate_DC[mask_validate]
    X_validate_IC = X_validate_IC[mask_validate]
    print("Now Train Data DC", X_train_DC.shape)
    print("Now Train Data IC", X_train_IC.shape)

muon_mask_train = (Y_train_file[:,9]) == 13
Y_train = np.array(muon_mask_train,dtype=int)
print(Y_train_file[:10,9])
print(Y_train[:10])
muon_mask_val = (Y_validate_file[:,9]) == 13
Y_validate = np.array(muon_mask_val,dtype=int)
print(Y_validate_file[:10,9])
print(Y_validate[:10])

# LOAD MODEL
from cnn_model_classification import make_network
model_DC = make_network(X_train_DC,X_train_IC,1,DC_drop_value,IC_drop_value,connected_drop_value)

# WRITE OWN LOSS FOR MORE THAN ONE REGRESSION OUTPUT
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.losses import BinaryCrossentropy
from keras.losses import SparseCategoricalCrossentropy

# Run neural network and record time ##
end_epoch = start_epoch + num_epochs
for epoch in range(start_epoch,end_epoch):
   
    learning_rate = initial_lr * math.pow(lr_drop, math.floor((1+epoch)/epochs_step_drop)*no_change) 
    print("Now using file %s with lr %.1E"%(input_file,learning_rate))
    
    model_DC.compile(loss=BinaryCrossentropy(),
    optimizer=Adam(lr=learning_rate))
    
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
    dt_epoch = (t1_epoch - t0_epoch)/60.
    
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
    afile.write(str(epoch+1) + '\t' + str(dt_epoch) + '\t' + str(dt_epoch) + '\t')
    for key in network_history.history.keys():
        afile.write(str(network_history.history[key][0]) + '\t')
    afile.write('\n')    
    afile.close()

    print(epoch,save_every,epoch%save_every,save_every-1)
    if epoch%save_every == (save_every-1):
        model_save_name = "%s%s_%iepochs_model.hdf5"%(save_folder_name,filename,epoch+1)
        model_DC.save(model_save_name)
        print("Saved model to %s"%model_save_name)
    
    
t1 = time.time()
print("This took me %f minutes"%((t1-t0)/60.))

model_DC.save("%s%s_model_final.hdf5"%(save_folder_name,filename))



#########################
# Version of CNN on 8 Nov 2019
# 
# Runs net and plots
# Takes in multiple files to train on (large dataset)
# Runs Energy and Zenith only (must do both atm)
####################################

import numpy
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
                    dest="first_variable", help = "name for first variable (energy, zenith only two supported)")
parser.add_argument("--lr", type=float,default=0.001,
                    dest="learning_rate",help="learning rate as a FLOAT")
parser.add_argument("--lr_drop", type=float,default=0.1,
                    dest="lr_drop",help="factor to drop learning rate by each lr_epoch")
parser.add_argument("--lr_epoch", type=int,default=None,
                    dest="lr_epoch",help="step size for number of epochs LR changes")

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
dropout = 0.2
DC_drop_value = dropout
IC_drop_value =dropout
connected_drop_value = dropout
start_epoch = args.start_epoch

old_model_given = args.model

if args.first_variable == "Zenith" or args.first_variable == "zenith" or args.first_variable == "Z" or args.first_variable == "z":
    first_var = "zenith"
    print("Assuming Zenith is the only variable to train for")
    assert train_variables==1,"DOES NOT SUPPORT ZENITH FIRST + additional variables"
elif args.first_variable == "energy" or args.first_variable == "Energy" or args.first_variable == "e" or args.first_variable == "E":
    first_var = "energy"
    print("training with energy as the first index")
elif args.first_variable == "class" or args.first_variable == "classification" or args.first_variable == "Class" or args.first_variable == "Classification" or args.first_variable == "C" or args.first_variable == "c":
    first_var = "class"
else:
    first_var = None

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

print("\nNetwork Parameters \nbatch_size: %i \ndropout: %f \nstarting learning rate: %f"%(batch_size,dropout,initial_lr))

#print("Starting at epoch: %s \nTraining until: %s epochs \nTraining on %s variables \nUsing Network Config in %s"%(start_epoch,start_epoch+num_epochs,train_variables,network))
print("Starting at epoch: %s \nTraining until: %s epochs \nTraining on %s variables with %s first"%(start_epoch,start_epoch+num_epochs,train_variables, first_var))

afile = file_names[0]
f = h5py.File(afile, 'r')
X_train_DC = f['X_train_DC'][:]
X_train_IC = f['X_train_IC'][:]
f.close()
del f
print("Train Data DC", X_train_DC.shape)
print("Train Data IC", X_train_IC.shape)

# LOAD MODEL
if first_var == "class":
    from cnn_classification_model import make_network
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
from keras.losses import SparseCategoricalCrossentropy

if first_var == "energy":
    def EnergyLoss(y_truth,y_predicted):
        return mean_absolute_percentage_error(y_truth[:,0],y_predicted[:,0])

    def ZenithLoss(y_truth,y_predicted):
        return mean_squared_error(y_truth[:,1],y_predicted[:,1])

if first_var == "zenith":
    def ZenithLoss(y_truth,y_predicted):
        return mean_squared_error(y_truth[:,1],y_predicted[:,0])

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
  

    # Compile model
    if train_variables == 1:
        if first_var == "energy":
            model_DC.compile(loss=CustomLoss,
                optimizer=Adam(lr=learning_rate),
                metrics=[EnergyLoss])
        if first_var == "zenith":
            model_DC.compile(loss=CustomLoss,
                optimizer=Adam(lr=learning_rate),
                metrics=[ZenithLoss])
        if first_var == "class":
            Y_train = numpy.array(Y_train[:,8],dtype=int)
            Y_validate = numpy.array(Y_validate[:,8],dtype=int)
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
    if epoch%len(file_names) == (len(file_names)-1):
        model_save_name = "%s%s_%iepochs_model.hdf5"%(save_folder_name,filename,epoch+1)
        model_DC.save(model_save_name)
        print("Saved model to %s"%model_save_name)
    
    
t1 = time.time()
print("This took me %f minutes"%((t1-t0)/60.))

model_DC.save("%s%s_model_final.hdf5"%(save_folder_name,filename))



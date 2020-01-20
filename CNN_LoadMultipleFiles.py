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
import os, sys
import argparse
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_files",type=str,default=None,
                    dest="input_files", help="name and path for input files")
parser.add_argument("-d", "--path",type=str,default='/mnt/research/IceCube/jmicallef/DNN_files/',
                    dest="path", help="path to input files")
parser.add_argument("-o", "--output_dir",type=str,default='/mnt/home/micall12/DNN_LE/',
                    dest="output_dir", help="path to output_plots directory, do not end in /")
parser.add_argument("-n", "--name",type=str,default='numu_flat_EZ_5_100_CC_cleaned',
                    dest="name", help="name for output directory and model file name")
parser.add_argument("-e","--epochs", type=int,default=30,
                    dest="epochs", help="number of epochs for neural net")
parser.add_argument("--start", type=int,default=0,
                    dest="start_epoch", help="epoch number to start at")
parser.add_argument("--variables", type=int,default=2,
                    dest="train_variables", help="1 for [energy], 2 for [energy, zenith], 3 for [energy, zenith, track]")
parser.add_argument("--model",default=None,
                    dest="model",help="Name of file with model weights to load--will start from here if file given")
args = parser.parse_args()

# Settings from args
input_files = args.input_files
path = args.path
num_epochs = args.epochs
filename = args.name

train_variables = args.train_variables
num_labels = train_variables
batch_size = 256
dropout = 0.2
learning_rate = 1e-3
DC_drop_value = dropout
IC_drop_value =dropout
connected_drop_value = dropout
min_energy = 5.
max_energy = 100.
start_epoch = args.start_epoch

old_model_given = args.model

save = True
save_folder_name = "%s/output_plots/%s/"%(args.output_dir,filename)
if save==True:
    if os.path.isdir(save_folder_name) != True:
        os.mkdir(save_folder_name)
        
use_old_reco = False

files_with_paths = path + input_files
file_names = sorted(glob.glob(files_with_paths))

print("\nFiles Used \nTraining %i files that look like %s \nStarting with model: %s \nSaving output to: %s"%(len(file_names),file_names[0],old_model_given,save_folder_name))

print("\nNetwork Parameters \nbatch_size: %i \ndropout: %f \nlearning rate: %f \nenergy range for plotting: %f - %f"%(batch_size,dropout,learning_rate,min_energy,max_energy))

print("Starting at epoch: %s \nTraining until: %s epochs \nTraining on %s variables"%(start_epoch,start_epoch+num_epochs,train_variables))


if train_variables == 2:
    print("\nASSUMING ORDER OF OUTPUT VARIABLES ARE ENERGY[0] and COS ZENITH[1] \n") 
if train_variables == 3:
    print("\nASSUMING ORDER OF OUTPUT VARIABLES ARE ENERGY[0], COS ZENITH[1], TRACK[2]\n") 

afile = file_names[0]
f = h5py.File(afile, 'r')
X_train_DC = f['X_train_DC'][:]
X_train_IC = f['X_train_IC'][:]
f.close()
del f
print("Train Data DC", X_train_DC.shape)
print("Train Data IC", X_train_IC.shape)



### Build The Network ##
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Lambda
from keras.layers import concatenate
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import initializers
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint


## CNN NETWORK ##

# DEEP CORE #
#print("Train Data DC", X_train_DC.shape)
strings = X_train_DC.shape[1]
dom_per_string = X_train_DC.shape[2]
dom_variables = X_train_DC.shape[3]

# Conv DC + batch normalization, later dropout and maxpooling
input_DC = Input(shape=(strings, dom_per_string, dom_variables))

conv1_DC = Conv2D(100,kernel_size=(strings,5),padding='same',activation='tanh')(input_DC)
batch1_DC = BatchNormalization()(conv1_DC)
pool1_DC = MaxPooling2D(pool_size=(1,2))(batch1_DC)
drop1_DC = Dropout(DC_drop_value)(pool1_DC)

conv2_DC = Conv2D(100,kernel_size=(strings,7),padding='same',activation='relu')(drop1_DC)
batch2_DC = BatchNormalization()(conv2_DC)
drop2_DC = Dropout(DC_drop_value)(batch2_DC)

conv3_DC = Conv2D(100,kernel_size=(strings,7),padding='same',activation='relu')(drop2_DC)
batch3_DC = BatchNormalization()(conv3_DC)
drop3_DC = Dropout(DC_drop_value)(batch3_DC)

conv4_DC = Conv2D(100,kernel_size=(strings,3),padding='valid',activation='relu')(drop3_DC)
batch4_DC = BatchNormalization()(conv4_DC)
pool4_DC = MaxPooling2D(pool_size=(1,2))(batch4_DC)
drop4_DC = Dropout(DC_drop_value)(pool4_DC)

conv5_DC = Conv2D(100,kernel_size=(1,7),padding='same',activation='relu')(drop4_DC)
batch5_DC = BatchNormalization()(conv5_DC)
drop5_DC = Dropout(DC_drop_value)(batch5_DC)

conv6_DC = Conv2D(100,kernel_size=(1,7),padding='same',activation='relu')(drop5_DC)
batch6_DC = BatchNormalization()(conv6_DC)
drop6_DC = Dropout(DC_drop_value)(batch6_DC)

conv7_DC = Conv2D(100,kernel_size=(1,1),padding='same',activation='relu')(drop6_DC)
batch7_DC = BatchNormalization()(conv7_DC)
drop7_DC = Dropout(DC_drop_value)(batch7_DC)

conv8_DC = Conv2D(100,kernel_size=(1,1),padding='same',activation='relu')(drop7_DC)
batch8_DC = BatchNormalization()(conv8_DC)
drop8_DC = Dropout(DC_drop_value)(batch8_DC)

flat_DC = Flatten()(drop8_DC)


# ICECUBE NEAR DEEPCORE #
#print("Train Data IC", X_train_IC.shape)
strings_IC = X_train_IC.shape[1]
dom_per_string_IC = X_train_IC.shape[2]
dom_variables_IC = X_train_IC.shape[3]

# Conv DC + batch normalization, later dropout and maxpooling
input_IC = Input(shape=(strings_IC, dom_per_string_IC, dom_variables_IC))

conv1_IC = Conv2D(100,kernel_size=(strings_IC,5),padding='same',activation='tanh')(input_IC)
batch1_IC = BatchNormalization()(conv1_IC)
pool1_IC = MaxPooling2D(pool_size=(1,2))(batch1_IC)
drop1_IC = Dropout(IC_drop_value)(pool1_IC)

conv2_IC = Conv2D(100,kernel_size=(strings_IC,7),padding='same',activation='relu')(drop1_IC)
batch2_IC = BatchNormalization()(conv2_IC)
drop2_IC = Dropout(IC_drop_value)(batch2_IC)

conv3_IC = Conv2D(100,kernel_size=(strings_IC,7),padding='same',activation='relu')(drop2_IC)
batch3_IC = BatchNormalization()(conv3_IC)
drop3_IC = Dropout(IC_drop_value)(batch3_IC)

conv4_IC = Conv2D(100,kernel_size=(strings_IC,3),padding='valid',activation='relu')(drop3_IC)
batch4_IC = BatchNormalization()(conv4_IC)
pool4_IC = MaxPooling2D(pool_size=(1,2))(batch4_IC)
drop4_IC = Dropout(IC_drop_value)(pool4_IC)

conv5_IC = Conv2D(100,kernel_size=(1,7),padding='same',activation='relu')(drop4_IC)
batch5_IC = BatchNormalization()(conv5_IC)
drop5_IC = Dropout(IC_drop_value)(batch5_IC)

conv6_IC = Conv2D(100,kernel_size=(1,7),padding='same',activation='relu')(drop5_IC)
batch6_IC = BatchNormalization()(conv6_IC)
drop6_IC = Dropout(IC_drop_value)(batch6_IC)

conv7_IC = Conv2D(100,kernel_size=(1,1),padding='same',activation='relu')(drop6_IC)
batch7_IC = BatchNormalization()(conv7_IC)
drop7_IC = Dropout(IC_drop_value)(batch7_IC)

conv8_IC = Conv2D(100,kernel_size=(1,1),padding='same',activation='relu')(drop7_IC)
batch8_IC = BatchNormalization()(conv8_IC)
drop8_IC = Dropout(IC_drop_value)(batch8_IC)

flat_IC = Flatten()(drop8_IC)


# PUT TOGETHER #
concatted = concatenate([flat_DC, flat_IC])

full1 = Dense(300,activation='relu')(concatted)
batch1_full = BatchNormalization()(full1)
dropf = Dropout(connected_drop_value)(batch1_full)

output = Dense(num_labels,activation='linear')(dropf)
model_DC = Model(inputs=[input_DC,input_IC],outputs=output)

####################


# WRITE OWN LOSS FOR MORE THAN ONE REGRESSION OUTPUT
from keras.losses import mean_squared_error
from keras.losses import mean_squared_logarithmic_error
from keras.losses import logcosh

def EnergyLoss(y_truth,y_predicted):
    #return mean_squared_logarithmic_error(y_truth[:,0],y_predicted[:,0]) #/120.
    return mean_squared_error(y_truth[:,0],y_predicted[:,0])

def ZenithLoss(y_truth,y_predicted):
    #return logcosh(y_truth[:,1],y_predicted[:,1])
    return mean_squared_error(y_truth[:,1],y_predicted[:,1])

def TrackLoss(y_truth,y_predicted):
    return mean_squared_error(y_truth[:,2],y_predicted[:,2])

if train_variables == 3:
    def CustomLoss(y_truth,y_predicted):
        energy_loss = EnergyLoss(y_truth,y_predicted)
        zenith_loss = ZenithLoss(y_truth,y_predicted)
        track_loss = TrackLoss(y_truth,y_predicted)
        return energy_loss + zenith_loss + track_loss

if train_variables == 2:
    def CustomLoss(y_truth,y_predicted):
        energy_loss = EnergyLoss(y_truth,y_predicted)
        zenith_loss = ZenithLoss(y_truth,y_predicted)
        return energy_loss + zenith_loss

if train_variables == 1:
    def CustomLoss(y_truth,y_predicted):
        energy_loss = EnergyLoss(y_truth,y_predicted)
        return energy_loss


# Run neural network and record time ##
loss = []
val_loss = []
energy_loss = []
zenith_loss = []
track_loss = []
val_energy_loss = []
val_zenith_loss = []
val_track_loss = []

end_epoch = start_epoch + num_epochs
current_epoch = start_epoch
t0 = time.time()
for epoch in range(start_epoch,end_epoch):
    
    ## NEED TO HAVE OUTPUT DATA ALREADY TRANSFORMED!!! ##
    #print("True Epoch %i/%i"%(epoch+1,num_epochs))
    
    # Get new dataset
    input_file = file_names[epoch%len(file_names)]
    print("Now using file %s"%input_file)
    f = h5py.File(input_file, 'r')
    Y_train = f['Y_train'][:]
    X_train_DC = f['X_train_DC'][:]
    X_train_IC = f['X_train_IC'][:]
    X_validate_DC = f['X_validate_DC'][:]
    X_validate_IC = f['X_validate_IC'][:]
    Y_validate = f['Y_validate'][:]
    f.close()
    del f
    
    Y_train_use = Y_train[:,:train_variables]
    Y_val_use = Y_validate[:,:train_variables]

    # Use old weights
    if epoch > 0 and not old_model_given:
        last_model = '%scurrent_model_while_running.hdf5'%save_folder_name
        model_DC.load_weights(last_model)
    elif old_model_given:
        print("Using given model %s"%old_model_given)
        model_DC.load_weights(old_model_given)
        old_model_given = None
    else:
        print("Training set: %i, Validation set: %i"%(len(Y_train_use),len(Y_val_use)))
        print(current_epoch,end_epoch)
    
    # Compile model
    if num_labels == 1:
        model_DC.compile(loss=CustomLoss,
              optimizer=Adam(lr=learning_rate),
              metrics=[EnergyLoss])
        losses_names = ['loss', 'val_loss','energy_loss', 'val_energy_loss']
    elif num_labels == 2:
        model_DC.compile(loss=CustomLoss,
              optimizer=Adam(lr=learning_rate),
              metrics=[EnergyLoss,ZenithLoss])
        losses_names = ['loss', 'val_loss','energy_loss', 'val_energy_loss', 'zenith_loss', 'val_zenith_loss']
    elif num_labels == 3:
        model_DC.compile(loss=CustomLoss,
              optimizer=Adam(lr=learning_rate),
              metrics=[EnergyLoss,ZenithLoss,TrackLoss])
        losses_names = ['loss', 'val_loss','energy_loss', 'val_energy_loss', 'zenith_loss', 'val_zenith_loss', 'track_loss', 'val_track_loss']
    else:
        print("Only supports 1, 2, or 3 labels (energy, zenith, track). Not compiling. This will fail")
    
    #Run one epoch with dataset
    network_history = model_DC.fit([X_train_DC, X_train_IC], Y_train_use,
                            validation_data= ([X_validate_DC, X_validate_IC], Y_val_use),
                            batch_size=batch_size,
                            initial_epoch= current_epoch,
                            epochs=current_epoch+1, #goes from intial to epochs, so need it to be greater than initial
                            callbacks = [ModelCheckpoint('%scurrent_model_while_running.hdf5'%save_folder_name)],
                            verbose=1)
    
    # Save loss
    loss = loss + network_history.history['loss']
    val_loss = val_loss + network_history.history['val_loss']
    energy_loss = energy_loss + network_history.history['EnergyLoss']
    val_energy_loss = val_energy_loss + network_history.history['val_EnergyLoss']
    losses = [loss, val_loss, energy_loss, val_energy_loss]
    
    if train_variables > 1:
        zenith_loss = zenith_loss + network_history.history['ZenithLoss']
        val_zenith_loss = val_zenith_loss + network_history.history['val_ZenithLoss']
        losses.append(zenith_loss)
        losses.append(val_zenith_loss)
    if train_variables > 2:
        track_loss = track_loss + network_history.history['TrackLoss']
        val_track_loss = val_track_loss + network_history.history['val_TrackLoss']
        losses.append(track_loss)
        losses.append(val_track_loss)
	
    #SAVE EVERY FULL PASS THROUGH DATA
    if epoch%len(file_names) == (len(file_names)-1):
        model_DC.save("%s%s_%iepochs_model.hdf5"%(save_folder_name,filename,current_epoch+1))
        afile = open("%ssaveloss_%iepochs.txt"%(save_folder_name,current_epoch+1),"w")
        
        losslen = len(losses_names)
        for a_list in range(0,losslen): 
            afile.write("%s = ["%losses_names[a_list])
            for a_loss in losses[a_list]:
                afile.write("%s, " %a_loss)
            afile.write("]\n")
        afile.close()
        
    current_epoch +=1
    
t1 = time.time()
print("This took me %f minutes"%((t1-t0)/60.))


# Put all the test sets together
Y_test_use = None
X_test_DC_use = None
X_test_IC_use = None

for file in file_names:
    f = h5py.File(file, 'r')
    Y_test = f['Y_test'][:]
    X_test_DC = f['X_test_DC'][:]
    X_test_IC = f['X_test_IC'][:]
    f.close()
    del f
    
    if Y_test_use is None:
        Y_test_use = Y_test
        X_test_DC_use = X_test_DC
        X_test_IC_use = X_test_IC
    else:
        Y_test_use = numpy.concatenate((Y_test_use, Y_test))
        X_test_DC_use = numpy.concatenate((X_test_DC_use, X_test_DC))
        X_test_IC_use = numpy.concatenate((X_test_IC_use, X_test_IC))
print(Y_test_use.shape)

# Score network
score = model_DC.evaluate([X_test_DC_use,X_test_IC_use], Y_test_use, batch_size=256)
print("final score on test data: loss: {:.4f} / accuracy: {:.4f}".format(score[0], score[1]))


model_DC.save("%s%s_model_final.hdf5"%(save_folder_name,filename))


# Predict with test data
t0 = time.time()
Y_test_predicted = model_DC.predict([X_test_DC_use,X_test_IC_use])
t1 = time.time()
print("This took me %f seconds for %i events"%(((t1-t0)),Y_test_predicted.shape[0]))

### SAVE OUTPUT TO FILE ##
if save==True:
    file = open("%sfinaloutput.txt"%save_folder_name,"w")
    file.write("training on {} samples, testing on {} samples".format(len(Y_train),len(Y_test)))
    file.write("final score on test data: loss: {:.4f} / accuracy: {:.4f}\n".format(score[0], score[1]))
    file.write("This took %f minutes\n"%((t1-t0)/60.))
    if train_variables > 1:
        losses = [loss, energy_loss, zenith_loss, val_loss, val_energy_loss, val_zenith_loss]
        losses_names = ['loss', 'energy_loss', 'zenith_loss', 'val_loss', 'val_energy_loss', 'val_zenith_loss']
    else:
        losses = [loss, energy_loss, val_loss, val_energy_loss]
        losses_names = ['loss', 'energy_loss', 'val_loss', 'val_energy_loss']
    losslen = len(losses_names)
    for a_list in range(0,losslen):
        file.write("%s = ["%losses_names[a_list])
        for a_loss in losses[a_list]:
            file.write("%s, " %a_loss)
        file.write("]\n")
    file.close()

### MAKE THE PLOTS ###
from PlottingFunctions import plot_single_resolution
from PlottingFunctions import plot_2D_prediction
from PlottingFunctions import plot_history
from PlottingFunctions import plot_bin_slices
from PlottingFunctions import plot_history_from_list
from PlottingFunctions import plot_history_from_list_split
from PlottingFunctions import plot_distributions

plot_history_from_list(loss,val_loss,save,save_folder_name,logscale=True)
if num_labels > 1:
    plot_history_from_list_split(energy_loss,val_energy_loss,zenith_loss,val_zenith_loss,save=save,savefolder=save_folder_name,logscale=True)


plots_names = ["Energy", "CosZenith", "Track"]
plots_units = ["GeV", "", "m"]
maxabs_factors = [100., 1., 200.]
if num_labels == 3: 
    maxvals = [max_energy, 1., max(Y_test_use[:,2])*maxabs_factor[2]]
else:
    maxvals = [max_energy, 1., 0.]
minvals = [min_energy, -1., 0.]
use_fractions = [True, False, True]
for num in range(0,num_labels):

    plot_num = num
    plot_name = plots_names[num]
    plot_units = plots_units[num]
    maxabs_factor = maxabs_factors[num]
    maxval = maxvals[num]
    minval = minvals[num]
    use_frac = use_fractions[num]
    print("Plotting %s at position %i in test output"%(plot_name, num))

    plot_2D_prediction(Y_test_use[:,plot_num]*maxabs_factor, Y_test_predicted[:,plot_num]*maxabs_factor,\
			save,save_folder_name,\
			minval=minval,maxval=maxval,\
			variable=plot_name,units=plot_units)
    plot_single_resolution(Y_test_use[:,plot_num]*maxabs_factor, Y_test_predicted[:,plot_num]*maxabs_factor,\
                       save=save,savefolder=save_folder_name,\
                       variable=plot_name,units=plot_units)
    plot_distributions(Y_test_use[:,plot_num]*maxabs_factor, Y_test_predicted[:,plot_num]*maxabs_factor,\
			save,save_folder_name,\
			variable=plot_name,units=plot_units)
    plot_bin_slices(Y_test_use[:,plot_num]*maxabs_factor, Y_test_predicted[:,plot_num]*maxabs_factor,\
                       use_fraction = use_frac,\
                       bins=10,min_val=minval,max_val=maxval,\
                       save=True,savefolder=save_folder_name,\
                       variable=plot_name,units=plot_units)
    if num > 0:
        plot_bin_slices(Y_test_use[:,num], Y_test_predicted[:,num], \
                       min_energy = min_energy, max_energy=max_energy, true_energy=Y_test_use[:,0]*max_energy, \
                       use_fraction = False, \
                       bins=10,min_val=minval,max_val=maxval,\
                       save=True,savefolder=save_folder_name,\
                       variable=plot_name,units=plot_units)

#########################
# Version of CNN on 9 Sept 2019
# 
# Runs net and plots
# Can take in energy, zenith, and track
# 1 = energy, 2 = energy and zenith, 3 = energy, zeith, track
#
#
#
#############################

import numpy
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
import matplotlib.pyplot as plt
import keras

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_file",type=str,default=None,
                    dest="input_file", help="name and path for input file")
parser.add_argument("-d", "--path",type=str,default='/mnt/scratch/micall12/training_files/',
                    dest="path", help="path to input files")
parser.add_argument("-n", "--name",type=str,default='nue_cascade_allfiles',
                    dest="name", help="name for output directory and model file name")
parser.add_argument("-e","--epochs", type=int,default=30,
                    dest="epochs", help="number of epochs for neural net")
parser.add_argument("--drop", type=float,default=0.2,
                    dest="dropout", help="change dropout for all network (should be > 0 and < 1.0)")
parser.add_argument("--lr", type=float,default=1e-3,
                    dest="learning_rate", help="set the learning rate")
parser.add_argument("--batch", type=int,default=256,
                    dest="batch_size", help="set the batch size")
parser.add_argument("--load_weights", type=str, default = False,
                    dest="load_weights", help="True if you want to load and use weights from old model")
parser.add_argument("--old_model", type=str, default = None,
                    dest ="old_model", help="Path + name of old model hdf5 file to load weights from")
parser.add_argument("--variables", type=int, default = 1,
                    dest ="train_variables", help="1 for energy only, 2 for energy and zenith")
args = parser.parse_args()

# Settings from args
input_file = args.input_file
path = args.path
num_epochs = args.epochs
train_variables = args.train_variables
batch_size = args.batch_size
learning_rate = args.learning_rate
DC_drop_value = args.dropout
IC_drop_value = args.dropout
connected_drop_value = args.dropout
filename = args.name

save = True
save_folder_name = "%s/output_plots/%s/"%(path,filename)
if save==True:
    if os.path.isdir(save_folder_name) != True:
        os.mkdir(save_folder_name)
        
use_old_weights = args.load_weights
old_model_name = args.old_model


# Import files

f = h5py.File(input_file, 'r')
Y_train = f['Y_train'][:]
Y_test = f['Y_test'][:]
X_train_DC = f['X_train_DC'][:]
X_test_DC = f['X_test_DC'][:]
X_train_IC = f['X_train_IC'][:]
X_test_IC = f['X_test_IC'][:]
X_validate_DC = f['X_validate_DC'][:]
X_validate_IC = f['X_validate_IC'][:]
Y_validate = f['Y_validate'][:]
f.close()
del f


# Mask Energy

mask_energy_train = numpy.logical_and(Y_train[:,0]>10.,Y_train[:,0]<150.)
mask_energy_test = numpy.logical_and(Y_test[:,0]>10.,Y_test[:,0]<150.)
mask_energy_validate = numpy.logical_and(Y_validate[:,0]>10.,Y_validate[:,0]<150.)

Y_train = numpy.array(Y_train)
Y_test = numpy.array(Y_test)
X_train_DC = numpy.array(X_train_DC)
X_test_DC = numpy.array(X_test_DC)
X_train_IC = numpy.array(X_train_IC)
X_test_IC = numpy.array(X_test_IC)
X_validate_DC = numpy.array(X_validate_DC)
X_validate_IC = numpy.array(X_validate_IC)
Y_validate = numpy.array(Y_validate)

Y_train = Y_train[mask_energy_train]
Y_test = Y_test[mask_energy_test]
X_train_DC = X_train_DC[mask_energy_train]
X_test_DC = X_test_DC[mask_energy_test]
X_train_IC = X_train_IC[mask_energy_train]
X_test_IC = X_test_IC[mask_energy_test]
X_validate_DC = X_validate_DC[mask_energy_validate]
X_validate_IC = X_validate_IC[mask_energy_validate]
Y_validate = Y_validate[mask_energy_validate]


# Return features and labels, to be used for network
num_features_DC = X_train_DC.shape[-1]
num_features_IC = X_train_IC.shape[-1]
num_labels = train_variables #Y_train.shape[-1] ## NEED TO CHANGE MANUALLY!
print("Training set: %i, Testing set: %i"%(len(Y_train),len(Y_test)))


# Transform Output labels


maxabs = 150. #Divide energy by maxabs
#Put variables together in training, testing labels
Y_train_use = numpy.zeros((Y_train.shape[0],num_labels))

for event in range(0,Y_train.shape[0]):
    Y_train_use[event,0] = Y_train[event,0]/float(maxabs) #energy
    if num_labels > 1:
        Y_train_use[event,1] = numpy.cos(Y_train[event,1]) #cos zenith
        #Y_train_use[event,1] = Y_train[event,1] #cos zenith
    if num_labels > 2:
        Y_train_use[event,2] = Y_train[event,7] #track

Y_val_use = numpy.zeros((Y_validate.shape[0],num_labels))
for event in range(0,Y_validate.shape[0]):
    Y_val_use[event,0] = Y_validate[event,0]/float(maxabs) #energy
    if num_labels > 1:
        Y_val_use[event,1] = numpy.cos(Y_validate[event,1]) #cos zenith
        #Y_val_use[event,1] = Y_validate[event,1] #zenith
    if num_labels > 2:
        Y_val_use[event,2] = Y_validate[event,7] #track

Y_test_use = numpy.zeros((Y_test.shape[0],num_labels))
for event in range(0,Y_test.shape[0]):
    Y_test_use[event,0] = Y_test[event,0]/float(maxabs) #energy
    if num_labels > 1:
        Y_test_use[event,1] = numpy.cos(Y_test[event,1]) #cos zenith
        #Y_test_use[event,1] = Y_test[event,1] # zenith
    if num_labels > 2:
        Y_test_use[event,2] = Y_test[event,7] #track


# Check Output Labels

"""
plt.figure()
plt.hist(Y_train_use[:,0],bins=150,range=(0.,1.));
plt.title("Energy Distribution")
plt.xlabel("energy (GeV)")


plt.figure()
plt.hist(Y_val_use[:,0],bins=75);
plt.title("Energy Distribution")
plt.xlabel("energy (GeV)")

plt.figure()
plt.hist(Y_test_use[:,0],bins=75);
plt.title("Energy Distribution")
plt.xlabel("energy (GeV)")

plt.figure()
plt.hist(Y_train_use[:,1],bins=75);
plt.title("Cos(Zenith) Distribution")
plt.xlabel("Cosine Zenith")

plt.figure()
plt.hist(Y_val_use[:,1],bins=75);
plt.title("Cos(Zenith) Distribution")
plt.xlabel("Cosine Zenith")

plt.figure()
plt.hist(Y_test_use[:,1],bins=75);
plt.title("Cos(Zenith) Distribution")
plt.xlabel("Cosine Zenith")
"""

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


## MIRCO NETWORK ##

# DEEP CORE #
print("Train Data DC", X_train_DC.shape)
strings = X_train_DC.shape[1]
dom_per_string = X_train_DC.shape[2]
dom_variables = X_train_DC.shape[3]

# Conv DC + batch normalization, later dropout and maxpooling
input_DC = Input(shape=(strings, dom_per_string, dom_variables))

conv1_DC = Conv2D(100,kernel_size=(strings,5),padding='same',activation='tanh')(input_DC) #tanh
batch1_DC = BatchNormalization()(conv1_DC)
pool1_DC = MaxPooling2D(pool_size=(1,2))(batch1_DC)
drop1_DC = Dropout(DC_drop_value)(pool1_DC)

conv2_DC = Conv2D(100,kernel_size=(strings,7),padding='same',activation='relu')(drop1_DC) #relu
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
print("Train Data IC", X_train_IC.shape)
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


# CALL OLD FILE TO LOAD
if use_old_weights:
    model_DC.load_weights(old_model_name)


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
    return mean_squared_logarithmic_error(y_truth[:,2],y_predicted[:,2])/10.

## Compile ##
if num_labels == 3:
    def CustomLoss(y_truth,y_predicted):
        energy_loss = EnergyLoss(y_truth,y_predicted)
        zenith_loss = ZenithLoss(y_truth,y_predicted)
        track_loss = TrackLoss(y_truth,y_predicted)
        return energy_loss + zenith_loss + track_loss

    model_DC.compile(loss=CustomLoss,
              optimizer=Adam(lr=learning_rate),
              metrics=[EnergyLoss,ZenithLoss,TrackLoss])

elif num_labels == 2:
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



# Run neural network and record time ##
t0 = time.time()
network_history = model_DC.fit([X_train_DC, X_train_IC], Y_train_use,
                            validation_data= ([X_validate_DC, X_validate_IC], Y_val_use), #validation_split=0.2,
                            batch_size=batch_size,
                            epochs=num_epochs,
                            #callbacks = [EarlyStopping(patience=6), ModelCheckpoint('%scurrent_model_while_running.hdf5'%save_folder_name)],
                            callbacks = [ModelCheckpoint('%scurrent_model_while_running.hdf5'%save_folder_name)],
                            verbose=1)

t1 = time.time()
print("This took me %f minutes"%((t1-t0)/60.))



# SAVE OUTPUT IN LOG FILE
print(network_history.history.keys())
print("loss = loss + ", network_history.history['loss'])
print("val_loss = val_loss + ", network_history.history['val_loss'])

print("energy_loss = energy_loss + ", network_history.history['EnergyLoss'])
if train_variables > 1:
    print("zenith_loss = zenith_loss + ", network_history.history['ZenithLoss'])
if train_variables > 2:
    print("track_loss = track_loss + ",network_history.history['TrackLoss'])

print("val_energy_loss = val_energy_loss + ",network_history.history['val_EnergyLoss'])
if train_variables > 1:
    print("val_zenith_loss = val_zenith_loss + ",network_history.history['val_ZenithLoss'])
if train_variables > 2:
    print("val_track_loss = val_track_loss + ",network_history.history['val_TrackLoss'])


#SCORE 
score = model_DC.evaluate([X_test_DC,X_test_IC], Y_test_use, batch_size=256)
print("final score on test data: loss: {:.4f} / accuracy: {:.4f}".format(score[0], score[1]))
print(network_history.history.keys())
print(score)

# SAVE MODEL
model_DC.save("%s%s_model.hdf5"%(save_folder_name,filename))

# PREDICT MODEL
Y_test_predicted = model_DC.predict([X_test_DC,X_test_IC])


if save==True:
    file = open("%soutput.txt"%save_folder_name,"w")
    file.write("training on {} samples, testing on {} samples".format(len(Y_train),len(Y_test)))
    file.write("final score on test data: loss: {:.4f} / accuracy: {:.4f}\n".format(score[0], score[1]))
    file.write("This took %f minutes"%((t1-t0)/60.))
    file.close()


from PlottingFunctions import plot_single_resolution
from PlottingFunctions import plot_2D_prediction
from PlottingFunctions import plot_history
from PlottingFucntions import plot_distributions

plot_history(network_history,save,save_folder_name)

#Plot energy
plot_2D_prediction(Y_test_use[:,0]*150., Y_test_predicted[:,0]*150.,\
                    save,save_folder_name,bins=140.,minval=10.,maxval=150,\
                    variable="Energy",units='GeV')
plot_single_resolution(Y_test_use[:,0]*150., Y_test_predicted[:,0]*150.,\
                        save=save,savefolder=save_folder_name,\
                        variable="Energy",units='GeV')
plot_distributions(Y_test_use[:,0]*150, Y_test_predicted[:,0]*150.,\
                    save,save_folder_name,variable="Energy",units='GeV')
#Plot Zenith
if num_labels > 1:
    plot_single_resolution(Y_test_use[:,1], Y_test_predicted[:,1],\
                            save=save,savefolder=save_folder_name,\
                            variable="CosZenith",units='')
    plot_2D_prediction(Y_test_use[:,1], Y_test_predicted[:,1],\
                        save,save_folder_name,minval=-1,maxval=1,\
                        variable="CosZenith",units='')
    plot_distributions(Y_test_use[:,1], Y_test_predicted[:,1],\
                        save,save_folder_name,variable="CosZenith",units='')
#Plot track
if num_labels > 2:
    plot_single_resolution(Y_test_use[:,2], Y_test_predicted[:,2],\
                            save=save,savefolder=save_folder_name,\
                            variable="Track",units='m')
    plot_2D_prediction(Y_test_use[:,2], Y_test_predicted[:,2],\
                        save,save_folder_name,minval=0,maxval=150,\
                        variable="Track",units='m')


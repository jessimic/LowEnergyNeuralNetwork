#########################
# Takes split input file (train, test, validate) and tests on all events
#   Excepts transformed input and output (energy and zenith)
#   Test net old model and plots output
#   Must manually update network model to MATCH
############################

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
import keras

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_file",type=str,default=None,
                    dest="input_file", help="names for input files")
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
parser.add_argument("--old_model", type=str, default = None,
                    dest ="old_model", help="Path + name of old model hdf5 file to load weights from")
parser.add_argument("--variables", type=int, default = 1,
                    dest ="train_variables", help="1 for energy only, 2 for energy and zenith")
parser.add_argument("--emax",type=float,default=200.0,
                    dest="emax",help="Cut anything greater than this energy (in GeV)")
parser.add_argument("--emin",type=float,default=0.0,
                    dest="emin",help="Cut anything less than this energy (in GeV)")
args = parser.parse_args()

# EXAMPLE OPTIONS FOR DEFAULT RUN:
# $ python CNN_CreateRunModel.py --input_file 'Level5_IC86.2013_genie_nue.012640.100.cascade.lt60_vertexDC.transformed.hdf5' --path '/mnt/scratch/micall12/training_files/' --name 'nue_cascade_allfiles' --epochs 30 --drop 0.2 --lr 1e-3 --batch 256 --old_model 'current_model_while_running.hdf5' --variables 2

### ONLY RUN ON TRANSFORMED FILES!!!!!!!!!! ###
## RUN TrasnformFiles first ### 
print("ASSUMING INPUT AND OUTPUT (energy and zenith) ARE TRANSFORMED!")

# Read in args and settings
file_dir = args.path
input_file = file_dir + args.input_file
num_epochs = args.epochs
learning_rate = args.learning_rate
batch_size = args.batch_size
DC_drop_value = args.dropout
IC_drop_value = args.dropout
connected_drop_value = args.dropout
old_model_name = args.old_model
emax = args.emax
emin = args.emin
train_variables = args.train_variables

filename = args.name
save = True
save_folder_name = "/mnt/home/micall12/DNN_LE/output_plots/%s/"%(filename)
if save==True:
    if os.path.isdir(save_folder_name) != True:
        os.mkdir(save_folder_name)


### Import Files ###
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

# Put whole file together into one large testing set
#Y_all_label = Y_train + Y_validate + Y_test
print("USING WHOLE FILE TO TEST")
allevents = Y_train.shape[0] + Y_validate.shape[0] + Y_test.shape[0]
Y_all_test = numpy.zeros((allevents,train_variables))
X_all_test_DC = numpy.zeros((allevents,X_train_DC.shape[1],X_train_DC.shape[2],X_train_DC.shape[3]))
X_all_test_IC = numpy.zeros((allevents,X_train_IC.shape[1],X_train_IC.shape[2],X_train_IC.shape[3]))
for e_index in range(0,Y_train.shape[0]):
    Y_all_test[e_index,0] = Y_train[e_index,0]
    if train_variables>1:
        Y_all_test[e_index,1] = Y_train[e_index,1] 
    X_all_test_DC[e_index,:,:,:] = X_train_DC[e_index,:,:,:]
    X_all_test_IC[e_index,:,:,:] = X_train_IC[e_index,:,:,:]
for e_index in range(Y_train.shape[0],Y_validate.shape[0]):
    Y_all_test[e_index,0] =  Y_validate[e_index,0] 
    if train_variables>1:
        Y_all_test[e_index,1] = Y_validate[e_index,1]
    X_all_test_DC[e_index,:,:,:] =  X_validate_DC[e_index,:,:,:]
    X_all_test_IC[e_index,:,:,:] = X_validate_IC[e_index,:,:,:]
for e_index in range(Y_validate.shape[0],Y_test.shape[0]):
    Y_all_test[e_index,0] =  Y_test[e_index,0]
    if train_variables>1:
        Y_all_test[e_index,1] = Y_test[e_index,1]
    X_all_test_DC[e_index,:,:,:] =  X_test_DC[e_index,:,:,:]
    X_all_test_IC[e_index,:,:,:] = X_test_IC[e_index,:,:,:]

print("shape after putting all sets together (no energy cut)")
print(Y_all_test.shape, X_all_test_DC.shape, X_all_test_IC.shape)

energy = numpy.array(Y_all_test[:,0])
mask_energy = numpy.logical_and(energy<emax,energy>emin)
X_all_test_DC = numpy.array(X_all_test_DC)
X_ecut_test_DC = X_all_test_DC[mask_energy]
X_all_test_IC = numpy.array(X_all_test_IC)
X_ecut_test_IC = X_all_test_IC[mask_energy]
Y_all_test = numpy.array(Y_all_test)
Y_ecut_test = Y_all_test[mask_energy]

print("shape after energy cut, keeping events in range %f - %f GeV"%(emin,emax))
print(X_ecut_test_DC.shape, X_ecut_test_IC.shape, Y_ecut_test.shape)

# Return features and labels, to be used for network
num_features_DC = X_train_DC.shape[-1]
num_features_IC = X_train_IC.shape[-1]
num_labels = train_variables

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


## MATCH NETWORK ##
##############################
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
model_DC = Model(inputs=[input_DC,input_IC],outputs=output) #lambda_layer)
######################


# LOAD OLD MODEL
model_DC.load_weights(old_model_name)

# WRITE OWN LOSS FOR MORE THAN ONE REGRESSION OUTPUT
from keras.losses import mean_squared_error
from keras.losses import mean_squared_logarithmic_error

def CustomLoss(y_truth,y_predicted):
    energy_loss = EnergyLoss(y_truth,y_predicted)
    zenith_loss = ZenithLoss(y_truth,y_predicted)
    return energy_loss + zenith_loss #+ track_loss

def EnergyLoss(y_truth,y_predicted):
    return mean_squared_logarithmic_error(y_truth[:,0],y_predicted[:,0]) #/120.

def ZenithLoss(y_truth,y_predicted):
    return mean_squared_error(y_truth[:,1],y_predicted[:,1])

## Compile ##
if train_variables == 2:
    model_DC.compile(loss=CustomLoss,
              optimizer=Adam(lr=1e-3),
              metrics=[EnergyLoss,ZenithLoss])
else:
    model_DC.compile(loss=EnergyLoss,
                optimizer=Adam(lr=1e-3),
                metrics=[EnergyLoss])


from PlottingFunctions import plot_single_resolution
from PlottingFunctions import plot_2D_prediction

#Use test set to predict values
Y_test_predicted = model_DC.predict([X_ecut_test_DC,X_ecut_test_IC])

Y_test = numpy.copy(Y_ecut_test) #reshape(Y_test_all_labels[:,:train_variables], Y_test.shape[0])

plot_2D_prediction(Y_ecut_test[:,0]*emax, Y_test_predicted[:,0]*emax,save,save_folder_name,minvag=emin,maxval=emax,variable="Energy",units='GeV')
plot_single_resolution(Y_ecut_test[:,0], Y_test_predicted[:,0],\
                       save=save,savefolder=save_folder_name,\
                       variable="Energy",units='GeV')
if train_variables == 2:
    plot_2D_prediction(Y_ecut_test[:,1], Y_test_predicted[:,1],save,save_folder_name,minval=-1.,maxval=1.,variable="CosZenith",units='')
    plot_single_resolution(Y_ecut_test[:,1], Y_test_predicted[:,1],\
                       save=save,savefolder=save_folder_name,\
                       variable="CosZenith",units='')

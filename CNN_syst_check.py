###################################
# Meant to check systematic sets after network is trained
#   Edit input files names to pull from syst sets
#   Edit model path AND MATCH NETWORK EXACTLY -- all manual
#   Outputs a number of comparson plots
#   Needs PlottingFunctions to access these plots
#####################################


import numpy
import h5py
import time
import os, sys
import random
from collections import OrderedDict
import itertools
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy import interpolate
import scipy.stats
#from scipy.signal import peak_widths

import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)

# Set up all the systematic sets
path = '/mnt/scratch/micall12/training_files/'
file1= path + 'Level5p_IC86.2013_genie_nue.012640.300.transformed.hdf5'
file2= path + 'Level5p_IC86.2013_genie_nue.012643.300.lt60_vertexDC.transformed.hdf5' #'Level5_IC86.2013_genie_nue.012643.100.cascade.lt60_vertexDC.transformed.hdf5'
file3= path + 'Level5p_IC86.2013_genie_nue.012663.300.transformed.hdf5'
file4= path + 'Level5p_IC86.2013_genie_nue.012603.300.transformed.hdf5'
file5= path + 'Level5p_IC86.2013_genie_nue.012600.300.transformed.hdf5'
name1= 'baseline (no RDE)' #'12640'
name2= '0.96 DOM eff (no RDE)' #"12643"
name3= '0.96 DOM eff, +10 holeice (no RDE)' #"12663"
name4= '0.96 DOM eff (w/ RDE)' #"12603"
name5='baseline (w/ RDE)' #"12600"

# Pick model to load
model_name=path + 'nue_5p_cascade_allfiles_drop20_model.hdf5'

# Make these match
filelist = [file1, file2, file3, file4, file5]
namelist = [name1, name2, name3, name4, name5]
num_namelist = ["640", "643", "663", "603", "600"]
num_labels=1

#Set up method to save output plots
save = True
save_folder_name = "compare"
numberlist = [12640, 12643, 12663, 12603, 12600]
for a_number in numberlist:
    save_folder_name += "_%s"%a_number
save_folder_name += "/"

index = 0
Y_test = {}
X_test_DC = {}
X_test_IC = {}
reco_test = {}
min_events = None
for file in filelist:
    keyname = "file_%i"%index
    f = h5py.File(file, 'r')
    Y_test[keyname] = f['Y_test'][:]
    X_test_DC[keyname] = f['X_test_DC'][:]
    X_test_IC[keyname] = f['X_test_IC'][:]
    reco_test[keyname] = f['reco_test'][:]
    f.close()
    del f
    
    if not min_events:
        min_events = len(Y_test[keyname])
    else:
        if min_events > len(Y_test[keyname]):
            min_events = len(Y_test[keyname])
        
    #if keyname=='file_0': #only use subset of big 
    #    Y_test[keyname]=Y_test[keyname][:13298]
    #    X_test_DC[keyname]=X_test_DC[keyname][:13298]
    #    X_test_IC[keyname]=X_test_IC[keyname][:13298]
        
    print("Testing set for %s: %i"%(namelist[index], len(Y_test[keyname]) ) )
    index+=1
    
print("Only saving a subset of all the events, using %i"%min_events)
for a_index in range(0,len(namelist)):
    keyname = "file_%i"%a_index
    Y_test[keyname]=Y_test[keyname][:min_events]
    X_test_DC[keyname]=X_test_DC[keyname][:min_events]
    X_test_IC[keyname]=X_test_IC[keyname][:min_events]
    reco_test[keyname] =  reco_test[keyname][:min_events]

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
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

## MIRCO NETWORK ##

dropout = 0.1
DC_drop_value = dropout
IC_drop_value = dropout
connected_drop_value = dropout

# DEEP CORE #
print("Test Data DC", X_test_DC['file_0'].shape)
strings = X_test_DC['file_0'].shape[1]
dom_per_string = X_test_DC['file_0'].shape[2]
dom_variables = X_test_DC['file_0'].shape[3]

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
print("Train Data IC", X_test_IC['file_0'].shape)
strings_IC = X_test_IC['file_0'].shape[1]
dom_per_string_IC = X_test_IC['file_0'].shape[2]
dom_variables_IC = X_test_IC['file_0'].shape[3]

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
lambda_layer = Lambda(lambda x: (x*335)/1.)(output)
model_DC = Model(inputs=[input_DC,input_IC],outputs=lambda_layer)

## Compile ##
model_DC.compile(loss='mean_squared_error',
              optimizer=Adam(lr=1e-3), #0.00001
              metrics=['mean_squared_error'])
print(model_name)
model_DC.load_weights(model_name)

# Use model to predict
Y_test_predicted_energy = {}
Y_test_energy = {}
Y_test_energy_reco = {}
for index in range(0,len(filelist)):
    keyname = "file_%i"%index
    t0 = time.time()
    Y_test_predicted = model_DC.predict([X_test_DC[keyname],X_test_IC[keyname]])
    t1 = time.time()
    Y_test_predicted_energy[keyname] = numpy.reshape(Y_test_predicted, len(Y_test_predicted))
    Y_test_energy[keyname] = numpy.reshape(Y_test[keyname][:,0], len(Y_test[keyname])) 
    Y_test_energy_reco[keyname] = numpy.reshape(reco_test[keyname][:,0], len(reco_test[keyname])) 
    
    
    print("This took me %f minutes for %i events"%(((t1-t0)/60.),len(Y_test_predicted)))

# SAVING 
print(save_folder_name)
if save==True:
    if os.path.isdir(save_folder_name) != True:
        os.mkdir(save_folder_name)

from PlottingFunctions import plot_2D_prediction
from PlottingFunctions import plot_distributions_CCNC 
from PlottingFunctions import plot_resolution_CCNC 
from PlottingFunctions import plot_single_resolution
from PlottingFunctions import plot_compare_resolution
from PlottingFunctions import plot_systematic_slices
from PlottingFunctions import plot_energy_slices

plot_2D_prediction(Y_test_energy['file_0'], Y_test_predicted_energy['file_0'],\
                    save=save,savefolder=save_folder_name)
plot_distributions_CCNC(Y_test['file_0'],Y_test_energy['file_0'], Y_test_predicted_energy['file_0'],\
                        save=save,savefolder=save_folder_name)
plot_resolution_CCNC(Y_test['file_0'],Y_test_energy['file_0'], Y_test_predicted_energy['file_0'],\
                        save=save,savefolder=save_folder_name)
plot_single_resolution(Y_test_energy['file_0'], Y_test_predicted_energy['file_0'],\
                        save=save,savefolder=save_folder_name)
plot_single_resolution(Y_test_energy['file_0'], Y_test_predicted_energy['file_0'],\
                        use_old_reco=True, old_reco=Y_test_energy_reco['file_0'],\
                        save=save,savefolder=save_folder_name)
plot_compare_resolution(Y_test_energy, Y_test_predicted_energy,namelist=namelist,num_namelist=num_namelist,\
                        save=save,savefolder=save_folder_name)
plot_systematic_slices(Y_test_energy, Y_test_predicted_energy,num_namelist,\
                        save=save,savefolder=save_folder_name)
plot_energy_slices(Y_test_energy['file_0'], Y_test_predicted_energy['file_0'],\
                    use_old_reco=True, old_reco=Y_test_energy_reco['file_0'],\
                    save=save,savefolder=save_folder_name)

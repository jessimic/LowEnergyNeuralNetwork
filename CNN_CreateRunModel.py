#########################
# Version of DNN using Mirco Config
# Set to take in Robust tranformed file
# Runs net and plots
############################3

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


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_file",type=str,default='Level5_IC86.2013_genie_nue.012640.100.cascade.lt60_vertexDC.transformed.hdf5',
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
parser.add_argument("--load_weights", type=bool, default = False,
                    dest="load_weights", help="True if you want to load and use weights from old model")
parser.add_argument("--old_model", type=str, default = None,
                    dest ="old_model", help="Path + name of old model hdf5 file to load weights from")
args = parser.parse_args()

# EXAMPLE OPTIONS FOR DEFAULT RUN:
# $ python CNN_CreateRunModel.py --input_file 'Level5_IC86.2013_genie_nue.012640.100.cascade.lt60_vertexDC.transformed.hdf5' --path '/mnt/scratch/micall12/training_files/' --name 'nue_cascade_allfiles' --epochs 30 --drop 0.2 --lr 1e-3 --batch 256 --load_weights False --old_model 'current_model_while_running.hdf5'

### ONLY RUN ON TRANSFORMED FILES!!!!!!!!!! ###
## RUN TrasnformFiles first ### 
### Import Files ###
file_dir = args.path
input_file = file_dir + args.input_file
num_epochs = args.epochs
learning_rate = args.learning_rate
batch_size = args.batch_size
DC_drop_value = args.dropout
IC_drop_value = args.dropout
connected_drop_value = args.dropout
use_old_weights = args.load_weights
old_model_name = args.old_model

filename = args.name + "_drop%i_lr%f_batch%i"%(args.dropout*100,learning_rate,batch_size)
save = True
save_folder_name = "/mnt/home/micall12/DNN_LE/output_plots/%s/"%(filename)
if save==True:
    if os.path.isdir(save_folder_name) != True:
        os.mkdir(save_folder_name)


f = h5py.File(input_file, 'r')
Y_train = f['Y_train'][:]
Y_test = f['Y_test'][:]
X_train_DC = f['X_train_DC'][:]
X_test_DC = f['X_test_DC'][:]
X_train_IC = f['X_train_IC'][:]
X_test_IC = f['X_test_IC'][:]
f.close()
del f

Y_train = Y_train[:,0] # ENERGY ONLY
#Y_test = Y_test[:,0] # ENERGY ONLY

# Return features and labels, to be used for network
num_features_DC = X_train_DC.shape[-1]
num_features_IC = X_train_IC.shape[-1]
num_labels = 1 #Y_train.shape[-1] ## NEED TO CHANGE MANUALLY!
print("Training set: %i, Testing set: %i"%(len(Y_train),len(Y_test)))


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
lambda_layer = Lambda(lambda x: (x*335)/1.)(output)
model_DC = Model(inputs=[input_DC,input_IC],outputs=lambda_layer)

#print(model_DC.summary())



## Compile ##
model_DC.compile(loss='mean_squared_error',
              optimizer=Adam(lr=learning_rate),
              metrics=['mean_squared_error'])

if use_old_weights:
    model_DC.loadwieghts(old_model_name)

## Run neural network and record time ##
t0 = time.time()
network_history = model_DC.fit([X_train_DC, X_train_IC], Y_train,
                            batch_size=batch_size,
                            validation_split=0.25,
                            epochs=num_epochs,
                            callbacks = [EarlyStopping(patience=6), ModelCheckpoint('%scurrent_model_while_running.hdf5'%save_folder_name)],
                            verbose=1)

t1 = time.time()
print("This took me %f minutes"%((t1-t0)/60.))


score = model_DC.evaluate([X_test_DC,X_test_IC], Y_test[:,0], batch_size=256)
print("final score on test data: loss: {:.4f} / accuracy: {:.4f}".format(score[0], score[1]))
print(network_history.history.keys())
print(score)

model_DC.save("%s%s_model.hdf5"%(save_folder_name,filename))

print(network_history.history['loss'])
print(network_history.history['val_loss'])


from PlottingFunctions import plot_history
from PlottingFunctions import plot_distributions_CCNC
from PlottingFunctions import plot_resolution_CCNC
from PlottingFunctions import plot_2D_prediction

#Use test set to predict values
Y_test_predicted = model_DC.predict([X_test_DC,X_test_IC])

Y_test_all_labels = numpy.copy(Y_test)
Y_test_predicted = numpy.reshape(Y_test_predicted, Y_test_predicted.shape[0])
Y_test = numpy.reshape(Y_test[:,0], Y_test.shape[0])

if save==True:
    file = open("%soutput.txt"%save_folder_name,"w")
    file.write("training on {} samples, testing on {} samples".format(len(Y_train),len(Y_test)))
    file.write("final score on test data: loss: {:.4f} / accuracy: {:.4f}\n".format(score[0], score[1]))
    file.write("This took %f minutes"%((t1-t0)/60.))
    file.close()



plot_history(network_history,save,save_folder_name)
plot_distributions_CCNC(Y_test_all_labels,Y_test,Y_test_predicted,save,save_folder_name)
plot_resolution_CCNC(Y_test_all_labels,Y_test,Y_test_predicted,save,save_folder_name)
plot_2D_prediction(Y_test, Y_test_predicted,save,save_folder_name)

#################################################################
# Neural Network for Low Energy IceCube Events
#
#   Takes arguments input file, output directory name, weight bool,
#        activation function, number of nodes per layer, save bool
#
#   Most of code not in functions: reads in hdf5 file already labeled,
#   splits into training and testing sets, adds weights, creates network,
#   trains network, and analyzes results through plots/stats
#
#   Contains functions:
#       create_weights which creates weights per bin
#       plot_history which plots loss vs. epochs
#       plot_prediction which plots testing set prediction vs. truth
#       find_plot_statistics which plots offset and variance stats
#
#################################################################

import numpy
import h5py
import time
import os, sys
import random
from collections import OrderedDict
import itertools
import matplotlib.pyplot as plt
import argparse

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras import initializers
from keras.optimizers import SGD
from keras.optimizers import Adam

#There is a MacOS problem...workaround it
os.environ['KMP_DUPLICATE_LIB_OK']='True'

## Create ability to change settings from terminal ##
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input",type=str,default='Level5_IC86.2013_genie_numu.014640.0000XX.hdf5',
                    dest="input_file", help="name of the input file")
parser.add_argument("--filename",type=str,default="subset",
                    dest="filename", help="number of files used for input")
parser.add_argument("-w", "--weights",type=bool, default=False,
                    dest="weights", help="Apply weighting by bin")
parser.add_argument("-f","--function", default="tanh",
                    dest="function", help = "activation function for hidden layers")
parser.add_argument("-n", "--nodes", type=int,nargs='+',default = [16, 16],
                    dest="nodes", help="number of nodes in each layer")
parser.add_argument("-s", "--save", default=True,
                    dest="save", help = "do you want to save the output and plots?")

args = parser.parse_args()
input_file = args.input_file
apply_weights = args.weights
filename = args.filename
activation_function = args.function
nodes = args.nodes
layers = len(nodes)
save = args.save

#Create directory to save to, based on settings
nodes_name = "L"
for n in nodes:
    nodes_name = nodes_name + "_%s"%n
save_folder_name = "%s_%s_%s_weights%r/"%(filename,nodes_name,activation_function,apply_weights)

### Import Files ###
f = h5py.File(input_file, 'r')
features = f['features'][:]
labels = f['labels'][:]
f.close()
del f

### Split into training and testing set ###
num_train = int(len(features)*0.9) # 90% of data is training data (traininig+validation), 10% is test data
print("training on {} samples, testing on {} samples".format(num_train, len(features)-num_train))

features_train = features.view((numpy.float32, len(features.dtype.names)))[:num_train]
labels_train   = labels[:num_train]
features_test  = features.view((numpy.float32, len(features.dtype.names)))[num_train:]
labels_test    = labels[num_train:]

### Specify type for training and testing ##
(X_train_raw, Y_train_raw) = (features_train, labels_train)
X_train_raw = X_train_raw.astype("float32")
Y_train_raw = Y_train_raw.astype("float32")

(X_test_raw, Y_test_raw) = (features_test, labels_test)
X_test_raw = X_test_raw.astype("float32")
Y_test_raw = Y_test_raw.astype("float32")


### Transform input features into specified range ###
transformer = RobustScaler().fit(X_train_raw) # determine how to scale on the training set
X_train_full = transformer.transform(X_train_raw)
X_test_full  = transformer.transform(X_test_raw)
# output is not transformed
Y_train_full = numpy.copy(Y_train_raw)
Y_test       = numpy.copy(Y_test_raw)
X_train_selected = numpy.copy(X_train_full)
X_test           = numpy.copy(X_test_full)

## Create ability to add weights ##
def create_weights(Y_train_full,X_train_selected):
    entries, bins = numpy.histogram(Y_train_full, range=[0.,60.], bins=60)
    bins[-1] = bins[-1]+0.1
    entries = entries

    Y_train_full_weights = 1./entries[numpy.digitize(Y_train_full,bins)-1]
    Y_train_full_weights = Y_train_full_weights/numpy.max(Y_train_full_weights)
    Y_train_full_weights = numpy.reshape(Y_train_full_weights,len(Y_train_full_weights))
    
    X_train = numpy.copy(X_train_selected)
    Y_train = numpy.copy(Y_train_full)
    Y_train_weights = numpy.copy(Y_train_full_weights)

    Y_train_weights_plot = numpy.copy(Y_train_weights)
    
    # artificaially increase some weights by factors of 10 or 100 to make the network pay more attention to them
    #Y_train_weights[Y_train.reshape(len(Y_train))>= 40] *= 10.
    
    return X_train, Y_train,Y_train_weights, Y_train_weights_plot

# Use weights
if apply_weights == True:
    X_train, Y_train,Y_train_weights, Y_train_weights_plot = create_weights(Y_train_full,X_train_selected)
else:
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_selected, Y_train_full, test_size=0.25)

# Return features and labels, to be used for network
num_features = X_train.shape[1]
num_labels = Y_train.shape[1]

#Define neural network
model = Sequential()
model.add(Dense(nodes[0], input_shape=(num_features,))) # tell first layer how many inputs
model.add(Activation(activation_function))
model.add(Dropout(0.1)) # "dropout" randomly drops weights during training only. can help prevent overtraining
for n in range(1,layers):
    model.add(Dense(nodes[n]))
    model.add(Activation(activation_function))
    model.add(Dropout(0.1))
model.add(Dense(num_labels, kernel_initializer=initializers.RandomUniform(minval=0., maxval=60., seed=None), activation='linear'))

weights = model.get_weights()
model.set_weights(weights)

## Run neural network and record time ##
t0 = time.time()
model.compile(loss='mean_squared_error',
              optimizer=Adam(lr=0.0001),
              metrics=['mean_squared_error'])

if apply_weights ==True:
    network_history = model.fit(X_train, Y_train,
                            batch_size=256, 
                            validation_split=0.25,
                            sample_weight=Y_train_weights,
                            epochs=45,
                            verbose=0)
else:
    network_history = model.fit(X_train, Y_train,
                            batch_size=256, 
                            validation_split=0.25,
                            epochs=45,
                            verbose=0)

t1 = time.time()

print("This took me %f minutes"%((t1-t0)/60.))
score = model.evaluate(X_test, Y_test, batch_size=256)
print("final score on test data: loss: {:.4f} / accuracy: {:.4f}".format(score[0], score[1]))
print(network_history.history.keys())
print(score)

## Functions to plot/look at stats ##
def plot_history(network_history,save=False,savefolder=None):
    """
    Plot history of neural network's loss vs. epoch
    Recieves:
        network_history = array, saved metrics from neural network training
        save = optional, bool to save plot
        savefolder = optional, output folder to save to, if not in current dir
    Returns:
        one plot, saved to files
    """
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(network_history.history['loss'])
    plt.plot(network_history.history['val_loss'])
    plt.legend(['Training', 'Validation'])
    if save == True:
        plt.savefig("%sloss_vs_epochs.png"%savefolder)
    
def plot_prediction(Y_test, Y_test_predicted,save=False,savefolder=None):
    """
    Plot testing set prediction vs truth and fractional error
    Recieves:
        Y_test = array, Y_test truth
        Y_test_prediction = array, neural network prediction output
        save = optional, bool to save plot
        savefolder = optional, output folder to save to, if not in current dir
    Returns:
        two plots, saved to files
    """
    plt.figure()
    cts,xbin,ybin,img = plt.hist2d(Y_test, Y_test_predicted, bins=60,)
    plt.plot([0,60],[0,60],'k:')
    plt.xlim(0,60)
    plt.ylim(0,60)
    plt.xlabel("True Energy")
    plt.ylabel("NN Predicted Energy")
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('counts', rotation=90)
    plt.set_cmap('viridis_r')
    plt.title("Prediction (from NN) vs Truth for Energy")
    if save == True:
        plt.savefig("%senergy_prediction_truth.png"%savefolder)
        
    fractional_error = abs(Y_test - Y_test_predicted)/ Y_test
    plt.figure()
    plt.title("Fractional Error vs. Energy")
    plt.hist2d(Y_test, fractional_error,bins=60);
    plt.xlabel("True Energy")
    plt.ylabel("Fractional Error")
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('counts', rotation=90)
    if save == True:
        plt.savefig("%sFractionalError.png"%savefolder)

def find_plot_statistics(Y_test,Y_test_predicted,save=False,savefolder=None):
    """
    Plot statistics on testing set for the neural network prediction
    Offset plot: (count # of times E_NN > E_truth vs E_NN < E_Truth)/Events
    Variance plot: squared difference of truth - NN prediction per bin
    Recieves:
        Y_test = array, Y_test truth
        Y_test_prediction = array, neural network prediction output
        save = optional, bool to save plot
        savefolder = optional, output folder to save to, if not in current dir
    Returns:
        two plots, saved to files
    """
    
    energy_space = numpy.arange(0,60,1)
    offset = numpy.zeros_like(Y_test)
    count_offset = numpy.zeros(len(energy_space))
    var_bin = numpy.zeros(len(energy_space))
    for energy in energy_space:
        remember_offset = 0
        remember_offset_num = 0.
        var = 0
        for index,y_value in enumerate(Y_test):
            if y_value > energy and y_value < energy+1:
                offset[index] = Y_test_predicted[index] - y_value
                remember_offset += 1*numpy.sign(offset[index])
                var += (Y_test_predicted[index] - y_value)**2
                remember_offset_num += 1
        if remember_offset_num == 0 : #Dont divide by zero
            remember_offset_num = 1
        count_offset[energy] = remember_offset/remember_offset_num
        var_bin[energy] = var/remember_offset_num
    
    #Offset plot    
    plt.figure()
    plt.plot(energy_space,count_offset)
    plt.xlabel("Energy (GeV)")
    plt.ylabel("Difference of True vs. NN (directional)")
    plt.title("( (# E_NN > E_true) - (# E_NN < E_true) )/ Events in bin")
    if save ==True:
        plt.savefig("%sCountOffsetDifference.png"%savefolder)
    #Variance Plot    
    plt.figure()
    plt.plot(energy_space,var_bin)
    plt.xlabel("Energy (GeV)")
    plt.ylabel("variance per bin")
    plt.title("Variance Per 1 GeV")
    if save ==True:
        plt.savefig("%sVariancePerGeV.png"%savefolder)


#Use test set to predict values
Y_test_predicted = model.predict(X_test)

Y_test_predicted = numpy.reshape(Y_test_predicted, Y_test_predicted.shape[0])
Y_test = numpy.reshape(Y_test, Y_test.shape[0])

if save==True:
    if os.path.isdir(save_folder_name) != True:
        os.mkdir(save_folder_name)
    file = open("%soutput.txt"%save_folder_name,"w") 
    file.write("training on {} samples, testing on {} samples".format(num_train, len(features)-num_train))
    file.write("final score on test data: loss: {:.4f} / accuracy: {:.4f}\n".format(score[0], score[1]))
    file.write("This took %f minutes"%((t1-t0)/60.))
    file.close()

plot_history(network_history,save,save_folder_name)
plot_prediction(Y_test, Y_test_predicted,save,save_folder_name)
find_plot_statistics(Y_test,Y_test_predicted,save,save_folder_name)


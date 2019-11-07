#######################################
# Takes initial hdf5 file and plots start time and charge
#   Plots all doms for all events
#   Slow way to do this...
#   There are better ways to do this...
#####################################


import numpy as np
import h5py
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_file",type=str,
                    default='/mnt/scratch/micall12/training_files/Level5_IC86.2013_genie_numu.014640.100.hdf5',
                    dest="input_file", help="input file (including path)")
args = parser.parse_args()
input_file = args.input_file
plot = False

f = h5py.File(input_file, "r")
features_DC = f["features_DC"][:]
features_IC = f["features_IC"][:]
labels = f["labels"][:]
f.close()
del f

DC_num_events = features_DC.shape[0]
if DC_num_events != features_IC.shape[0]:
    print("DC not equal to IC")
if DC_num_events != labels.shape[0]:
    print("DC not equal to Labels")


# Inforation: sum charges, sum charge <500ns, sum charge <100ns, time first pulse, time when 20 % of charge, time when 50% charge, Time of last pulse, Charge weighted mean time of pulses, Charge weighted standard deviation of pulse times

# features_DC[event, string #, DOM #, parameters]
# features_DC[index, index2, index3, 3 = time first pulse]

def GetPulseData(features):
    num_events = features.shape[0]
    time_array = []
    charge_array = []
    for index in range(0,num_events):
        for index2 in range(0,features.shape[1]):
            for index3 in range(0,features.shape[2]):
                start_time = features[index,index2, index3, 3]
                charge = features[index,index2,index3,0]
                #if start_time > 0:
                time_array.append(start_time)
                charge_array.append(charge)
    return time_array, charge_array

time_array, charge_array  = GetPulseData(features_DC)
# Labels: [energy, zenith, azimuth, time, track length, vertex x, vertex y, vertex z]
print(labels.shape)
time_vertex = labels[:,3]
energy = labels[:,0]

if plot:

    plt.figure()
    plt.hist(time_array,bins=20,histtype='step',linewidth=2,fill=False)
    plt.title("Pulse \"Start\" Times > 0")
    plt.xlabel("time stamp")

    plt.figure()
    plt.hist(time_vertex,bins=20,histtype='step',linewidth=2,fill=False)
    plt.title("True Vertex Times")
    plt.xlabel("time stamp")
    
    plt.figure()
    plt.hist(energy,bins=20,histtype='step',linewidth=2,fill=False)
    plt.title("True Energy")
    plt.xlabel("energy (GeV)")

    plt.show()

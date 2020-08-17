#################################
# Plots input and output features for ONE file
#   Inputs:
#       -i input file:  name of ONE file
#       -d  path:       path to input files
#       -o  outdir:     path to output_plots directory or where final dir will be created
#       -n  name:       Name of directory to create in outdir (associated to filenames)
#       --emax:         Energy max cut, plot events below value, used for UN-TRANSFORM
#       --emin:         Energy min cut, plot events above value
#       --tmax:         Track factor to multiply, use for UN-TRANSFORM
#   Outputs:
#       File with count in each bin
#       Histogram plot with counts in each bin
#################################

import numpy as np
import h5py
import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_file",default=None,
                    type=str,dest="input_file", help="name for ONE input file")
parser.add_argument("-d", "--path",type=str,default='/mnt/scratch/micall12/training_files/',
                    dest="path", help="path to input files")
parser.add_argument("-o", "--outdir",type=str,default='/mnt/home/micall12/LowEnergyNeuralNetwork/output_plots/',
                    dest="outdir", help="out directory for plots")
parser.add_argument("-n", "--name",default=None,
                    dest="name", help="name for output folder")
parser.add_argument("--filenum",default=None,
                    dest="filenum", help="number for file, if multiple with same name")
parser.add_argument("--emax",type=float,default=100.0,
                    dest="emax",help="Cut anything greater than this energy (in GeV)")
parser.add_argument("--emin",type=float,default=5.0,
                    dest="emin",help="Cut anything less than this energy (in GeV)")
parser.add_argument("--tmax",type=float,default=200.0,
                    dest="tmax",help="Multiplication factor for track length")
parser.add_argument("-c", "--cuts",type=str, default="CC",
                    dest="cuts", help="Type of events to keep (all, cascade, track, CC, NC, etc.)")
parser.add_argument("--do_cuts", default=False,action='store_true',
                        dest='do_cuts',help="Apply cuts! Don't ignore energy and event cut")
args = parser.parse_args()

input_file = args.path + args.input_file
output_path = args.outdir
name = args.name
outdir = output_path + name
if os.path.isdir(outdir) != True:
    os.mkdir(outdir)
print("Saving plots to %s"%outdir)

if args.filenum:
    filenum = str(args.filenum)
else:
    filenum=args.filenum
energy_min = args.emin
energy_max = args.emax
track_max = args.tmax
cut_name = args.cuts
do_cuts = args.do_cuts

# Here in case you want only one (input can take a few min)
do_output = True
do_input = True
file_was_transformed = True

f = h5py.File(input_file, 'r')
if file_was_transformed:
    Y_test = f['Y_test'][:]
    X_test_DC = f['X_test_DC'][:]
    X_test_IC = f['X_test_IC'][:]
else:
    Y_test = f['labels'][:]
    X_test_DC = f['features_DC'][:]
    X_test_IC = f['features_IC'][:]

try:
    Y_train = f['Y_train'][:]
    X_train_DC = f['X_train_DC'][:]
    X_train_IC = f['X_train_IC'][:]

    Y_validate = f['Y_validate'][:]
    X_validate_DC = f['X_validate_DC'][:]
    X_validate_IC = f['X_validate_IC'][:]
except:    
    Y_train = None
    X_train_DC = None
    X_train_IC = None

    Y_validate = None
    X_validate_DC = None
    X_validate_IC = None

try:
    reco_test = f['reco_test'][:]
    reco_train = f['reco_train'][:]
    reco_validate = f['reco_validate'][:]
except:
    reco_test = None
    reco_train = None
    reco_validate = None

f.close()
del f

if Y_train is None: #Test only file
    Y_labels = Y_test
    X_DC = X_test_DC
    X_IC = X_test_IC
else:
    Y_labels = np.concatenate((Y_test,Y_train,Y_validate))
    X_DC = np.concatenate((X_test_DC,X_train_DC,X_validate_DC))
    X_IC = np.concatenate((X_test_IC,X_train_IC,X_validate_IC))
print(Y_labels.shape,X_DC.shape,X_IC.shape)


# Untransform so energy and track are in original range. NOTE: zenith still cos(zenith)
if file_was_transformed:
    print("MULTIPLYING ENERGY BY %f and TRACK BY %f to undo transform"%(energy_max,track_max))
    print("ASSUMING TRACK IS AT INDEX 2")
    Y_labels[:,0] = Y_labels[:,0]*energy_max
    Y_labels[:,2] = Y_labels[:,2]*track_max

if reco_test is not None:
    reco_labels = np.concatenate((reco_test,reco_train,reco_validate))

# Apply Cuts
from handle_data import CutMask
if do_cuts:
    print("CUTTING ON ENERGY [%f,%f] AND EVENT TYPE %s"%(energy_min,energy_max,cut_name))
    mask = CutMask(Y_labels)
    cut_energy = np.logical_and(Y_labels[:,0] > energy_min, Y_labels[:,0] <energy_max)
    all_cuts = np.logical_and(mask[cut_name], cut_energy)
    Y_labels = Y_labels[all_cuts]
    X_DC = X_DC[all_cuts]
    X_IC = X_IC[all_cuts]
    if reco_test is not None:
        reco_labels = reco_labels[all_cuts]


def plot_output(Y_values,outdir,filenumber=None,transformed=file_was_transformed):
    if file_was_transformed:
        names = ["Energy", "Cosine Zenith", "Track Length", "Time", "X", "Y", "Z", "Azimuth"]
        units = ["(GeV)", "", "(m)", "(s)", "(m)", "(m)", "(m)", "(rad)"]
    else:
        names = ["Energy", "Cosine Zenith", "Azimuth", "Time", "X", "Y", "Z", "Track Length"]
        units = ["(GeV)", "", "(rad)", "(s)", "(m)", "(m)", "(m)", "(m)"]
    for i in range(0,len(names)):
        plt.figure()
        plt.hist(Y_values[:,i],bins=100);
        plt.title("%s Distribution"%names[i],fontsize=25)
        plt.xlabel("%s %s"%(names[i],units[i]),fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        if filenum:
            filenum_name = "_%s"%filenum
        else:
            filenum_name = ""
        plt.savefig("%s/Output_%s%s.png"%(outdir,names[i].replace(" ", ""),filenum_name))
    
    num_events = Y_values.shape[0]
    flavor = list(Y_values[:,9])
    print("Fraction NuMu: %f"%(flavor.count(14)/num_events))
    print("Fraction Track: %f"%(sum(Y_values[:,8])/num_events))
    print("Fraction Antineutrino: %f"%(sum(Y_values[:,10])/num_events))
    print("Fraction CC: %f"%(sum(Y_values[:,11])/num_events))

def plot_energy_zenith(Y_values,outdir,filenumber=None):
    plt.figure()
    cts,xbin,ybin,img = plt.hist2d(Y_values[:,0], Y_values[:,1], bins=100)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('counts', rotation=90)
    plt.set_cmap('ocean_r')
    plt.xlabel("True Neutrino Energy (GeV)",fontsize=15)
    plt.ylabel("True Neutrino Cosine Zenith",fontsize=15)
    plt.title("True Energy vs Cosine Zenith Distribution",fontsize=20)
    plt.savefig("%s/EnergyZenith%s.png"%(outdir,filenum))

def plot_input(X_values_DC,X_values_IC,outdir,filenumber=None):
    name = ["Charge (p.e.)", "Raw Time of First Pulse (ns)", "Raw Time of Last Pulse (ns)", "Charge weighted mean of pulse times", "Charge weighted std of pulse times"]
    IC_label = "IC"
    DC_label = "DC"
    
    print(X_values_DC.shape,X_values_IC.shape)
    for i in range(0,X_values_DC.shape[-1]):

        DC_data = X_values_DC[...,i].flatten()
        IC_data = X_values_IC[...,i].flatten()

        min_range = min(min(DC_data),min(IC_data))
        max_range = max(max(DC_data),max(IC_data))
        plt.figure()
        plt.hist(IC_data,log=True,bins=100,range=[min_range,max_range],color='g',label=IC_label,alpha=0.5);
        plt.hist(DC_data,log=True,bins=100,range=[min_range,max_range],color='b',label=DC_label,alpha=0.5);
        plt.title(name[i],fontsize=25)
        plt.xlabel(name[i],fontsize=15)
        plt.legend(fontsize=15)
        if filenum:
            filenum_name = "_%s"%filenum
        else:
            filenum_name = ""
        plt.savefig("%s/Input_Variable%i%s.png"%(outdir,i,filenum_name))

if do_output:
    plot_output(Y_labels,outdir,filenumber=filenum)
    if reco_test is not None:
        plot_output(reco_labels,outdir,filenumber="%s_reco"%filenum)
if do_input:
    plot_input(X_DC,X_IC,outdir,filenumber=filenum)

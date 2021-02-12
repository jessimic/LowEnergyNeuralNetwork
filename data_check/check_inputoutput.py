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
parser.add_argument("--transformed",default=False,action='store_true',
                    dest="transformed", help="add flag if labels truth input is already transformed")
parser.add_argument("--no_input",default=True,action='store_false',
                    dest="no_input", help="add flag if you don't want to plot input")
parser.add_argument("--no_output",default=True,action='store_false',
                    dest="no_output", help="add flag if you don't want to plot output")
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
do_output = args.no_output
do_input = args.no_input
file_was_transformed = args.transformed

f = h5py.File(input_file, 'r')
try:
    Y_test = f['Y_test'][:]
except:
    try:
        Y_test = f['Y_test_use'][:]
    except:
        Y_test = f['labels'][:]
    
try:
    X_test_DC = f['X_test_DC'][:]
    X_test_IC = f['X_test_IC'][:]
except:
    try:
        X_test_DC = f['features_DC'][:]
        X_test_IC = f['features_IC'][:]
    except:
        print("No input features in file")
        X_test_DC = np.array([np.nan])
        X_test_IC = np.array([np.nan]) 

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
except:
    reco_test = None
try:
    reco_train = f['reco_train'][:]
    reco_validate = f['reco_validate'][:]
except:
    reco_train = None
    reco_validate = None

try:
    weights_test = f['weights_test'][:]
except:
    wegiths_test = None
try:
    weights_train = f['weights_train'][:]
    weights_validate = f['weights_validate'][:]
except:
    weights_train = None
    weightsvalidate = None

try:
    label_names = f['output_label_names'][:]
    output_names = [n.decode('utf8') for n in label_names]
    output_factors = f['output_transform_factors'][:]
    input_factors = [25., 4000., 4000., 4000., 2000.] #f['input_transform_factors'][:]
    print("USING HARD CODED INPUT FACTORS")
    print("MULTIPLYING ENERGY BY %f and TRACK BY %f to undo transform"%(output_factors[0],track_max))
    print("ASSUMING TRACK IS AT INDEX 2")
except:
    if file_was_transformed:
        output_names = ["Energy", "Cosine Zenith", "Track Length", "Time", "X", "Y", "Z", "Azimuth", "Zenith"]
        output_factors = [emax, 1., tmax, 1., 1., 1., 1., 1., 1.]
        input_factors = [25., 4000., 4000., 4000., 2000.] 
        print("MULTIPLYING ENERGY BY %f and TRACK BY %f to undo transform"%(energy_max,track_max))
        print("ASSUMING TRACK IS AT INDEX 2")
    else:
        output_names = ["Energy", "Cosine Zenith", "Azimuth", "Time", "X", "Y", "Z", "Track Length", "Zenith"]
        output_factors = [1., 1., 1., 1., 1., 1., 1., 1., 1.]
        input_factors = [1., 1., 1., 1., 1.] 


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
print(len(output_factors))

reco_labels = reco_test
if reco_train is not None:
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
    if reco_labels is not None:
        reco_labels = reco_labels[all_cuts]


def plot_output(Y_values,outdir,filenumber=None,names=output_names,transform=output_factors,weights=None):
    if file_was_transformed:
        units = ["(GeV)", "", "(m)", "(s)", "(m)", "(m)", "(m)", "(rad)", "(rad)"]
    else:
        units = ["(GeV)", "", "(rad)", "(s)", "(m)", "(m)", "(m)", "(m)", "(rad)"]
    
    plt.figure()
    plt.hist(Y_values[:,0]*transform[0],bins=100,weights=weights);
    plt.title("%s Distribution"%names[0],fontsize=25)
    plt.xlabel("%s %s"%(names[0],units[0]),fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    #plt.yscale('log')
    if filenum:
        filenum_name = "_%s"%filenum
    else:
        filenum_name = ""
    #plt.savefig("%s/Output_%s%s.png"%(outdir,names[0].replace(" ", ""),filenum_name))
    plt.savefig("%s/Output_Energy%s.png"%(outdir,filenum_name),bbox_inches='tight')
    
    plt.figure()
    plt.hist(Y_values[:,1]*transform[1],bins=100,weights=weights);
    plt.title("%s Distribution"%names[1],fontsize=25)
    plt.xlabel("%s %s"%(names[1],units[1]),fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    if filenum:
        filenum_name = "_%s"%filenum
    else:
        filenum_name = ""
    #plt.savefig("%s/Output_%s%s.png"%(outdir,names[1].replace(" ", ""),filenum_name))
    plt.savefig("%s/Output_CosZenith%s.png"%(outdir,filenum_name),bbox_inches='tight')
    
    plt.figure()
    plt.hist(Y_values[:,2]*transform[2],bins=100,weights=weights);
    plt.title("%s Distribution"%names[2],fontsize=25)
    plt.xlabel("%s %s"%(names[2],units[2]),fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.yscale('log')
    if filenum:
        filenum_name = "_%s"%filenum
    else:
        filenum_name = ""
    #plt.savefig("%s/Output_%s%s.png"%(outdir,names[1].replace(" ", ""),filenum_name))
    plt.savefig("%s/Output_TrackLength%s.png"%(outdir,filenum_name),bbox_inches='tight')

    if Y_values.shape[-1] == 13:
        plt.figure()
        plt.hist(Y_values[:,12],bins=100,weights=weights);
        plt.title("Zenith Distribution",fontsize=25)
        plt.xlabel("Zenith %s"%(units[-1]),fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        if filenum:
            filenum_name = "_%s"%filenum
        else:
            filenum_name = ""
        #plt.savefig("%s/Output_%s%s.png"%(outdir,names[12].replace(" ", ""),filenum_name))
        plt.savefig("%s/Output_Zenith%s.png"%(outdir,filenum_name),bbox_inches='tight')


    row_index = 0
    col_index = 0
    rows = 3
    cols = 3
    fig, ax = plt.subplots(rows,cols,figsize=(15,15))
    for i in range(0,len(units)):
        if i == 8:
            try:
                values = Y_values[:,12]
                a_name = "Zenith"
            except:
                break
        else:
            values = Y_values[:,i]*transform[i]
            a_name = names[i]
        ax[row_index, col_index].hist(values,bins=100,weights=weights);
        ax[row_index, col_index].set_title("%s Distribution"%a_name,fontsize=15)
        ax[row_index, col_index].set_xlabel("%s %s"%(a_name,units[i]),fontsize=15)
        #ax[row_index, col_index].set_xticks(fontsize=15)
        #ax[row_index, col_index].set_yticks(fontsize=15)
        if col_index < (cols-1):
            col_index += 1
        else:
            row_index += 1
            col_index = 0
    plt.suptitle("Output Distributions %s"%filenum,fontsize=25)
    if filenum:
        filenum_name = "_%s"%filenum
    else:
        filenum_name = ""
    plt.savefig("%s/AllOutput%s.png"%(outdir,filenum_name))
    
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
    plt.savefig("%s/EnergyCosZenith%s.png"%(outdir,filenum))

    if Y_values.shape[-1] ==13:
        plt.figure()
        cts,xbin,ybin,img = plt.hist2d(Y_values[:,0], Y_values[:,12], bins=100)
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('counts', rotation=90)
        plt.set_cmap('ocean_r')
        plt.xlabel("True Neutrino Energy (GeV)",fontsize=15)
        plt.ylabel("True Neutrino Zenith (rad)",fontsize=15)
        plt.title("True Energy vs Zenith Distribution",fontsize=20)
        plt.savefig("%s/EnergyZenith%s.png"%(outdir,filenum))

def plot_input(X_values_DC,X_values_IC,outdir,filenumber=None,transform=input_factors):
    name = ["Charge/25 (p.e./25)", "Time of First Pulse/4000 (ns/4000)", "Time of Last Pulse/4000 (ns/4000)", "Charge weighted mean of pulse times/4000", "Charge weighted std of pulse times/2000"]
    IC_label = "IC"
    DC_label = "DC"
    
    print(X_values_DC.shape,X_values_IC.shape)

    row_index = 0
    col_index = 0
    rows = 2
    cols = 3
    fig, ax = plt.subplots(rows,cols,figsize=(15,10))
    for i in range(0,X_values_DC.shape[-1]):

        DC_data = X_values_DC[...,i].flatten()
        IC_data = X_values_IC[...,i].flatten()

        min_range = min(min(DC_data),min(IC_data))
        max_range = max(max(DC_data),max(IC_data))
        #plt.figure()
        ax[row_index, col_index].hist(IC_data,log=True,bins=100,range=[min_range,max_range],color='g',label=IC_label,alpha=0.5);
        ax[row_index, col_index].hist(DC_data,log=True,bins=100,range=[min_range,max_range],color='b',label=DC_label,alpha=0.5);
        ax[row_index, col_index].set_title(name[i],fontsize=15)
        ax[row_index, col_index].set_xlabel(name[i],fontsize=15)
        ax[row_index, col_index].legend(fontsize=15)
        if col_index < (cols-1):
            col_index += 1
        else:
            row_index += 1
            col_index = 0
    plt.suptitle("Transformed Input Distributions %s"%filenum,fontsize=25)
    if filenum:
        filenum_name = "_%s"%filenum
    else:
        filenum_name = ""
    plt.savefig("%s/AllInput_%s.png"%(outdir,filenum_name))

if do_output:
    plot_output(Y_labels,outdir,filenumber=filenum,names=output_names,weights=None) #weights=weights_test[:,8]/1510.
    print("HARD CODED WEIGHTS!!!!!!!!!!!!")
    #if reco_test is not None:
    #    plot_output(reco_labels,outdir,filenumber="%s_reco"%filenum)
if do_input:
    plot_input(X_DC,X_IC,outdir,filenumber=filenum,transform=input_factors)

#################################
# Creates a flat (in energy) sample, if possible
# Uses the "max per bin" request to level out events per bin
# Handles any size bins, suggested 1-2 GeV
#   Inputs:
#       -i input files: name of file (can use * and ?)
#       -d path:        path to input files
#       -o ouput:       name of output file, placed in path directory
#       -r reco:        True if file has old reco (Pegleg) array
#       -b bin_size:    Size of bin in energy (GeV)
#       --max_per_bin:  number of events per bin you want
#       --emax:         maximum energy to cut at (keeps everything below)
#       --cutDC:        True will apply cut so only events with vertex in DeepCore are kept
#       --shuffle:      True will shuffle events before saving
#################################

import numpy as np
import glob
import h5py
import argparse
import matplotlib.pyplot as plt
from handle_data import CutMask

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_files",default='Level5_IC86.2013_genie_numu.014640.*.hdf5',
                    type=str,dest="input_files", help="names for input files")
parser.add_argument("--add_file",default=None,
                    type=str,dest="add_file", help="different named file to add")
parser.add_argument("-d", "--path",type=str,default='/mnt/scratch/micall12/training_files/',
                    dest="path", help="path to input files")
parser.add_argument("-o", "--output",type=str,default='Level5_IC86.2013_genie_numu.014640.energy_sorted',
                    dest="output", help="names for output files")
parser.add_argument("-r", "--reco",type=str,default="False",
                    dest="reco", help="True if using Level5p or have a pegleg reco")
parser.add_argument("-b", "--bin_size",type=float,default=1.,
                    dest="bin_size", help="Size of energy bins in GeV (default = 1GeV)")
parser.add_argument("--max_per_bin",type=int,default=10000,
                    dest="max_evt_per_bin",help="Max number of energy events per bin")
parser.add_argument("--emax",type=float,default=200.0,
                    dest="emax",help="Value to perform energy cut (less than or equal to this value)")
parser.add_argument("--cutDC",type=str,default=False,
                    dest="cutDC",help="Do you want to save only vertex in DC events")
parser.add_argument("--shuffle",type=str, default=False,
                    dest="shuffle", help="True if you want to shuffle")
parser.add_argument("-c", "--cuts",type=str, default="all",
                    dest="cuts", help="Type of events to keep (all, cascade, track, CC, NC, etc.)")
args = parser.parse_args()
input_files = args.input_files
path = args.path
output = args.output
bin_size = args.bin_size
max_events_per_bin = args.max_evt_per_bin
add_file = args.add_file
emax = args.emax
cut_name = args.cuts
print("Keeping %s event types"%cut_name)
if args.reco == 'True' or args.reco == 'true':
    use_old_reco = True
    print("Expecting old reco values in files, pulling from pegleg frames")
else:
    use_old_reco = False
if args.cutDC == 'True' or args.cutDC == 'true':
    cut_DC = True
else:
    cut_DC = False
if args.shuffle == "True" or args.shuffle == "true":
    shuffle = True
else:
    shuffle = False

file_names = path + input_files
event_file_names = sorted(glob.glob(file_names))
if add_file:
    print("Adding this file %s"%add_file)
    event_file_names.insert(0,path + add_file)
assert event_file_names,"No files loaded, please check path."

full_features_DC = None
full_features_IC = None
full_labels = None
full_reco = None

bins = int(emax/float(bin_size))
if emax%bin_size !=0:
    bins +=1 #Put remainder into additional bin
count_energy = np.zeros((bins))
max_per_bin = max_events_per_bin
count_no_save = 0
for a_file in event_file_names:

    f = h5py.File(a_file, "r")
    file_features_DC = f["features_DC"][:]
    file_features_IC = f["features_IC"][:]
    file_labels = f["labels"][:]
    if use_old_reco:
        file_reco_labels = f["reco_labels"][:]
    f.close()
    del f

    mask = CutMask(file_labels)

    if cut_DC == True:
        nu_x = file_labels[:,4]
        nu_y = file_labels[:,5]
        nu_z = file_labels[:,6]
        radius = 90
        x_origin = 54
        y_origin = -36

    energy = file_labels[:,0]
    keep_index = [False]*len(energy)
    print("Total events this file: %i"%len(energy))

    # Check how many events already in each bin, save if under max
    for index,e in enumerate(energy):
        if e > emax:
            continue
        
        if cut_DC == True:
            shift_x = nu_x[index] - x_origin
            shift_y = nu_y[index] - y_origin
            z_val = nu_z[index]
            radius_calculation = np.sqrt(shift_x**2+shift_y**2)
            if( radius_calculation > radius or z_val > 192 or z_val < -505 ):
                continue

        if mask[cut_name][index] == False:
            continue

        e_bin = int(e/float(bin_size))
        if count_energy[e_bin] < max_per_bin:
            keep_index[index] = True
            count_energy[e_bin] += 1

    keep_index = np.array(keep_index)

    if full_features_DC is None:
        full_features_DC = file_features_DC[keep_index]
    else:
        full_features_DC = np.concatenate((full_features_DC, file_features_DC[keep_index]))
    
    if full_features_IC is None:
        full_features_IC = file_features_IC[keep_index]
    else:
        full_features_IC = np.concatenate((full_features_IC, file_features_IC[keep_index]))

    if full_labels is None:
        full_labels = file_labels[keep_index]
    else:
        full_labels = np.concatenate((full_labels, file_labels[keep_index]))

    if use_old_reco:
        if full_reco is None:
            full_reco = file_reco[keep_index]
        else:
            full_reco = np.concatenate((full_reco, file_reco[keep_index]))

    print("Total events saved: %i, Saved this file: %i, Finsihed file: %s"%(sum(count_energy),sum(keep_index),a_file))

    if sum(keep_index) == 0:
        count_no_save += 1

    #quit_files = 20
    #if count_no_save > quit_files:
    #    print("Haven't seen any new events in %i files, quitting..."%quit_files)
    #    break

    if np.all(count_energy >= max_per_bin):
        break
    else:
        print(count_energy)

if shuffle == True:
    print("Finished concatonating all the files. Now I will shuffle..")
    from handle_data import Shuffler
    shuffled_features_DC, shuffled_features_IC, shuffled_labels, \
    shuffled_reco, shuffled_initial_stats, shuffled_num_pulses = \
    Shuffler(full_features_DC,full_features_IC,full_labels, \
    full_reco, use_old_reco_flag=use_old_reco)

    full_features_DC = shuffled_features_DC
    full_features_IC = shuffled_features_IC
    full_labels      = shuffled_labels
    if use_old_reco:
        full_reco    = shuffled_reco



#Save output to hdf5 file
print(count_energy)
output_name = path + output + "flat_%sbins_%sevtperbin.hdf5"%(bins,max_events_per_bin)
print(output_name)
f = h5py.File(output_name, "w")
f.create_dataset("features_DC", data=full_features_DC)
f.create_dataset("features_IC", data=full_features_IC)
f.create_dataset("labels", data=full_labels)
if use_old_reco:
    f.create_dataset("reco_labels", data=reco_labels)
f.close()

#################################
# Checks number of events in energy bin from hdf5 training data sets
#   Inputs:
#       -i input files: name of files (can use * and ?)
#       -d  path:       path to input files
#       -o  outdir:     path to output_plots directory or where final dir will be created
#       -n  name:       Name of directory to create in outdir (associated to filenames)
#       -c  cuts:       name of cuts you want to apply (i.e. track only = track)
#       --emax:         Energy max cut, keep all events below value
#       --emin:         Energy min cut, keep all events above value
#       --tmax:         Track factor to multiply, only used IF TRANFORMED IS TRUE
#       --transformed:  use flag if file has already been transformed
#       --labels:       name of truth array to load (labels, Y_test, Y_train, etc.)
#       --bin_size:     Size (in GeV) for bins to distribute energy into
#       --start:        Name of vertex start cut
#       --end:          Name of ending position cut
#   Outputs:
#       File with count in each bin
#       Histogram plot with counts in each bin
#################################

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import glob
import os
from handle_data import CutMask
from handle_data import VertexMask

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_files",default=None,
                    type=str,dest="input_files", help="names for input files")
parser.add_argument("-d", "--path",type=str,default='/mnt/scratch/micall12/training_files/',
                    dest="path", help="path to input files")
parser.add_argument("-o", "--outdir",type=str,default='/mnt/home/micall12/LowEnergyNeuralNetwork/output_plots/',
                    dest="outdir", help="out directory for plots")
parser.add_argument("-n", "--name",type=str,default="None",
                    dest="name", help="name for output folder")
parser.add_argument("-b", "--bin_size",type=float,default=1.,
                    dest="bin_size", help="Size of energy bins in GeV (default = 1GeV)")
parser.add_argument("--emax",type=float,default=100.0,
                    dest="emax",help="Cut anything greater than this energy (in GeV)")
parser.add_argument("--emin",type=float,default=5.0,
                    dest="emin",help="Cut anything less than this energy (in GeV)")
parser.add_argument("--efactor",type=float,default=100,
                    dest="efactor",help="Multiplication factor for energy")
parser.add_argument("--tfactor",type=float,default=200,
                    dest="tfactor",help="Multiplication factor for track")
parser.add_argument("-c", "--cuts",type=str, default="CC",
                    dest="cuts", help="Type of events to keep (all, cascade, track, CC, NC, etc.)")
parser.add_argument("--labels",type=str,default="labels",
                    dest="labels", help="name of truth array to read in from input files")
parser.add_argument("--transformed",default=False,action='store_true',
                    dest="transformed", help="add flag if labels truth input is already transformed")
parser.add_argument("-s", "--start",type=str, default="all_start",
                    dest="start_cut", help="Vertex start cut (all_start, old_start_DC, start_DC, start_IC, start_IC19)")
parser.add_argument("-e", "--end",type=str, default="all_end",
                    dest="end_cut", help="End position cut (end_start, end_IC7, end_IC19)")
parser.add_argument("--apply_cuts",default=False,action='store_true',
                    dest="apply_cuts", help="add flag if you actually want to cut on energy, cuts, and vertex")
args = parser.parse_args()

input_files = args.input_files
files_with_paths = args.path + input_files
event_file_names = sorted(glob.glob(files_with_paths))
assert event_file_names,"No files loaded, please check path."

apply_cuts = args.apply_cuts
if apply_cuts:
    print("CUTTING ON ENERGY, CUTS, and VERTEX GIVEN")
else:
    print("NO CUTS APPLIED")

emax = args.emax
emin = args.emin
efactor = args.efactor
tfactor = args.tfactor
bin_size = args.bin_size
cut_name = args.cuts
energy_bin_array = np.arange(emin,emax,bin_size)
truth_name = args.labels
start_cut = args.start_cut
end_cut = args.end_cut
transformed = args.transformed

azimuth_index = 2
track_index = 7


print("Cutting Emax %.f, emin %.f, with event type %s, start cut: %s and end cut: %s"%(emax,emin,cut_name,start_cut,end_cut))
    
if args.name is "None":
    file_name = event_file_names[0].split("/")
    output = file_name[-1][:-5]
else:
    output = args.name
outdir = args.outdir + output
if os.path.isdir(outdir) != True:
    os.mkdir(outdir)

count_events = 0
file_count = 0

print("Evts in File\t After Cut (if applied)")
for a_file in event_file_names:
    
    ### Import Files ###
    f = h5py.File(a_file, 'r')
    file_labels = f[truth_name][:]
    f.close()
    del f
    
    if file_labels.shape[0] == 0:
        print("Empty file...skipping...")
        continue

    if apply_cuts:
        if transformed:
            azimuth_indx = 7
            track_index = 2
            file_labels[:,0] = file_labels[:,0]*efactor
            file_labels[:,1] = np.arccos(file_labels[:,1])
            file_labels[:,track_index] = file_labels[:,0]*tfactor

        
        type_mask = CutMask(file_labels)
        vertex_mask = VertexMask(file_labels,azimuth_index=azimuth_index,track_index=track_index,max_track=tfactor)
        vertex_cut = np.logical_and(vertex_mask[start_cut], vertex_mask[end_cut])
        energy_cut = np.logical_and(emin, emax)
        mask = np.logical_and(np.logical_and(type_mask[cut_name], vertex_cut), energy_cut)
    else:
        mask = np.ones(file_labels.shape[0])

    mask = np.array(mask,dtype=bool)

    # Make cuts for event type and energy
    energy = np.array(file_labels[:,0])
    total_events = len(energy)
    energy = energy[mask]
    events_after_cut = len(energy)
    if file_count < 10:
        print("%i\t %i\t"%(total_events,events_after_cut))

    #Sort into bins and count
    count_events += events_after_cut
    file_count += 1
    if file_count%100 == 0 and file_count > 0:
        print("Gone through %i files so far"%file_count)
    
print("Total events (after cuts, if applied): %i"%(count_events))

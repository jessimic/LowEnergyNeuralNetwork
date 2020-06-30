#################################
# Creates a flat (in energy) sample, if possible
# Uses the "max per bin" request to level out events per bin
# Handles any size bins, suggested 1-2 GeV
#   Inputs:
#       -i input files: name of file (can use * and ?)
#       --add_file:     option to add ONE file of different pattern
#       -d path:        path to input files
#       -o ouput:       name of output file, placed in path directory
#       -r reco:        True if file has old reco (Pegleg) array
#       -b bin_size:    Size of bin in energy (GeV)
#       --max_per_bin:  number of events per bin you want
#       --emax:         maximum energy to cut at (keeps everything below)
#       --emin:         minimum energy to cut at (keeps everything above)
#       --start:        Name of vertex start cut
#       --end:          Name of ending position cut
#       --shuffle:      True will shuffle events before saving
#       --num_out:      number of output files to split output into (default = 1, i.e. no split)
#################################

import numpy as np
import glob
import h5py
import argparse
from handle_data import CutMask
from handle_data import VertexMask

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_files",default=None,
                    type=str,dest="input_files", help="names for input files")
parser.add_argument("--add_file",default=None,
                    type=str,dest="add_file", help="different named file to add")
parser.add_argument("-d", "--path",type=str,default='/mnt/scratch/micall12/training_files/',
                    dest="path", help="path to input files")
parser.add_argument("-o", "--output",type=str,default='flat_energy_distribution',
                    dest="output", help="names for output files")
parser.add_argument("-r", "--reco",type=str,default="False",
                    dest="reco", help="True if using Level5p or have a pegleg reco")
parser.add_argument("-b", "--bin_size",type=float,default=1.,
                    dest="bin_size", help="Size of energy bins in GeV (default = 1GeV)")
parser.add_argument("--max_per_bin",type=int,default=10000,
                    dest="max_evt_per_bin",help="Max number of energy events per bin")
parser.add_argument("--emax",type=float,default=200.0,
                    dest="emax",help="Cut anything greater than this energy (in GeV)")
parser.add_argument("--emin",type=float,default=0.0,
                    dest="emin",help="Cut anything less than this energy (in GeV)")
parser.add_argument("--shuffle",type=str, default=True,
                    dest="shuffle", help="True if you want to shuffle")
parser.add_argument("-c", "--cuts",type=str, default="all",
                    dest="cuts", help="Type of events to keep (all, cascade, track, CC, NC, etc.)")
parser.add_argument("-s", "--start",type=str, default="all_start",
                    dest="start_cut", help="Vertex start cut (all_start, old_start_DC, start_DC, start_IC, start_IC19)")
parser.add_argument("-e", "--end",type=str, default="all_end",
                    dest="end_cut", help="End position cut (end_start, end_IC7, end_IC19)")
parser.add_argument("--num_out",type=int,default=1,
                    dest="num_out",help="number of output files you want to split the output into")
args = parser.parse_args()

input_files = args.input_files
path = args.path
output = args.output
add_file = args.add_file
num_outputs = args.num_out

bin_size = args.bin_size
max_events_per_bin = args.max_evt_per_bin

emax = args.emax
emin = args.emin
cut_name = args.cuts

# Vertex Cuts (no transformation done yet)
start_cut = args.start_cut
end_cut = args.end_cut
azimuth_index = 2
track_index = 7
track_max = 1.0

print("Keeping %s event types"%cut_name)
if args.reco == 'True' or args.reco == 'true':
    use_old_reco = True
    print("Expecting old reco values in files, pulling from pegleg frames")
else:
    use_old_reco = False
if args.shuffle == "False" or args.shuffle == "false":
    shuffle = False
    assert num_outputs==1, "MUST SHUFFLE IF NUMBER OUTPUT FILES > 1!"
else:
    shuffle = True

print("Saving PEGLEG info: %s \nBin size: %f GeV \nMax Count Per Bin: %i events \nEnergy Range: %f - %f \nStarting cut: %s \nEnding cut: %s \nShuffling: %s \nKeeping event types: %s"%(use_old_reco,bin_size,max_events_per_bin,emin,emax,start_cut,end_cut,shuffle,cut_name)) 

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
full_stats = None
full_pulses_per_dom = None
full_trig_times = None

bins = int((emax-emin)/float(bin_size))
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
    file_stats = f["initial_stats"][:] 
    file_pulses_per_dom = f["num_pulses_per_dom"][:]
    try:
        file_trig_times = f["trigger_times"][:]
    except:
        file_trig_times = None
    if use_old_reco:
        file_reco_labels = f["reco_labels"][:]
    f.close()
    del f
   
    if file_labels.shape[0] == 0:
        print("Empty file...skipping...")
        continue
    
    # Applying cuts
    type_mask = CutMask(file_labels)
    vertex_mask = VertexMask(file_labels,azimuth_index=azimuth_index,track_index=track_index,max_track=track_max)
    vertex_cut = np.logical_and(vertex_mask[start_cut], vertex_mask[end_cut])
    mask = np.logical_and(type_mask[cut_name], vertex_cut)
    mask = np.array(mask,dtype=bool)

    energy = file_labels[:,0]
    keep_index = [False]*len(energy)
    print("Total events this file: %i"%len(energy))

    # Check how many events already in each bin, save if under max
    for index,e in enumerate(energy):
        if e > emax:
            continue
        if e < emin:
            continue

        if mask[index] == False:
            continue

        e_bin = int((e-emin)/float(bin_size))
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
    
    if full_stats is None:
        full_stats = file_stats[keep_index]
    else:
        full_stats = np.concatenate((full_stats, file_stats[keep_index]))
    
    if full_pulses_per_dom is None:
        full_pulses_per_dom = file_pulses_per_dom[keep_index]
    else:
        full_pulses_per_dom = np.concatenate((full_pulses_per_dom, file_pulses_per_dom[keep_index]))
   
    if file_trig_times is not None:
        if full_trig_times is None:
            full_trig_times = file_trig_times[keep_index]
        else:
            full_trig_times = np.concatenate((full_trig_times, file_trig_times[keep_index]))

    if use_old_reco:
        if full_reco is None:
            full_reco = file_reco[keep_index]
        else:
            full_reco = np.concatenate((full_reco, file_reco[keep_index]))

    print("Total events saved: %i, Saved this file: %i, Finsihed file: %s"%(sum(count_energy),sum(keep_index),a_file))

    if sum(keep_index) == 0:
        count_no_save += 1

    quit_files = 5
    if count_no_save > quit_files:
        print("Haven't seen any new events in %i files, quitting..."%quit_files)
        break

    if np.all(count_energy >= max_per_bin):
        print("All bins filled, quitting...")
        break
    else:
        print(count_energy)

if shuffle == True:
    print("Finished concatonating all the files. Now I will shuffle..")
    from handle_data import Shuffler
    full_features_DC, full_features_IC, full_labels, \
    full_reco, full_stats, full_pulses_per_dom, full_trig_times = \
    Shuffler(full_features_DC,full_features_IC,full_labels, \
    full_reco=full_reco, full_initial_stats=full_stats, \
    full_num_pulses=full_pulses_per_dom, full_trig_times=full_trig_times, \
    use_old_reco_flag=use_old_reco)

    over_emax = full_labels[:,0] > emax
    under_emin = full_labels[:,0] < emin
    assert sum(over_emax)==0, "Have events greater than emax in final sample"
    assert sum(under_emin)==0, "Have events less than emin in final sample"
    if cut_name == "CC":
        isCC = full_labels[:,11] == 1
        assert sum(isCC)==full_labels.shape[0], "Have NC events in data"

#Save output to hdf5 file
print(count_energy)
print("Total events saved: %i"%full_features_DC.shape[0])
events_per_file = int(full_features_DC.shape[0]/num_outputs) + 1
for sep_file in range(0,num_outputs):
    start = events_per_file*sep_file
    if sep_file < num_outputs-1:
        end = events_per_file*(sep_file+1)
    else:
        end = full_features_DC.shape[0]

    if num_outputs > 1:
        filenum = "_file%02d"%sep_file
    else:
        filenum = ""
    output_name = output_name = path + output + "lt%03d_"%emax + "%s_"%cut_name + "%s_%s_"%(start_cut,end_cut) + "flat_%sbins_%sevtperbin"%(bins,max_events_per_bin) + filenum + ".hdf5"
    print("I put evnts %i - %i into %s"%(start,end,output_name))

    f = h5py.File(output_name, "w")
    f.create_dataset("features_DC", data=full_features_DC[start:end])
    f.create_dataset("features_IC", data=full_features_IC[start:end])
    f.create_dataset("labels", data=full_labels[start:end])
    if full_reco is not None:
        f.create_dataset("reco_labels",data=full_reco[start:end])
    if full_stats is not None:
        f.create_dataset("initial_stats",data=full_stats[start:end])
    if full_pulses_per_dom is not None:
        f.create_dataset("num_pulses_per_dom",data=full_pulses_per_dom[start:end])
    if full_trig_times is not None:
        f.create_dataset("trigger_times",data=full_trig_times[start:end])
    f.close()

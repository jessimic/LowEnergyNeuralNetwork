#################################
# Concatonates, cuts, and shuffles hdf5 training data sets
#   Inputs:
#       -i input files: name of file (can use * and ?)
#       -d  path:       path to input files
#       -o  output:      name of output file, placed in path directory
#       -c  cuts:       name of cuts you want to apply (i.e. track only = track)
#       -r  reco:       bool if files have pegleg reco in them
#       --num_out:      number of output files to split output into (default = 1, i.e. no split)
#       --emax:         Energy cut, keep all events below value
#       --find_minmax   Find min and max of data sets and save to output file, true by default
#       --find_quartiles    Find quartiles of darta sets and save to output file, false by default
#                            NOTE: CANT DO BOTH MINMAX AND FIND QUARTILES TRUE
#       --shuffle       True if you want to shuffle order of events, default is true
#################################

import numpy as np
import glob
import h5py
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_files",default=None,
                    type=str,dest="input_files", help="names for input files")
parser.add_argument("-d", "--path",type=str,default='/mnt/scratch/micall12/training_files/',
                    dest="path", help="path to input files")
parser.add_argument("-o", "--output",type=str,default='cut_concat_separated',
                    dest="output", help="names for output files")
parser.add_argument("-c", "--cuts", type=str, default="all",
                    dest="cuts", help="name of cuts applied (see name options on line 38)")
parser.add_argument("-r","--reco",type=str, default=False,
                    dest="reco", help="bool if the file has pegleg reco info + initial stats, etc. (if using Level5p files)")
parser.add_argument("--num_out",type=int,default=1,
                    dest="num_out",help="number of output files you want to split the output into")
parser.add_argument("--emax",type=float,default=200.0,
                    dest="emax",help="Max energy to keep, cut anything above")
parser.add_argument("--find_minmax",type=str, default=True,
                    dest="find_minmax",help="True if you want it to return min and max values from data, and save them in output file")
parser.add_argument("--find_quartiles",type=str, default=False,
                    dest="find_quartiles",help="True if you want it to return quartiles and save them in output file")
parser.add_argument("--shuffle",type=str, default=True,
                    dest="shuffle", help="False if you don't want to shuffle")
args = parser.parse_args()
input_files = args.input_files
path = args.path
output = args.output
cut_name = args.cuts
num_outputs = args.num_out
assert num_outputs<100, "NEED TO CHANGE FILENAME TO ACCOMODATE THIS MANY NUMBERS"
emax = args.emax
if args.reco == "True" or args.reco == "true":
    use_old_reco = True
    print("Expecting old reco values in file, from pegleg, etc.")
else:    
    use_old_reco = False
if args.find_quartiles == "True" or args.find_quartiles == "true":
    find_quartiles = True
else:    
    find_quartiles = False
if args.find_minmax == "False" or args.find_minmax == "false":
    find_minmax = False
else:    
    find_minmax = True
if find_minmax and find_quartiles:
    assert False, "Can't have both find_minmax and find_quartiles true. Pick for Robust or MinMax/MaxAbs scaler"
if args.shuffle == "False" or args.shuffle == "false":
    shuffle = False
else:    
    shuffle = True
# cut names: track, cascade, CC, NC, track CC, track NC, cascade CC, cascade NC 

print("Keeping %s event types"%cut_name)
print("Saving PEGLEG info: %s \nNumber output files: %i \nFindMinMax: %s Find quartiles: %s \nEnergy Max: %f GeV \nShuffling: %s \nKeeping event types: %s"%(use_old_reco,num_outputs,find_minmax,find_quartiles,emax,shuffle,cut_name)) 

file_names = path + input_files
event_file_names = sorted(glob.glob(file_names))

#event_file_names = ["/mnt/scratch/micall12/training_files/NuMu_140000_level2.zst_cleaned_lt100_CC_flat_95bins_36034evtperbinall.lt100_file00.hdf5",\
#                    "/mnt/scratch/micall12/training_files/NuMu_140000_level2.zst_cleaned_lt100_CC_flat_95bins_36034evtperbinall.lt100_file01.hdf5",\
#                    "/mnt/scratch/micall12/training_files/NuMu_140000_level2.zst_cleaned_lt100_CC_flat_95bins_36034evtperbinall.lt100_file02.hdf5",\
#                    "/mnt/scratch/micall12/training_files/NuE_120000_level2_cleaned_lt100_vertexDC_CC_flat_95bins_15478evtperbinall.allfiles.CC.lt100_file00.hdf5"]
#print("I AM USING HARDCODED FILENAMES, IGNORING YOUR INPUT ARG!!!!!!")
assert event_file_names,"No files loaded, please check path."

full_features_DC = None
full_features_IC = None
full_labels = None
full_reco = None
full_initial_stats = None
full_num_pulses = None


# Labels: [ nu energy, nu zenith, nu azimuth, nu time, nu x, nu y, nu z, track length (0 for cascade), isTrack (track = 1, cascasde = 0), flavor, type (anti = 1), isCC (CC=1, NC = 0)]

for a_file in event_file_names:

    f = h5py.File(a_file, "r")
    file_features_DC = f["features_DC"][:]
    file_features_IC = f["features_IC"][:]
    file_labels = f["labels"][:]
    if use_old_reco:
        file_reco = f["reco_labels"][:]
        file_initial_stats = f["initial_stats"][:]
        file_num_pulses = f["num_pulses_per_dom"][:]
    f.close()
    del f

    if file_labels.shape[-1] != 12:
        print("Skipping file %s, output labels not expected and CutMask could be cutting the wrong thing"%a_file)
        continue
   
    from handle_data import CutMask
    mask = CutMask(file_labels)
    e_mask = file_labels[:,0]<emax
    keep_index = np.logical_and(mask[cut_name],e_mask)
    number_events = sum(keep_index)

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

        if full_initial_stats is None:
            full_initial_stats = file_initial_stats[keep_index]
        else:
            full_initial_stats  = np.concatenate((full_initial_stats , file_initial_stats[keep_index]))

        if full_num_pulses is None:
            full_num_pulses = file_num_pulses[keep_index]
        else:
            full_num_pulses = np.concatenate((full_num_pulses, file_num_pulses[keep_index]))
        

    print("Events this file: %i, Saved this file: %i, Cumulative saved: %i\n Finsihed file: %s"%(number_events,np.count_nonzero(keep_index),full_labels.shape[0],a_file))

if shuffle:
    print("Finished concatonating all the files. Now I will shuffle..")
    from handle_data import Shuffler

    shuffled_features_DC, shuffled_features_IC, shuffled_labels, \
    shuffled_reco, shuffled_initial_stats, shuffled_num_pulses = \
    Shuffler(full_features_DC,full_features_IC,full_labels, \
    full_reco, full_initial_stats,full_num_pulses,use_old_reco_flag=use_old_reco)
else: 
    shuffled_features_DC, shuffled_features_IC, shuffled_labels, \
    shuffled_reco, shuffled_initial_stats, shuffled_num_pulses = \
    full_features_DC,full_features_IC,full_labels, \
    full_reco, full_initial_stats,full_num_pulses


if find_quartiles:
    from get_statistics import GetQuartilesList
    from scaler_transformations import new_transform
    #low_stat = q1, high_stat = max
    low_stat_DC, high_stat_DC = GetQuartilesList(full_features_DC)
    low_stat_DC = new_tranform(low_stat_DC)
    high_stat_DC = new_tranform(high_stat_DC)
    low_stat_IC, high_stat_IC = GetQuartilesList(full_features_IC)
    low_stat_IC = new_tranform(low_stat_IC)
    high_stat_IC = new_tranform(high_stat_IC)
    low_stat_labels, high_stat_labels = GetQuartilesList(full_labels)
    if use_old_reco:
        low_stat_reco, high_stat_reco = GetQuartilesList(full_reco)
if find_minmax:
    from get_statistics import GetMinMaxList
    #low_stat = min, high_stat = max
    low_stat_DC, high_stat_DC = GetMinMaxList(full_features_DC)
    low_stat_DC = new_tranform(low_stat_DC)
    high_stat_DC = new_tranform(high_stat_DC)
    print("Max list for DC inputs: %f"%high_stat_DC)
    low_stat_IC, high_stat_IC = GetMinMaxList(full_features_IC)
    low_stat_IC = new_tranform(low_stat_IC)
    high_stat_IC = new_tranform(high_stat_IC)
    print("Max list for IC inputs: %f"%high_stat_IC)
    low_stat_labels, high_stat_labels = GetMinMaxList(full_labels)
    if use_old_reco:
        low_stat_reco, high_stat_reco = GetMinMaxList(full_reco)

#Save output to hdf5 file
print("Total events saved: %i"%full_features_DC.shape[0])
cut_name_nospaces = cut_name.replace(" ","")
events_per_file = int(full_features_DC.shape[0]/num_outputs) + 1
for sep_file in range(0,num_outputs):
    start = events_per_file*sep_file
    if sep_file < num_outputs-1:
        end = events_per_file*(sep_file+1)
    else:
        end = full_features_DC.shape[0]
    output_name = path + output +  cut_name_nospaces + ".lt" + str(int(emax)) +  "_file%02d.hdf5"%sep_file 
    print("I put evnts %i - %i into %s"%(start,end,output_name))
    f = h5py.File(output_name, "w")
    f.create_dataset("features_DC", data=shuffled_features_DC[start:end])
    f.create_dataset("features_IC", data=shuffled_features_IC[start:end])
    f.create_dataset("labels", data=shuffled_labels[start:end])
    if use_old_reco:
        f.create_dataset("reco_labels",data=shuffled_reco[start:end])
        f.create_dataset("initial_stats",data=shuffled_initial_stats[start:end])
        f.create_dataset("num_pulses_per_dom",data=shuffled_num_pulses[start:end])
    if find_quartiles or find_minmax:
        f.create_dataset("low_stat_DC", data=low_stat_DC)
        f.create_dataset("high_stat_DC", data=high_stat_DC)
        f.create_dataset("low_stat_IC", data=low_stat_IC)
        f.create_dataset("high_stat_IC", data=high_stat_IC)
        f.create_dataset("low_stat_labels", data=low_stat_labels)
        f.create_dataset("high_stat_labels", data=high_stat_labels)
        if use_old_reco:
            f.create_dataset("low_stat_reco", data=low_stat_reco)
            f.create_dataset("high_stat_reco", data=high_stat_reco)
    f.close()

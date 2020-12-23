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
parser.add_argument("-r", "--reco",default=False,action='store_true',
                    dest="reco", help="use flag if you want to read in old reco array")
parser.add_argument("-b", "--bin_size",type=float,default=1.,
                    dest="bin_size", help="Size of energy bins in GeV (default = 1GeV)")
parser.add_argument("--max_per_bin",type=int,default=10000,
                    dest="max_evt_per_bin",help="Max number of energy events per bin")
parser.add_argument("--emax",type=float,default=200.0,
                    dest="emax",help="Cut anything greater than this energy (in GeV)")
parser.add_argument("--emin",type=float,default=0.0,
                    dest="emin",help="Cut anything less than this energy (in GeV)")
parser.add_argument("--shuffle",default=False,action='store_true',
                    dest="shuffle", help="use flag if you want to shuffle files")
parser.add_argument("-c", "--cuts",type=str, default="all",
                    dest="cuts", help="Type of events to keep (all, cascade, track, CC, NC, etc.)")
parser.add_argument("-s", "--start",type=str, default="all_start",
                    dest="start_cut", help="Vertex start cut (all_start, old_start_DC, start_DC, start_IC, start_IC19)")
parser.add_argument("-e", "--end",type=str, default="all_end",
                    dest="end_cut", help="End position cut (end_start, end_IC7, end_IC19)")
parser.add_argument("--num_out",type=int,default=1,
                    dest="num_out",help="number of output files you want to split the output into")
parser.add_argument("--transformed",default=False,action='store_true',
                    dest="transformed", help="add flag if labels truth input is already transformed")
parser.add_argument("--efactor",type=float,default=100.0,
                    dest="efactor",help="Factor to adjust energy by, if transformed")
parser.add_argument("--tfactor",type=float,default=200.0,
                    dest="tfactor",help="Factor to adjust track by, if transformed")
parser.add_argument("--verbose",default=False,action='store_true',
                    dest="verbose", help="Print histogram counts each file")
parser.add_argument("--split",default=False,action='store_true',
                    dest="split", help="set flag if you want to split in to train, test, validate")
parser.add_argument("--no_validation",default=True,action='store_false',
                    dest="no_validation", help="set flag if you DO NOT want validation set when splitting")
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
print("Keeping %s event types"%cut_name)

verbose = args.verbose
split_data = args.split
create_validation = args.no_validation
use_old_reco = args.reco
if use_old_reco:
    print("Expecting old reco values in files, pulling from pegleg frames")
shuffle = args.shuffle

# Vertex Cuts (no transformation done yet)
start_cut = args.start_cut
end_cut = args.end_cut
transformed = args.transformed
if transformed:
    azimuth_index = 7
    track_index = 2
    tfactor = args.tfactor
    efactor = args.efactor
else:
    azimuth_index = 2
    track_index = 7
    tfactor = 1.0
    efactor = 1.0


print("Saving PEGLEG info: %s \nBin size: %f GeV \nMax Count Per Bin: %i events \nEnergy Range: %f - %f \nStarting cut: %s \nEnding cut: %s \nShuffling: %s \nKeeping event types: %s"%(use_old_reco,bin_size,max_events_per_bin,emin,emax,start_cut,end_cut,shuffle,cut_name)) 
if transformed:
    print("Assuming file is already transformed by %f for energy and %f for track, and cosine zenith"%(efactor,tfactor))

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
full_weights = None
#Check if exists and make sure they match
full_output_transform = None
full_output_names = None
full_input_transform = None

bins = int((emax-emin)/float(bin_size))
if emax%bin_size !=0:
    bins +=1 #Put remainder into additional bin
count_energy = np.zeros((bins))
max_per_bin = max_events_per_bin
count_no_save = 0
count_out_bounds  = 0
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
    try:    
        file_reco_labels = f["reco_labels"][:]
    except:
        assert not use_old_reco, "No reco_labels found!"
        file_reco_labels = None
    try:
        file_weights = f["weights"][:]
    except:
        file_weights = None
    try:
        file_input_transform = f["input_transform_factors"][:]
    except:
        file_input_transform = None
    try:
        file_output_transform = f["output_transform_factors"][:]
    except:
        file_output_transform = None
    try:
        file_output_names = f["output_label_names"][:]
    except:
        file_output_names = None
     
    f.close()
    del f
  
    if file_labels.shape[0] == 0:
        print("Empty file...skipping...")
        continue
    
    # Applying cuts
    type_mask = CutMask(file_labels)
    vertex_mask = VertexMask(file_labels,azimuth_index=azimuth_index,track_index=track_index,max_track=tfactor)
    vertex_cut = np.logical_and(vertex_mask[start_cut], vertex_mask[end_cut])
    mask = np.logical_and(type_mask[cut_name], vertex_cut)
    mask = type_mask[cut_name]
    mask = np.array(mask,dtype=bool)

    if transformed:
        energy = file_labels[:,0]*efactor
    else:
        energy=file_labels[:,0]
    keep_index = [False]*len(energy)
    print("Total events this file: %i"%len(energy))

    # Check how many events already in each bin, save if under max
    for index,e in enumerate(energy):
        if e > emax:
            count_out_bounds += 1
            continue
        if e < emin:
            count_out_bounds += 1
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

    if file_weights is not None:
        if full_weights is None:
            full_weights = file_weights[keep_index]
        else:
            full_weights = np.concatenate((full_weights, file_weights[keep_index]))  
 
    if file_trig_times is not None:
        if full_trig_times is None:
            full_trig_times = file_trig_times[keep_index]
        else:
            full_trig_times = np.concatenate((full_trig_times, file_trig_times[keep_index]))

    if file_reco_labels is not None:
        if full_reco is None:
            full_reco = file_reco[keep_index]
        else:
            full_reco = np.concatenate((full_reco, file_reco[keep_index]))
    
    #Check input/output transformations & names arrays for continuity
    if file_input_transform is not None:
        if full_input_transform is None:
            full_input_transform = file_input_transform
        else:
            #check they match
            comparison = full_input_transform == file_input_transform
            assert comparison.all(), "Mismatched input transform at file %i"%a_file
    if file_output_transform is not None:
        if full_output_transform is None:
            full_output_transform = file_output_transform
        else:
            #check they match
            comparison = full_output_transform == file_output_transform
            assert comparison.all(), "Mismatched input transform at file %i"%a_file
    if file_output_names is not None:
        if full_output_names is None:
            full_output_names = file_output_names
        else:
            #check they match
            comparison = full_output_names == file_output_names
            assert comparison.all(), "Mismatched input name at file %i"%a_file

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
        if verbose:
            print(count_energy)

if count_out_bounds > 0:
    print("Got rid of %i events out of the energy bounds [%f, %f]"%(count_out_bounds, emin, emax))

# CHECKS
if transformed:
    multiply_by = efactor
else:
    multiply_by = 1 
over_emax = full_labels[:,0] > emax/multiply_by
under_emin = full_labels[:,0] < emin/multiply_by
assert sum(over_emax)==0, "Have events greater than emax in final sample"
assert sum(under_emin)==0, "Have events less than emin in final sample"
if cut_name == "CC":
    isCC = full_labels[:,11] == 1
    assert sum(isCC)==full_labels.shape[0], "Have NC events in data"

# Optional Shuffle
if shuffle == True:
    print("Finished concatonating all the files. Now I will shuffle..")
    from handle_data import Shuffler
    full_features_DC, full_features_IC, full_labels, \
    full_reco, full_stats, full_pulses_per_dom, full_trig_times, full_weights = \
    Shuffler(full_features_DC,full_features_IC,full_labels, \
    full_reco=full_reco, full_initial_stats=full_stats, \
    full_num_pulses=full_pulses_per_dom, full_trig_times=full_trig_times, \
    full_weights=full_weights, use_old_reco_flag=use_old_reco)


#SAVING OUT TO FILES (split train, test, validate optional)
if split_data:
    from handle_data import SplitTrainTest
    X_train_DC, X_train_IC, Y_train, \
    X_test_DC, X_test_IC, Y_test, \
    X_validate_DC, X_validate_IC, Y_validate,\
    reco_train, reco_test, reco_validate,\
    weights_train, weights_test, weights_validate\
    = SplitTrainTest(full_features_DC,full_features_IC,full_labels,\
    reco=full_reco,use_old_reco=use_old_reco,\
    weights=full_weights,create_validation=create_validation,\
    fraction_test=0.1,fraction_validate=0.2)

    #print(weights_train, weights_test, weights_validate)

    # Save output to hdf5 file with split

    print(count_energy)
    print("Total events saved: %i"%Y_train.shape[0])

    test_per_file = int(X_test_DC.shape[0]/num_outputs) + 1
    train_per_file = int(X_train_DC.shape[0]/num_outputs) + 1
    if create_validation:
        validate_per_file = int(X_validate_DC.shape[0]/num_outputs) + 1
    print("%i test events per %i file(s)"%(test_per_file,num_outputs))
    for sep_file in range(0,num_outputs):
        savedlist = []
        if num_outputs > 1:
            filenum = "_file%02d"%sep_file
        else:
            filenum = ""
        output_name = path + output + "_E%i"%emin + "to%i_"%emax + "%s_"%cut_name + "%s_%s_"%(start_cut,end_cut) + "flat_%sbins_%sevtperbin"%(bins,max_events_per_bin) + filenum + ".hdf5"

        test_start = test_per_file*sep_file
        train_start = train_per_file*sep_file
        if create_validation:
            validate_start = validate_per_file*sep_file
        if sep_file < num_outputs-1:
            test_end = test_per_file*(sep_file+1)
            train_end = train_per_file*(sep_file+1)
            if create_validation:
                validate_end = validate_per_file*(sep_file+1)
        else:
            test_end = X_test_DC.shape[0]
            train_end = X_train_DC.shape[0]
            if create_validation:
                validate_end = X_validate_DC.shape[0]

        print("I put %i - %i test events into %s"%(test_start,test_end,output_name))
        print("I put %i - %i train events and %i - %i validate events into %s"%(train_start,train_end,validate_start,validate_end,output_name))
        f = h5py.File(output_name, "w")
        f.create_dataset("Y_test", data=Y_test[test_start:test_end])
        f.create_dataset("X_test_DC", data=X_test_DC[test_start:test_end])
        f.create_dataset("X_test_IC", data=X_test_IC[test_start:test_end])
        savedlist.append("test")
        if full_weights is not None:
            f.create_dataset("weights_test", data=weights_test[test_start:test_end])
            savedlist.append("weights")

        f.create_dataset("Y_train", data=Y_train[train_start:train_end])
        f.create_dataset("X_train_DC", data=X_train_DC[train_start:train_end])
        f.create_dataset("X_train_IC", data=X_train_IC[train_start:train_end])
        savedlist.append("train")
        if full_weights is not None:
            f.create_dataset("weights_train", data=weights_train[train_start:train_end])
        
        if create_validation:
            f.create_dataset("Y_validate", data=Y_validate[validate_start:validate_end])
            f.create_dataset("X_validate_IC", data=X_validate_IC[validate_start:validate_end])
            f.create_dataset("X_validate_DC", data=X_validate_DC[validate_start:validate_end])
            savedlist.append("validation")
            if full_weights is not None:
                f.create_dataset("weights_validate", data=weights_validate[validate_start:validate_end])
        
        if full_reco is not None:
            f.create_dataset("reco_test",data=reco_test[test_start:test_end])
            f.create_dataset("reco_train",data=reco_train[train_start:train_end])
            savedlist.append("old reco")
            if create_validation:
                f.create_dataset("reco_validate",data=reco_validate[validate_start:validate_end]) 

        if full_output_names is not None:
            f.create_dataset("output_label_names",data=full_output_names)
            savedlist.append("output label names")
        if full_output_transform is not None:
            f.create_dataset("output_transform_factors",data=full_output_transform)
            savedlist.append("output transform factors")
        if full_input_transform is not None:
            f.create_dataset("input_transform_factors",data=full_input_transform)
            savedlist.append("input transform factors")

        f.close()
        print("Saved: ", savedlist)

else:
    #Save output to hdf5 file without splitting
    print(count_energy)
    print("Total events saved: %i"%full_features_DC.shape[0])
    events_per_file = int(full_features_DC.shape[0]/num_outputs) + 1
    for sep_file in range(0,num_outputs):
        savedlist = []
        start = events_per_file*sep_file
        if sep_file < num_outputs-1:
            end = events_per_file*(sep_file+1)
        else:
            end = full_features_DC.shape[0]

        if num_outputs > 1:
            filenum = "_file%02d"%sep_file
        else:
            filenum = ""
        output_name = path + output + "lt%03d_"%emax + "%s_"%cut_name + "%s_%s_"%(start_cut,end_cut) + "flat_%sbins_%sevtperbin"%(bins,max_events_per_bin) + filenum + ".hdf5"
        print("I put evnts %i - %i into %s"%(start,end,output_name))

        f = h5py.File(output_name, "w")
        f.create_dataset("features_DC", data=full_features_DC[start:end])
        f.create_dataset("features_IC", data=full_features_IC[start:end])
        f.create_dataset("labels", data=full_labels[start:end])
        savedlist.append("features DC")
        savedlist.append("features IC")
        savedlist.append("labels")
        if full_weights is not None:
            f.create_dataset("weights", data=full_weights[start:end])
            savedlist.append("weights")
        if full_reco is not None:
            f.create_dataset("reco_labels",data=full_reco[start:end])
            savedlist.append("old reco")
        if full_stats is not None:
            f.create_dataset("initial_stats",data=full_stats[start:end])
            savedlist.append("initial stats")
        if full_pulses_per_dom is not None:
            f.create_dataset("num_pulses_per_dom",data=full_pulses_per_dom[start:end])
            savedlist.append("number of pulses per DOM")
        if full_trig_times is not None:
            f.create_dataset("trigger_times",data=full_trig_times[start:end])
            savedlist.append("trigger times")
        if full_output_names is not None:
            f.create_dataset("output_label_names",data=full_output_names)
            savedlist.append("output label names")
        if full_output_transform is not None:
            f.create_dataset("output_transform_factors",data=full_output_transform)
            savedlist.append("output transform factors")
        if full_input_transform is not None:
            f.create_dataset("input_transform_factors",data=full_input_transform)
            savedlist.append("input transform factors")
        f.close()

        print("Saved: ", savedlist)

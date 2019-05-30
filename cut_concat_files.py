#################################
# Concatonates, cuts, and shuffles hdf5 training data sets
#   Inputs:
#       -i input files: name of file (can use * and ?)
#       -d path: path to input files
#       -o ouput: name of output file, placed in path directory
#       -c cuts: name of cuts you want to apply (i.e. track only = track)
#       -r reco: bool if files have pegleg reco in them
#################################

import numpy as np
import glob
import h5py
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_files",default='Level5_IC86.2013_genie_numu.014640.*.hdf5',
                    type=str,dest="input_files", help="names for input files")
parser.add_argument("-d", "--path",type=str,default='/mnt/scratch/micall12/training_files/',
                    dest="path", help="path to input files")
parser.add_argument("-o", "--output",type=str,default='Level5_IC86.2013_genie_numu.014640.',
                    dest="output", help="names for output files")
parser.add_argument("-c", "--cuts", type=str, default="track",
                    dest="cuts", help="name of cuts applied (see name options on line 38)")
parser.add_argument("-r","--reco",type=bool, default=False,
                    dest="reco", help="bool if the file has pegleg reco info + initial stats, etc. (if using Level5p files)")
args = parser.parse_args()
input_files = args.input_files
path = args.path
output = args.output
cuts = args.cuts
if args.reco == "True" or args.reco == "true":
    use_old_reco = True
else:    
    use_old_reco = False


# cut names: track, cascasde, CC, NC, track CC, track NC, cascade CC, cascade NC, track CC cascade CC, track NC cascade NC 

print("I am saving only %s events"%cuts)

file_names = path + input_files
event_file_names = sorted(glob.glob(file_names))
assert event_file_names,"No files loaded, please check path."

full_features_DC = None
full_features_IC = None
full_labels = None
full_reco = None
full_initial_stats = None
full_num_pulses = None

def define_cuts(mask_list,logical_and_or="and"):
    mask = {}
    max_masks = len(mask_list)
    for i in range(0,max_masks):
        this_cut = mask_list[i]
        mask_name = "mask" + str(i+1)
        if this_cut =="track": #track only
            mask_type = isTrack==1
        elif this_cut =="cascade": #cascade only
            mask_type = isTrack==0
        elif this_cut =="CC": # CC only
            mask_type = isCC==1
        elif this_cut =="NC": # NC only
            mask_type = isCC==0
        else:
            print("I don't know what cut this is, I'm going to break...")
            break
        
        if i==0:
            mask[mask_name] = mask_type
        else:
            last_mask_name = "mask" + str(i)
            if logical_and_or == "and":
                mask[mask_name] = np.logical_and( mask_type, mask[last_mask_name]) #Cumulative logical and mask
            elif logical_and_or == "or":
                mask[mask_name] = np.logical_or( mask_type, mask[last_mask_name]) #Cumulative logical or mask

    final_mask = "mask" + str(max_masks)
    keep_index = np.where(mask[final_mask])
    return keep_index

# Labels: [ nu energy, nu zenith, nu azimuth, nu time, nu x, nu y, nu z, track length (0 for cascade), isTrack (track = 1, cascasde = 0), flavor, type (anti = 1), isCC (CC=1, NC = 0)]

for a_file in event_file_names:
    print("Pulling data from %s"%a_file)

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

    energy = file_labels[:,0]
    zenith = file_labels[:,1]
    isTrack = file_labels[:,8]
    flavor = file_labels[:,9]
    isCC = file_labels[:,11]
    number_events = len(energy)

    mask = {}
    mask['track'] = isTrack==1 
    mask['cascade'] = isTrack==0 
    mask['CC'] = isTrack==1 
    mask['NC'] = isTrack==0
    mask['track CC'] = np.logical_and( mask['track'], mask['CC'] )
    mask['track NC'] = np.logical_and( mask['track'], mask['NC'] )
    mask['cascade CC'] = np.logical_and( mask['cascade'], mask['CC'] )
    mask['cascade NC'] = np.logical_and( mask['cascade'], mask['NC'] )
    mask['track CC cascade CC'] = np.logical_or( np.logical_and(mask['track'], mask['CC']), np.logical_and(mask['cascade'], mask['CC']) )
    mask['track NC cascade NC'] = np.logical_or( np.logical_and(mask['track'], mask['NC']), np.logical_and(mask['cascade'], mask['NC']) )

    # Check how many events already in each bin, save if under max

    assert len(file_features_DC.shape) == 4, "Features shape is wrong, code can't handle this"


    keep_index = mask[cuts]

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
    
print("Finished concatonating all the files. Now I will shuffle..")

def Shuffler(full_features_DC,full_features_IC,full_labels, full_reco=None, full_initial_stats=None, full_num_pulses=None):
    """Shuffle the contents of the arrays
        Receives:
        full_features_DC = fully concatenated DC array
        full_features_IC = fully concatenated IC array
        full_labels = fully concatenated labels array
    """
    shuffled_features_DC = np.zeros_like(full_features_DC)
    shuffled_features_IC = np.zeros_like(full_features_IC)
    shuffled_labels = np.zeros_like(full_labels)
    if full_reco:
        shuffled_reco = np.zeros_like(full_reco)
        shuffled_initial_stats = np.zeros_like(full_initial_stats)
        shuffled_num_pulses = np.zeros_like(full_num_pulses)
    else:
        shuffled_reco = None
        shuffled_initial_stats = None
        shuffled_num_pulses = None
    random_order = np.arange(0,full_features_DC.shape[0])
    np.random.shuffle(random_order)
    for evt_num in range(0,len(random_order)):
        shuffled_features_DC[evt_num] = full_features_DC[random_order[evt_num]]
        shuffled_features_IC[evt_num] = full_features_IC[random_order[evt_num]]
        shuffled_labels[evt_num] = full_labels[random_order[evt_num]]
        shuffled_reco[evt_num] = full_reco[random_order[evt_num]]
        shuffled_initial_stats[evt_num] = full_initial_stats[random_order[evt_num]]
        shuffled_num_pulses[evt_num] = full_num_pulses[random_order[evt_num]]

    return shuffled_features_DC, shuffled_features_IC, shuffled_labels, shuffled_reco, shuffled_initial_stats, shuffled_num_pulses

if use_old_reco:
    shuffled_features_DC, shuffled_features_IC, shuffled_labels, \
    shuffled_reco, shuffled_initial_stats, shuffled_num_pulses = \
    Shuffler(full_features_DC,full_features_IC,full_labels, \
    full_reco, full_initial_stats,full_num_pulses)
else:
    shuffled_features_DC, shuffled_features_IC, shuffled_labels = Shuffler(full_features_DC,full_features_IC,full_labels)


#Save output to hdf5 file
print("Total events saved: %i"%full_features_DC.shape[0])
cut_name = cuts.replace(" ","") 
output_name = path + output +  cut_name + ".lt60_vertexDC.hdf5" 
print("I put everything into %s"%output_name)
f = h5py.File(output_name, "w")
f.create_dataset("features_DC", data=shuffled_features_DC)
f.create_dataset("features_IC", data=shuffled_features_IC)
f.create_dataset("labels", data=shuffled_labels)
if use_old_reco:
    f.create_dataset("reco_labels",data=shuffled_reco)
    f.create_dataset("initial_stats",data=shuffle_initial_stats)
    f.create_dataset("num_pulses_per_dom",data=shuffle_num_pulses)
f.close()

######################################
# Simply splits initial hdf5 file into multiple output files
#   Takes in features_DC, features_IC, and labels from hdf5 file
########################################

import argparse
import h5py

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_file",default=None,
                    type=str,dest="input_file", help="name for ONE input file")
parser.add_argument("-d", "--path",type=str,default='/mnt/scratch/micall12/training_files/',
                    dest="path", help="path to input files")
parser.add_argument("-n", "--num_out",type=int,default=None,
                    dest="num_out",help="number of output files you want to split the output into")
args = parser.parse_args()

path = args.path
input_file = path + args.input_file
output = str(input_file)[:-5]
num_outputs = args.num_out

#Read in file
f = h5py.File(input_file, "r")
full_features_DC = f["features_DC"][:]
full_features_IC = f["features_IC"][:]
full_labels = f["labels"][:]
# See if there is anything else to pull in from the file, skip if not
try:
    full_reco = f["reco_labels"][:]
except:
    full_reco = None
try:
    full_initial_stats = f["initial_stats"][:]
except:
    full_initial_stats = None
try:
    full_num_pulses = f["num_pulses_per_dom"][:]
except:
    full_num_pulses = None
try:
    full_trig_times = f["trigger_times"][:]
except:
    full_trig_times = None
f.close()
del f

#Save output to hdf5 file
print("Total events saved: %i"%full_features_DC.shape[0])
events_per_file = int(full_features_DC.shape[0]/num_outputs) + 1
for sep_file in range(0,num_outputs):
    start = events_per_file*sep_file
    if sep_file < num_outputs-1:
        end = events_per_file*(sep_file+1)
    else:
        end = full_features_DC.shape[0]
    output_name = output + "_file%02d.hdf5"%sep_file
    print("I put evnts %i - %i into %s"%(start,end,output_name))
    f = h5py.File(output_name, "w")
    f.create_dataset("features_DC", data=full_features_DC[start:end])
    f.create_dataset("features_IC", data=full_features_IC[start:end])
    f.create_dataset("labels", data=full_labels[start:end])
    if full_reco is not None:
        f.create_dataset("reco_labels",data=full_reco[start:end])
    if full_initial_stats is not None:
        f.create_dataset("initial_stats",data=full_initial_stats[start:end])
    if full_num_pulses is not None:
        f.create_dataset("num_pulses_per_dom",data=full_num_pulses[start:end])
    if full_trig_times is not None:
        f.create_dataset("trigger_times",data=full_trig_times[start:end])
    f.close()


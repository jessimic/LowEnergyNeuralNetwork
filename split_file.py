######################################
# Simply splits initial hdf5 file into multiple output files
#   Takes in features_DC, features_IC, and labels from hdf5 file
########################################

import h5py

path = "/mnt/scratch/micall12/training_files/"
a_file = path + "Level5p_IC86.2013_genie_nue.300.timebinscascade.lt60_vertexDC_fullfile.hdf5" 
cuts = "cascade"
num_outputs = 40
output = "Level5p_IC86.2013_genie_nue.300.timebins."
use_old_reco = True

f = h5py.File(a_file, "r")
full_features_DC = f["features_DC"][:]
full_features_IC = f["features_IC"][:]
full_labels = f["labels"][:]
if use_old_reco:
    full_reco = f["reco_labels"][:]
    full_initial_stats = f["initial_stats"][:]
    full_num_pulses = f["num_pulses_per_dom"][:]
f.close()
del f

#Save output to hdf5 file
print("Total events saved: %i"%full_features_DC.shape[0])
cut_name = cuts.replace(" ","")
events_per_file = int(full_features_DC.shape[0]/num_outputs) + 1
for sep_file in range(0,num_outputs):
    start = events_per_file*sep_file
    if sep_file < num_outputs-1:
        end = events_per_file*(sep_file+1)
    else:
        end = full_features_DC.shape[0]
    output_name = path + output +  cut_name + ".lt60_vertexDC_file" + str(sep_file) + ".hdf5"
    print("I put evnts %i - %i into %s"%(start,end,output_name))
    f = h5py.File(output_name, "w")
    f.create_dataset("features_DC", data=full_features_DC[start:end])
    f.create_dataset("features_IC", data=full_features_IC[start:end])
    f.create_dataset("labels", data=full_labels[start:end])
    if use_old_reco:
        f.create_dataset("reco_labels",data=full_reco[start:end])
        f.create_dataset("initial_stats",data=full_initial_stats[start:end])
        f.create_dataset("num_pulses_per_dom",data=full_num_pulses[start:end])
    f.close()


### WARNING -- NOT TESTED

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
#       --shuffle       True if you want to shuffle order of events, default is true
#       -v validate:    bool if output should have validation arrays
#       --scaler:       name of scaler to use (MaxAbs, MinMax, Robust)
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
parser.add_argument("-o", "--outdir",type=str,default='/mnt/scratch/micall12/training_files/',
                    dest="outdir", help="path for storing output file")
parser.add_argument("-n", "--name",type=str,default='cut_concat_separated',
                    dest="name", help="name for output file")
parser.add_argument("-c", "--cuts", type=str, default="all",
                    dest="cuts", help="name of cuts applied (see name options on line 38)")
# cut names: track, cascade, CC, NC, track CC, track NC, cascade CC, cascade NC 
parser.add_argument("-r","--reco",type=str, default=False,
                    dest="reco", help="bool if the file has pegleg reco info + initial stats, etc. (if using Level5p files)")
parser.add_argument("--num_out",type=int,default=1,
                    dest="num_out",help="number of output files you want to split the output into")
parser.add_argument("--emax",type=float,default=200.0,
                    dest="emax",help="Max energy to keep, cut anything above")
parser.add_argument("--emin",type=float,default=5.0,
                    dest="emin",help="Min energy to keep, cut anything below")
parser.add_argument("--start",type=str,default="all_start",
                    dest="start",help="vertex start cut (all_start, start_DC, old_start_DC, start_IC7)")
parser.add_argument("--end",type=str,default="all_end",
                    dest="end",help="end position cut (all_end, end_IC7, end_IC19)")
parser.add_argument("--tfactor",type=float,default=200.,
                    dest="tfactor", help="factor to divide track length by for transforming output")
parser.add_argument("--efactor",type=float,default=100,
                    dest="efactor", help="factor to divide energy by for transforming output")
parser.add_argument("--not_transformed", default=False,action='store_true',
                    dest="not_transformed", help="set flag if dataset is NOT tranformed")
parser.add_argument("--shuffle", default=False,action='store_true',
                    dest="shuffle", help="set flag if you want to shuffle")
parser.add_argument("--no_validation",default=False,action='store_true',
                    dest="no_validation", help="set flag to NOT separate validation")
parser.add_argument("--test_only", default=False,action='store_true',
                        dest='test_only',help="Put all events into test arrays only")
parser.add_argument("--no_cuts",default=False,action='store_true',
                    dest='no_cuts',help="set flag to NOT APPLY energy, vertex, or type cut. WILL NOT AFFECT COS ZENITH CUT.")
parser.add_argument("--do_upgoing_cut",default=False,action='store_true',
                    dest='do_upgoing_cut',help="set flag to cut on cosine zenith < 0.3 (cut out downgoing events)")
parser.add_argument("--split_train",default=False,action='store_true',
                    dest='split_train',help="set flag to split into train, test, validation sets")
parser.add_argument("--max_count",default=None,
                    dest="max_count",help="Max number of files to use")
parser.add_argument("--total_events",default=None,
                    dest="total_events",help="Total number of events to use, cut after shuffle")
parser.add_argument("--test_fraction",default=0.1,type=float,
                    dest="test_fraction",help="Fraction of events for testing sample")
parser.add_argument
args = parser.parse_args()
input_files = args.input_files
path = args.path
outdir = args.outdir
output = args.name
num_outputs = args.num_out
assert num_outputs<100, "NEED TO CHANGE FILENAME TO ACCOMODATE THIS MANY NUMBERS"
max_count = args.max_count

do_cuts = not(args.no_cuts)
do_upgoing_cut = args.do_upgoing_cut
transformed = not(args.not_transformed)
split_train = args.split_train
shuffle = args.shuffle
create_validation = not(args.no_validation)
total_events = args.total_events
test_fraction = args.test_fraction
if total_events is not None:
    assert shuffle == True, "Must shuffle if you wanted to make a cut on total number of events"
    total_events = int(total_events)

if transformed:
    azimuth_index=7
    track_index=2
    track_factor = args.tfactor
else:
    azimuth_index=2
    track_index=7
    track_fractor = 1
    
cut_name = args.cuts
emax = args.emax
emin = args.emin
start_cut = args.start
end_cut = args.end
if args.efactor is None:
    energy_factor = emax
else:
    energy_factor = args.efactor
if not transformed:
    energy_factor = 1
test_only = args.test_only

if args.reco == "True" or args.reco == "true":
    use_old_reco = True
    print("Expecting old reco values in file, from pegleg, etc.")
else:    
    use_old_reco = False

print("\nSaving old reco info: %s \nNumber output files: %i \nDo Cuts: %s \nAssuming Transformed Input: %s (Energy Factor %i, Track Factor %i) \nShuffling: %s \nSplit Train/Test: %s \nMake validation: %s \nTest Only: %s\n"%(use_old_reco,num_outputs,do_cuts,transformed, energy_factor, track_factor, shuffle,split_train, create_validation,test_only)) 
if total_events is not None:
    print("Cutting total events to be at or less than %i"%total_events)
if do_cuts:
    print("\nEnergy Max: %f GeV \nEnergy Min: %f GeV \nStarting Vertex Cut: %s \nEnding Vertex Cut: %s \nType Cut: %s\n"%(emax,emin,start_cut,end_cut,cut_name)) 
if do_upgoing_cut:
    print("Applying cut on upgoing events, keeping cosine zenith < 0.3")

file_names = path + input_files
event_file_names = sorted(glob.glob(file_names))
assert event_file_names,"No files loaded, please check path."

full_features_DC = None
full_features_IC = None
full_labels = None
full_reco = None
full_initial_stats = None
full_num_pulses = None
full_trig_times = None
full_weights = None

# Labels: [ nu energy, nu zenith, nu azimuth, nu time, nu x, nu y, nu z, track length (0 for cascade), isTrack (track = 1, cascasde = 0), flavor, type (anti = 1), isCC (CC=1, NC = 0)]

count_files = 0
for a_file in event_file_names:
    
    if max_count is not None:
        if count_files > int(max_count):
            break

    f = h5py.File(a_file, "r")
    file_features_DC = f["features_DC"][:]
    file_features_IC = f["features_IC"][:]
    file_labels = f["labels"][:]
    try:
        file_weights = f["weights"][:]
    except:
        print("no weights included")
        pass
    if use_old_reco:
        file_reco = f["reco_labels"][:]
    f.close()
    del f

    #if file_labels.shape[-1] != 12:
    #    print("Skipping file %s, output labels not expected and CutMask could be cutting the wrong thing"%a_file)
    #    continue
  
    if do_upgoing_cut:
        if transformed:
            mask_coszen = file_labels[:,1] < 0.3
        else:
            mask_coszen = np.cos(file_labels[:,1] < 0.3)
        keep_index = mask_coszen

    if do_cuts:
        from handle_data import CutMask
        from handle_data import VertexMask
        type_mask = CutMask(file_labels)
        vertex_mask = VertexMask(file_labels,azimuth_index=azimuth_index,track_index=track_index,max_track=track_factor)
        vertex_cut = np.logical_and(vertex_mask[start_cut], vertex_mask[end_cut])
        mask = np.logical_and(type_mask[cut_name], vertex_cut)
        mask = np.array(mask,dtype=bool)
        e_mask = np.logical_and(np.array(file_labels[:,0])>(emin/energy_factor), np.array(file_labels[:,0])<(emax/energy_factor))
        keep_index = np.logical_and(mask,e_mask)
    else:
        keep_index = np.ones(file_labels.shape[0],dtype=bool) #true energy > 0, should be all true
        assert sum(keep_index)==len(keep_index), "True energy is not > 0 for all events"
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
    
    if file_weights is not None:
        if full_weights is None:
            full_weights = file_weights[keep_index]
        else:
            full_weights = np.concatenate((full_weights, file_weights[keep_index]))

    if use_old_reco:
        if full_reco is None:
            full_reco = file_reco[keep_index]
        else:
            full_reco = np.concatenate((full_reco, file_reco[keep_index]))
    
    count_files +=1
    print("Events this file: %i, Saved this file: %i, Cumulative saved: %i\n Finsihed file: %s"%(number_events,np.count_nonzero(keep_index),full_labels.shape[0],a_file))
del file_features_DC
del file_features_IC
del file_labels
if use_old_reco:
    del file_reco

if shuffle:
    print("Finished concatonating all the files. Now I will shuffle..")
    from handle_data import Shuffler

    full_features_DC, full_features_IC, full_labels, \
    full_reco, full_initial_stats, full_num_pulses,full_trig_times,full_weights = \
    Shuffler(full_features_DC,full_features_IC,full_labels,\
    full_reco=full_reco, full_initial_stats=full_initial_stats,\
    full_num_pulses=full_num_pulses,full_weights=full_weights,use_old_reco_flag=use_old_reco)

if total_events is not None:
    if full_labels.shape[0] <= total_events:
        print("Total number of events is %i, which is less than requested total events at %i. NO CUT ON TOTAL EVENTS APPLIED"%(full_labels.shape[0], total_events))
    else:
        print("Only saving %i events of %i available, with total_events arg in play"%(total_events,full_labels.shape[0]))
        full_features_DC = full_features_DC[:total_events]
        full_features_IC = full_features_IC[:total_events]
        full_labels = full_labels[:total_events]
        if full_reco is not None:
            full_reco = full_reco[:total_events]
        if full_initial_stats is not None:
            full_initial_stats = full_initial_stats[:total_events]
        if full_num_pulses is not None:
            full_num_pulses = full_num_pulses[:total_events]
        if full_trig_times is not None:
            full_trig_times = full_trig_times[:total_events]
        if full_weights is not None:
            full_weights = full_weights[:total_events]

# Check that it follows the usual energy and CC cuts
if do_cuts:
    over_emax = full_labels[:,0] > emax/energy_factor
    under_emin = full_labels[:,0] < emin/energy_factor
    assert sum(over_emax)==0, "Have events greater than emax in final sample"
    assert sum(under_emin)==0, "Have events less than emin in final sample"
    if cut_name == "CC":
        isCC = full_labels[:,11] == 1
        assert sum(isCC)==full_labels.shape[0], "Have NC events in data"


#Split data
if test_only:
    assert split_train == False, "Cannot split training set after assuming only making a testing set"
    print("MAKING ALL %i EVENTS FOR TEST OUTPUT"%full_labels.shape[0])
    X_test_DC = full_features_DC
    X_test_IC = full_features_IC
    Y_test = full_labels
    weights_test = full_weights
    if use_old_reco == True:
        reco_test = full_reco
if split_train:
    from handle_data import SplitTrainTest
    X_train_DC, X_train_IC, Y_train, \
    X_test_DC, X_test_IC, Y_test, \
    X_validate_DC, X_validate_IC, Y_validate,\
    reco_train, reco_test, reco_validate,\
    weights_train, weights_test, weights_validate\
    = SplitTrainTest(full_features_DC,full_features_IC,full_labels,\
    reco=full_reco,use_old_reco=use_old_reco,\
    weights=full_weights,create_validation=create_validation,\
    fraction_test=test_fraction,fraction_validate=0.2)


#Save output to hdf5 file
print("Total events saving: %i"%full_features_DC.shape[0])
print("Total number of files used: %i"%count_files)

#Save output to hdf5 file
if do_cuts:
    cut_name_nospaces = cut_name.replace(" ","")
    cut_file_name = "." + cut_name_nospaces + ".%iGeV_to_%iGeV_%s_%s"%(emin,emax,start_cut,end_cut) 
else:
    cut_file_name = ".no_cuts"


if split_train:
    train_per_file = int(X_train_DC.shape[0]/num_outputs) + 1
    validate_per_file = int(X_validate_DC.shape[0]/num_outputs) + 1
    print("Saving %i train events and %i validate events per %i file(s)"%(train_per_file,validate_per_file,num_outputs))

if split_train or test_only:
    test_per_file = int(X_test_DC.shape[0]/num_outputs) + 1
    print("%i test events per %i file(s)"%(test_per_file,num_outputs))
    for sep_file in range(0,num_outputs):
        if num_outputs > 1:
            end = "_file%02d.hdf5"%sep_file
        else:
            end = ".hdf5"
        output_file = outdir + output + cut_file_name + end

        test_start = test_per_file*sep_file
        if not test_only:
            train_start = train_per_file*sep_file
            validate_start = validate_per_file*sep_file
        if sep_file < num_outputs-1:
            test_end = test_per_file*(sep_file+1)
            if not test_only:
                train_end = train_per_file*(sep_file+1)
                validate_end = validate_per_file*(sep_file+1)
        else:
            test_end = X_test_DC.shape[0]
            if not test_only:
                train_end = X_train_DC.shape[0] 
                validate_end = X_validate_DC.shape[0] 

        print("I put %i - %i test events into %s"%(test_start,test_end,output_file))
        if not test_only:
            print("I put %i - %i train events and %i - %i validate events into %s"%(train_start,train_end,validate_start,validate_end,output_file))
        f = h5py.File(output_file, "w")
        f.create_dataset("Y_test", data=Y_test[test_start:test_end])
        f.create_dataset("X_test_DC", data=X_test_DC[test_start:test_end])
        f.create_dataset("X_test_IC", data=X_test_IC[test_start:test_end])
        f.create_dataset("weights_test", data=weights_test[test_start:test_end])
        #f.attrs['output_label_names'] = [a.encode('utf8') for a in output_label_names]
        #f.create_dataset("output_label_names",data=f.attrs['output_label_names'])
        #f.create_dataset("input_transform_factors",data=input_transform_factors)
        #f.create_dataset("output_transform_factors",data=output_transform_factors)

        if not test_only:
            f.create_dataset("Y_train", data=Y_train[train_start:train_end])
            f.create_dataset("X_train_DC", data=X_train_DC[train_start:train_end])
            f.create_dataset("X_train_IC", data=X_train_IC[train_start:train_end])
            f.create_dataset("weights_train", data=weights_train[train_start:train_end])
            if create_validation:
                f.create_dataset("Y_validate", data=Y_validate[validate_start:validate_end])
                f.create_dataset("X_validate_IC", data=X_validate_IC[validate_start:validate_end])
                f.create_dataset("X_validate_DC", data=X_validate_DC[validate_start:validate_end])
                f.create_dataset("weights_validate", data=weights_validate[validate_start:validate_end])
        if use_old_reco:
            f.create_dataset("reco_test",data=reco_test[test_start:test_end])
            if not test_only:
                f.create_dataset("reco_train",data=reco_train[train_start:train_end])
                if create_validation:
                    f.create_dataset("reco_validate",data=reco_validate[validate_start:validate_end])
        f.close()

else:
    events_per_file = int(full_features_DC.shape[0]/num_outputs) + 1        
    print("%i events per %i file(s)"%(events_per_file,num_outputs))
    for sep_file in range(0,num_outputs):
        if num_outputs > 1:
            end = "_file%02d.hdf5"%sep_file
        else:
            end = ".hdf5"
        output_file = outdir + output + cut_file_name + end
        start = events_per_file*sep_file
        if sep_file < num_outputs-1:
            end = events_per_file*(sep_file+1)
        else:
            end = full_features_DC.shape[0]
        print("I put evnts %i - %i into %s"%(start,end,output_file))
        f = h5py.File(output_file, "w")
        f.create_dataset("features_DC", data=full_features_DC[start:end])
        f.create_dataset("features_IC", data=full_features_IC[start:end])
        f.create_dataset("labels", data=full_labels[start:end])
        #f.create_dataset("initial_stats",data=full_initial_stats[start:end])
        #f.create_dataset("num_pulses_per_dom",data=full_num_pulses[start:end])
        f.create_dataset("weights",data=full_weights[start:end])
        if use_old_reco:
            f.create_dataset("reco_labels",data=full_reco[start:end])
        f.close()

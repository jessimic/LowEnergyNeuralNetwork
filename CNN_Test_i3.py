#############################
# Read IceCube files and create training file (hdf5)
#   Modified from code written by Claudio Kopper
#   get_observable_features = access data from IceCube files
#   read_files = read in files and add truth labels
#   Can take 1 or multiple files
#   Input:
#       -i input: name of input file, include path
#       -n name: name for output file, automatically puts in my scratch
#       -r reco: flag to save Level5p pegleg reco output (to compare)
#       --emax: maximum energy saved (60 is default, so keep all < 60 GeV)
#       --cleaned: if you want to pull from SRTTWOfflinePulsesDC, instead of SplitInIcePulses
#       --true_name: string of key to check true particle info from against I3MCTree[0] 
##############################

import numpy as np
import h5py
import argparse

from icecube import icetray, dataio, dataclasses
from I3Tray import I3Units

from collections import OrderedDict
import itertools
import random

import time

## Create ability to change settings from terminal ##
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input",type=str,default=None,
                    dest="input_file", help="path and name of the input file")
parser.add_argument("-n", "--name",type=str,default=None,
                    dest="output_name",help="name for output file (no path)")
parser.add_argument("-o", "--outdir",type=str,default='/mnt/scratch/micall12/training_files/',
                    dest="output_dir", help="path of ouput file")
parser.add_argument("--model_dir",type=str,default='/mnt/home/micall12/LowEnergyNeuralNetwork/output_plots/',
                    dest="model_dir", help="path for where to pull the model from")
parser.add_argument("--model_name",type=str,
                    dest="model_name", help="name of output folder where model is located")
parser.add_argument("-e","--epochs",default=None,
                    dest="epochs", help="epoch number for model to use")
parser.add_argument("--cleaned",type=str,default="False",
                    dest="cleaned", help="True if wanted to use SRTTWOfflinePulsesDC")
args = parser.parse_args()

input_file = args.input_file
output_dir = args.output_dir
output_name = args.output_name
if args.cleaned == "True" or args.cleaned == "true":
    use_cleaned_pulses = True
else:
    use_cleaned_pulses = False
efactor=100.

model_name = args.model_name
model_path = args.model_dir + args.model_name
if args.epochs is None:
    model_name ="%s/%s_final_model.hdf5"%(model_path,model_name)
else:
    model_name ="%s/%s_%sepochs_model.hdf5"%(model_path,model_name,args.epochs)
def get_observable_features(frame,low_window=-500,high_window=4000):
    """
    Load observable features from IceCube files
    Receives:
        frame = IceCube object type from files
    Returns:
        observable_features: Observables dictionary
    """
    if use_cleaned_pulses:
        ice_pulses = dataclasses.I3RecoPulseSeriesMap.from_frame(frame,'SRTTWOfflinePulsesDC')
    else:
        ice_pulses = dataclasses.I3RecoPulseSeriesMap.from_frame(frame,'SplitInIcePulses')

    #First cut: Check if there are 8 cleaned pulses > 0.25 PE
    cleaned_ice_pulses = dataclasses.I3RecoPulseSeriesMap.from_frame(frame,'SRTTWOfflinePulsesDC')
    count_cleaned_pulses = 0
    clean_pulses_8_or_more = False
    for omkey, pulselist in cleaned_ice_pulses:
        if clean_pulses_8_or_more == True:
            break
        for pulse in pulselist:

            a_charge = pulse.charge

            #Cut any pulses < 0.25 PE
            if a_charge < 0.25:
                continue

            #Count number pulses > 0.25 PE in event
            count_cleaned_pulses +=1
            if count_cleaned_pulses >=8:
                clean_pulses_8_or_more = True
                break
    #Look inside ice pulses and get stats on charges and time
    # DC = deep core which is certain strings/DOMs in IceCube
    store_string = []
    IC_near_DC_strings = [17, 18, 19, 25, 26, 27, 28, 34, 35, 36, 37, 38, 44, 45, 46, 47, 54, 55, 56]
    #IC_near_DC_strings = [26, 27, 35, 36, 37, 45, 46]
    ICstrings = len(IC_near_DC_strings)
    DC_strings = [79, 80, 81, 82, 83, 84, 85, 86]

    #Five summary variables: sum charges, time first pulse, Time of last pulse, Charge weighted mean time of pulses, Charge weighted standard deviation of pulse times
    array_DC = np.zeros([len(DC_strings),60,5]) # [string, dom_index, charge & time summary]
    array_IC_near_DC = np.zeros([len(IC_near_DC_strings),60,5]) # [string, dom_index, charge & time summary]
    initial_stats = np.zeros([4])
    num_pulses_per_dom = np.zeros([len(DC_strings),60,1])
    count_outside = 0
    charge_outside = 0
    count_inside = 0
    charge_inside = 0
    total_pulses = 0

    # Config 1011 is SMT3
    # dataclasses.TriggerKey(source, ttype, config_id)
    triggers = frame['I3TriggerHierarchy']
    trigger_time = None
    num_extra_DC_triggers = 0
    for trig in triggers:
        key_str = str(trig.key)
        s = key_str.strip('[').strip(']').split(':')
        if len(s) > 2:
            config_id = int(s[2])
            if config_id == 1011:
                if trigger_time:
                    num_extra_DC_triggers +=1
                trigger_time = trig.time
                
    if trigger_time == None:
        shift_time_by = 0
    else:
        shift_time_by = trigger_time

    #Start by making all times negative shift time (to distinguish null from 0)
    array_DC[...,1:] = -20000
    array_IC_near_DC[...,1:] = -20000
    
    #Only go through pulse series if we're keeping it    
    if clean_pulses_8_or_more == True:
        for omkey, pulselist in ice_pulses:
            dom_index =  omkey.om-1
            string_val = omkey.string
            timelist = []
            chargelist = []

            DC_flag = False
            IC_near_DC_flag = False

            
            for pulse in pulselist:
                
                charge = pulse.charge

                #Cut any pulses < 0.25 PE
                if charge < 0.25:
                    continue
                
                # Quantize pulse chargest to make all seasons appear the same
                quanta = 0.05
                charge = (np.float64(charge) // quanta) * quanta + quanta / 2.

                if string_val not in store_string:
                    store_string.append(string_val)

                # Check IceCube near DeepCore DOMs
                if( (string_val in IC_near_DC_strings) and dom_index<60):
                    string_index = IC_near_DC_strings.index(string_val)
                    timelist.append(pulse.time)
                    chargelist.append(charge)
                    IC_near_DC_flag = True

                # Check DeepCore DOMS
                elif ( (string_val in DC_strings) and dom_index<60): #dom_index >=10
                    string_index = DC_strings.index(string_val)
                    timelist.append(pulse.time)
                    chargelist.append(charge)
                    DC_flag = True


                else:
                    count_outside +=1
                    charge_outside += charge
                    
            if DC_flag == True or IC_near_DC_flag == True:

                charge_array = np.array(chargelist)
                time_array = np.array(timelist)
                time_array = [ (t_value - shift_time_by) for t_value in time_array ]
                time_shifted = [ (t_value - time_array[0]) for t_value in time_array ]
                time_shifted = np.array(time_shifted)
                mask_500 = time_shifted<500
                mask_100 = time_shifted<100

                # Remove pulses so only those in certain time window are saved
                original_num_pulses = len(timelist)
                time_array_in_window = list(time_array)
                charge_array_in_window = list(charge_array)
                for time_index in range(0,original_num_pulses):
                    time_value =  time_array[time_index]
                    if time_value < low_window or time_value > high_window:
                        time_array_in_window.remove(time_value)
                        charge_array_in_window.remove(charge_array[time_index])
                charge_array = np.array(charge_array_in_window)
                time_array = np.array(time_array_in_window)
                assert len(charge_array)==len(time_array), "Mismatched pulse time and charge"
                if len(charge_array) == 0:
                    continue

                #Original Stats
                count_inside += len(chargelist)
                charge_inside += sum(chargelist)

                # Check that pulses are sorted in time
                for i_t,time in enumerate(time_array):
                    assert time == sorted(time_array)[i_t], "Pulses are not pre-sorted!"

                # Charge weighted mean and stdev
                weighted_avg_time = np.average(time_array,weights=charge_array)
                weighted_std_time = np.sqrt( np.average((time_array - weighted_avg_time)**2, weights=charge_array) )


            if DC_flag == True:
                array_DC[string_index,dom_index,0] = sum(chargelist)
                array_DC[string_index,dom_index,1] = time_array[0]
                array_DC[string_index,dom_index,2] = time_array[-1]
                array_DC[string_index,dom_index,3] = weighted_avg_time
                array_DC[string_index,dom_index,4] = weighted_std_time
                
                num_pulses_per_dom[string_index,dom_index,0] = len(chargelist)
            
            if IC_near_DC_flag == True:
                array_IC_near_DC[string_index,dom_index,0] = sum(chargelist)
                array_IC_near_DC[string_index,dom_index,1] = time_array[0]
                array_IC_near_DC[string_index,dom_index,2] = time_array[-1]
                array_IC_near_DC[string_index,dom_index,3] = weighted_avg_time
                array_IC_near_DC[string_index,dom_index,4] = weighted_std_time

    return array_DC, array_IC_near_DC, trigger_time, num_extra_DC_triggers


def apply_transform(features_DC, features_IC, labels=None, energy_factor=100., track_factor=200.,transform="MaxAbs"):
    from scaler_transformations import TransformData, new_transform
    static_stats = [25., 4000., 4000., 4000., 2000.]
    low_stat_DC = static_stats
    high_stat_DC = static_stats
    low_stat_IC = static_stats
    high_stat_IC = static_stats

    features_DC = new_transform(features_DC)
    features_DC = TransformData(features_DC, low_stats=low_stat_DC, high_stats=high_stat_DC, scaler=transform)
    features_IC = new_transform(features_IC)
    features_IC = TransformData(features_IC, low_stats=low_stat_IC, high_stats=high_stat_IC, scaler=transform)
    #labels[:,0] = labels[:,0]/float(energy_factor) #energy
    #labels[:,1] = np.cos(labels[:,1]) #cos zenith

    return features_DC, features_IC


def cnn_test(features_DC, features_IC, load_model_name, output_variables=1,DC_drop_value=0.2,IC_drop_value=0.2,connected_drop_value=0.2):
    from cnn_model import make_network
    
    model_DC = make_network(features_DC,features_IC, output_variables, DC_drop_value, IC_drop_value,connected_drop_value)
    model_DC.load_weights(load_model_name)
    
    Y_test_predicted = model_DC.predict([features_DC,features_IC])

    return Y_test_predicted


def read_files(filename):
    """
    Read list of files, make sure they pass L5 cuts, create truth labels
    Receives:
        filename_list = list of strings, filenames to read data from
    Returns:
        output_features_DC = dict with input observable features from the DC strings
        output_features_IC = dict with input observable features from the IC strings
        output_labels = dict with output labels  (energy, zenith, azimith, time, x, y, z, 
                        tracklength, isTrack, flavor ID, isAntiNeutrino, isCC)
        output_reco_labels = dict with PegLeg output labels (energy, zenith, azimith, time, x, y, z)
        output_initial_stats = array with info on number of pulses and sum of charge "inside" the strings used 
                                vs. "outside", i.e. the strings not used (pulse count outside, charge outside,
                                pulse count inside, charge inside) for finding statistics
        output_num_pulses_per_dom = array that only holds the number of pulses seen per DOM (finding statistics)
        output_trigger_times = list of trigger times for each event (used to shift raw pulse times)
    """
    print("reading file: {}".format(filename))
    event_file = dataio.I3File(filename)

    output_features_DC = []
    output_features_IC = []
    output_headers = []

    for frame in event_file:
        if frame.Stop == icetray.I3Frame.Physics:
            header = frame["I3EventHeader"]

            if header.sub_event_stream != "InIceSplit":
                continue


            DC_array, IC_near_DC_array, trig_time, extra_triggers = get_observable_features(frame)

            # Check if there were multiple SMT3 triggers or no SMT3 triggers
            # Skip event if so
            if extra_triggers > 0 or trig_time == None:
                skipped_triggers +=1
                continue
            
            header_numbers = np.array( [ float(header.run_id), float(header.sub_run_id), float(header.event_id)] )
            output_headers.append(header_numbers)
            output_features_DC.append(DC_array)
            output_features_IC.append(IC_near_DC_array)

        # close the input file once we are done
    del event_file

    output_headers = np.asarray(output_headers)
    output_features_DC = np.asarray(output_features_DC)
    output_features_IC = np.asarray(output_features_IC)

    return  output_features_DC, output_features_IC, output_headers


def test_write(filename_list, model_name,output_dir, output_name, efactor=100.):


    for a_file in filename_list:
        if output_name is None:
            basename = a_file.split("/")[-1] 
            basename = basename[:-7]
            output_name = str(basename) + "_FLERCNN"
        outfile = dataio.I3File(output_dir+output_name+".i3.zst",'w')
        print("Writing to %s"%(output_dir+output_name+".i3.zst"))
        
        DC_array, IC_near_DC_array, header_array = read_files(a_file)
        print(DC_array.shape, IC_near_DC_array.shape)
        DC_array, IC_near_DC_array = apply_transform(DC_array, IC_near_DC_array)


        t0 = time.time()
        energy = cnn_test(DC_array, IC_near_DC_array, model_name)
        t1 = time.time()
        print("Time to run CNN Test on %i events: %f seconds"%(DC_array.shape[0],t1-t0))
            
        index = 0
        event_file = dataio.I3File(a_file)
        for frame in event_file:
            if frame.Stop == icetray.I3Frame.Physics:
                header = frame["I3EventHeader"]

                #Check Header
                if header.sub_event_stream != "InIceSplit":
                    continue
                if float(header.run_id) != header_array[index][0]:
                    print("Run ID is off")
                    continue
                if float(header.sub_run_id) != header_array[index][1]:
                    print("Sub Run ID is off")
                    continue
                if float(header.event_id) != header_array[index][2]:
                    print("Event ID is off")
                    continue

                frame['FLERCNN_energy'] = dataclasses.I3Double(energy[index][0]*efactor)
                outfile.push(frame)
                index +=1
            else:
                outfile.push(frame)
    return 0

#Construct list of filenames
import glob

event_file_names = sorted(glob.glob(input_file))
assert event_file_names,"No files loaded, please check path."
time_start=time.time()
test_write(event_file_names, model_name, output_dir, output_name, efactor)
time_end=time.time()
print("Total time: %f"%(time_end-time_start))


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
parser.add_argument("-n", "--name",default=None,
                    dest="output_name",help="name for output file (no path)")
parser.add_argument("-o", "--outdir",type=str,default='/mnt/scratch/micall12/training_files/',
                    dest="output_dir", help="path of ouput file")
parser.add_argument("-r", "--reco",type=str,default="False",
                    dest="reco", help="True if using Level5p or have a pegleg reco")
parser.add_argument("--reco_type",type=str,default="pegleg",
                    dest="reco_type", help="Options are pegleg or retro")
parser.add_argument("--efactor",type=float,default=100.0,
                    dest="efactor",help="Value to transform (divide by) energy")
parser.add_argument("--tfactor",type=float,default=200.0,
                    dest="tfactor",help="Value to transform (divide by) track length")
parser.add_argument("--cleaned",type=str,default="False",
                    dest="cleaned", help="True if wanted to use SRTTWOfflinePulsesDC")
parser.add_argument("--true_name",type=str,default=None,
                    dest="true_name", help="Name of key for true particle info if you want to check with I3MCTree[0]")
parser.add_argument("--check_filters", default=False,action='store_true',
                        dest='check_filters',help="check for L2-L5 filters")
args = parser.parse_args()

input_file = args.input_file
output_dir = args.output_dir
output_name = args.output_name
efactor=args.efactor
tfactor=args.tfactor
true_name = args.true_name
reco_type = args.reco_type
if args.reco == 'True' or args.reco == 'true':
    use_old_reco = True
    print("Expecting old reco values in files, pulling from %s frames"%reco_type)
else:
    use_old_reco = False
if args.cleaned == "True" or args.cleaned == "true":
    use_cleaned_pulses = True
else:
    use_cleaned_pulses = False
if args.check_filters == "True" or args.check_filters == "true":
    check_filters = True
else:
    check_filters = False

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


        initial_stats[0] = count_outside
        initial_stats[1] = charge_outside
        initial_stats[2] = count_inside
        initial_stats[3] = charge_inside
        
    return array_DC, array_IC_near_DC, initial_stats, num_pulses_per_dom, trigger_time, num_extra_DC_triggers, ICstrings, clean_pulses_8_or_more

def apply_transform(features_DC, features_IC, output_labels=None, energy_factor=100., track_factor=200.,transform="MaxAbs",transform_output=True):
    from scaler_transformations import TransformData, new_transform
    static_stats = [25., 4000., 4000., 4000., 2000.]
    low_stat_DC = static_stats
    high_stat_DC = static_stats
    low_stat_IC = static_stats
    high_stat_IC = static_stats
    input_transform_factors = np.array([high_stat_DC,high_stat_IC])

    features_DC = new_transform(features_DC)
    features_DC = TransformData(features_DC, low_stats=low_stat_DC, high_stats=high_stat_DC, scaler=transform)
    features_IC = new_transform(features_IC)
    features_IC = TransformData(features_IC, low_stats=low_stat_IC, high_stats=high_stat_IC, scaler=transform)

    output_label_names = np.array(["Energy", "Zenith", "Aziuth", "Time", "X", "Y", "Z", "Track", "IsTrack", "Flavor", "IsAntineutrino", "IsCC"])
    output_names = np.array(output_label_names)
    output_transform_factors = np.ones((len(output_names)))
    if transform_output:

        # switch track and azimuth positions
        track = np.copy(output_labels[:,7])
        azimuth = np.copy(output_labels[:,2])

        #Make changes to output_labels array
        output_labels[:,0] = output_labels[:,0]/float(energy_factor) #energy
        output_labels[:,1] = np.cos(output_labels[:,1]) #cos zenith
        output_labels[:,2] = track/float(track_factor) #MAKE TRACK THIRD INPUT
        output_labels[:,7] = azimuth #MOVE AZIMUTH TO WHERE TRACK WAS

        #Track changes in output arrarys
        output_label_names[1] = "Cosine Zenith"
        output_label_names[2] = "Track"
        output_label_names[7] = "Azimuth"
        output_transform_factors[0] = energy_factor
        output_transform_factors[2] = track_factor

        return features_DC, features_IC, output_labels, output_label_names, input_transform_factors, output_transform_factors

def read_files(filename_list, use_old_reco, check_filters, true_name, reco_type):
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
    output_features_DC = []
    output_features_IC = []
    output_labels = []
    output_reco_labels = []
    output_initial_stats = []
    output_num_pulses_per_dom = []
    output_trigger_times = []
    output_weights = []
    isOther_count = 0
    passed_all_filters = 0
    skipped_triggers = 0
    probnu = 0
    less_8_hits = 0


    for event_file_name in filename_list:
        print("reading file: {}".format(event_file_name))
        event_file = dataio.I3File(event_file_name)

        for frame in event_file:
            if frame.Stop == icetray.I3Frame.Physics:
                header = frame["I3EventHeader"]
                #print("Saved events: ", len(output_labels))

                if header.sub_event_stream != "InIceSplit":
                    continue
            
                # ALWAYS USE EVENTS THAT PASSES CLEANING!
                #if use_cleaned_pulses:
                #try:
                #    cleaned = frame["SRTTWOfflinePulsesDC"]
                #except:
                #    continue
                
                # Check filters
                if check_filters:
                    DCFilter = frame["FilterMask"].get("DeepCoreFilter_13").condition_passed
                    L3Filter = frame["L3_oscNext_bool"]
                    L4Filter = frame["L4_oscNext_bool"]
                    L5Filter = frame["L5_oscNext_bool"]
                    if (DCFilter and L3Filter and L4Filter and L5Filter):
                        passed_all_filters +=1
                    else:
                        continue

                # GET TRUTH LABELS
                nu = frame["I3MCTree"][0]
                
                if true_name:
                    nu_check = frame[true_name]
                    assert nu==nu_check,"CHECK I3MCTree[0], DOES NOT MATCH TRUTH IN GIVEN KEY"
                
                if (nu.type != dataclasses.I3Particle.NuMu and nu.type != dataclasses.I3Particle.NuMuBar\
                    and nu.type != dataclasses.I3Particle.NuE and nu.type != dataclasses.I3Particle.NuEBar\
                    and nu.type != dataclasses.I3Particle.NuTau and nu.type != dataclasses.I3Particle.NuTauBar):
                    print("PARTICLE IS NOT NEUTRUNO!! Skipping event...")
                    continue           
 
                nu_x = nu.pos.x
                nu_y = nu.pos.y
                nu_z = nu.pos.z
                nu_zenith = nu.dir.zenith
                nu_azimuth = nu.dir.azimuth
                nu_energy = nu.energy
                nu_time = nu.time
                isTrack = frame['I3MCWeightDict']['InteractionType']==1.   # it is a cascade with a trac
                isCascade = frame['I3MCWeightDict']['InteractionType']==2. # it is just a cascade
                isCC = frame['I3MCWeightDict']['InteractionType']==1.
                isNC = frame['I3MCWeightDict']['InteractionType']==2.
                isOther = not isCC and not isNC
                L4_NoiseClassifier_ProbNu = frame['L4_NoiseClassifier_ProbNu'].value
               


                if use_old_reco:

                    if reco_type == "retro":
                        fit_success = ( "retro_crs_prefit__fit_status" in frame ) and frame["retro_crs_prefit__fit_status"] == 0
                        if fit_success:
                            reco_energy = frame['L7_reconstructed_total_energy'].value
                            reco_time = frame['L7_reconstructed_time'].value
                            reco_zenith = frame['L7_reconstructed_zenith'].value
                            reco_azimuth = frame['L7_reconstructed_azimuth'].value
                            reco_x = frame['L7_reconstructed_vertex_x'].value
                            reco_y = frame['L7_reconstructed_vertex_y'].value
                            reco_z = frame['L7_reconstructed_vertex_z'].value
                            reco_length = frame['L7_reconstructed_track_length'].value
                            reco_casc_energy = frame['L7_reconstructed_cascade_energy'].value
                            reco_track_energy = frame['L7_reconstructed_track_energy'].value
                            reco_em_casc_energy = frame['L7_reconstructed_em_cascade_energy'].value
                        else:
                            continue
                            #reco_energy =0
                            #reco_time = 0
                            #reco_zenith =0
                            #reco_azimuth = 0 
                            #reco_x = 0
                            #reco_y = 0
                            #reco_z = 0
                            #reco_length =0
                            #reco_casc_energy = 0
                            #reco_track_energy = 0
                            #reco_em_casc_energy = 0

                    if reco_type == "pegleg":
                        if not frame.Has('IC86_Dunkman_L6_PegLeg_MultiNest8D_NumuCC'):
                            continue
                        reco_nu = frame['IC86_Dunkman_L6_PegLeg_MultiNest8D_NumuCC']
                        reco_length = reco_nu.length
                        reco_energy = reco_nu.energy
                        reco_time = reco_nu.time
                        reco_zenith = reco_nu.dir.zenith
                        reco_azimuth = reco_nu.dir.azimuth
                        reco_x = reco_nu.pos.x
                        reco_y = reco_nu.pos.y
                        reco_z = reco_nu.pos.z
                        reco_casc_energy = frame['IC86_Dunkman_L6_PegLeg_MultiNest8D_HDCasc'].energy
                        reco_track_energy = frame['IC86_Dunkman_L6_PegLeg_MultiNest8D_Track'].energy
                        reco_em_casc_energy = frame['IC86_Dunkman_L6_PegLeg_MultiNest8D_EMCasc'].energy


                # input file sanity check: this should not print anything since "isOther" should always be false
                if isOther:
                    print("isOTHER - not Track or Cascade...skipping event...")
                    isOther_count += 1
                    continue

                    #print(frame['I3MCWeightDict'])
                
                # set track classification for numu CC only
                if ((nu.type == dataclasses.I3Particle.NuMu or nu.type == dataclasses.I3Particle.NuMuBar) and isCC):
                    isTrack = True
                    isCascade = False
                    if frame["I3MCTree"][1].type == dataclasses.I3Particle.MuMinus or frame["I3MCTree"][1].type == dataclasses.I3Particle.MuPlus:
                        track_length = frame["I3MCTree"][1].length
                    else:
                        print("Second particle in MCTree not muon for numu CC? Skipping event...")
                        continue
                else:
                    isTrack = False
                    isCascade = True
                    track_length = 0
            
                #Save flavor and particle type (anti or not)
                if (nu.type == dataclasses.I3Particle.NuMu):
                    neutrino_type = 14
                    particle_type = 0 #particle
                elif (nu.type == dataclasses.I3Particle.NuMuBar):
                    neutrino_type = 14
                    particle_type = 1 #antiparticle
                elif (nu.type == dataclasses.I3Particle.NuE):
                    neutrino_type = 12
                    particle_type = 0 #particle
                elif (nu.type == dataclasses.I3Particle.NuEBar):
                    neutrino_type = 12
                    particle_type = 1 #antiparticle
                elif (nu.type == dataclasses.I3Particle.NuTau):
                    neutrino_type = 16
                    particle_type = 0 #particle
                elif (nu.type == dataclasses.I3Particle.NuTauBar):
                    neutrino_type = 16
                    particle_type = 1 #antiparticle
                else:
                    print("Do not know first particle type in MCTree, should be neutrino, skipping this event")
                    continue
                
                DC_array, IC_near_DC_array,initial_stats,num_pulses_per_dom, trig_time, extra_triggers, ICstrings, has_8_hits  = get_observable_features(frame)

                # Check if there were multiple SMT3 triggers or no SMT3 triggers
                # Skip event if so
                if extra_triggers > 0 or trig_time == None:
                    skipped_triggers +=1
                    continue
                
                # Remove events with less than 8 hits
                if has_8_hits == False:
                    less_8_hits += 1
                    #print("event has less than 8 hits")
                    continue

                if L4_NoiseClassifier_ProbNu < 0.95:
                    probnu +=1
                    continue

                # regression variables
                # OUTPUT: [ nu energy, nu zenith, nu azimuth, nu time, nu x, nu y, nu z, track length (0 for cascade), isTrack, flavor, type (anti = 1), isCC, nu zenith (will not be transformed to cos zenith)]
                output_labels.append( np.array([ float(nu_energy), float(nu_zenith), float(nu_azimuth), float(nu_time), float(nu_x), float(nu_y), float(nu_z), float(track_length), float(isTrack), float(neutrino_type), float(particle_type), float(isCC), float(nu_zenith) ]) )

                if use_old_reco:
                    output_reco_labels.append( np.array([ float(reco_energy), float(reco_zenith), float(reco_azimuth), float(reco_time), float(reco_x), float(reco_y), float(reco_z), float(reco_length), float(reco_track_energy), float(reco_casc_energy), float(reco_em_casc_energy), float(reco_zenith) ]) )

                #Save weights
                #[File, RunID, SubrunID, EventID, NEvents, OneWeight, NormalizedOneWeight, GENIEWeight, InteractionProbabilityWeight, SinglePowerLawFlux_flux => 0.00154532, SinglePowerLawFlux_index, SinglePowerLawFlux_norm, SinglePowerLawFlux_weight, TotalInteractionProbabilityWeight, weight] 
                weights = frame['I3MCWeightDict']
                #from dragon_weights import weight_frame 
                if reco_type == "pegleg":
                    #the_weight = weight_frame(frame)
                    output_weights.append( np.array([ float(header.run_id), float(header.sub_run_id), float(header.event_id), float(weights["NEvents"]), float(weights["OneWeight"]),  float(weights["GENIEWeight"]), float(weights["PowerLawIndex"]), float(0.3)]) ) #, float(the_weight)  ]) )
                else:
                    output_weights.append( np.array([ float(header.run_id), float(header.sub_run_id), float(header.event_id), float(weights["NEvents"]), float(weights["OneWeight"]), float(weights["GENIEWeight"]),float(weights["PowerLawIndex"]), float(weights["gen_ratio"]), float(weights["weight"]) ]) )
            # close the input file once we are done
        

                output_features_DC.append(DC_array)
                output_features_IC.append(IC_near_DC_array)
                output_initial_stats.append(initial_stats)
                output_num_pulses_per_dom.append(num_pulses_per_dom)
                output_trigger_times.append(trig_time)

        del event_file

    output_features_DC=np.asarray(output_features_DC)
    output_features_IC=np.asarray(output_features_IC)
    output_labels=np.asarray(output_labels)
    output_initial_stats=np.asarray(output_initial_stats)
    output_num_pulses_per_dom=np.asarray(output_num_pulses_per_dom)
    output_trigger_times = np.asarray(output_trigger_times)
    output_weights = np.asarray(output_weights)
    if use_old_reco:
        output_reco_labels=np.asarray(output_reco_labels)

    if skipped_triggers > 0:
        print("Skipped %i events due to no or double triggers"%skipped_triggers)
    if less_8_hits > 0:
        print("Skipped %i events due to less than 8 hits"%less_8_hits)
    if probnu > 0:
        print("Skipped %i events due to ProbNu < 0.95"%probnu)
        

    return output_features_DC, output_features_IC, output_labels, output_reco_labels, output_initial_stats, output_num_pulses_per_dom, output_trigger_times, output_weights, ICstrings


#Construct list of filenames
import glob

event_file_names = sorted(glob.glob(input_file))
assert event_file_names,"No files loaded, please check path."
time_start=time.time()

features_DC, features_IC, labels, reco_labels, initial_stats, num_pulses_per_dom, trigger_times, weights, ICstrings = read_files(event_file_names, use_old_reco, check_filters, true_name, reco_type)

features_DC, features_IC, labels, output_label_names, input_transform_factors, output_transform_factors = apply_transform(features_DC, features_IC, output_labels=labels, energy_factor=efactor, track_factor=tfactor)

time_end=time.time()
print("Total time: %f"%(time_end-time_start))
print("Total events : %i"%labels.shape[0])

#Save output to hdf5 file
number_files = len(event_file_names)
num_file_name = ""
if number_files > 1:
    num_file_name = "_%sfiles"%number_files
if not output_name:
    output_name = event_file_names[0].split("/")[-1]
if use_cleaned_pulses:
    output_name += "_cleanedpulses"
output_path = output_dir + output_name + "_transformed_IC" + str(ICstrings) + num_file_name + ".hdf5"
f = h5py.File(output_path, "w")
f.create_dataset("features_DC", data=features_DC)
f.create_dataset("features_IC", data=features_IC)
f.create_dataset("labels", data=labels)
if use_old_reco:
    f.create_dataset("reco_labels", data=reco_labels)
f.create_dataset("initial_stats", data=initial_stats)
f.create_dataset("num_pulses_per_dom", data=num_pulses_per_dom)
f.create_dataset("trigger_times",data=trigger_times)
f.create_dataset("weights",data=weights)
f.attrs['output_label_names'] = [a.encode('utf8') for a in output_label_names]
f.create_dataset("output_label_names",data=f.attrs['output_label_names'])
f.create_dataset("input_transform_factors",data=input_transform_factors)
f.create_dataset("output_transform_factors",data=output_transform_factors)
f.close()

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

import numpy
import h5py
import argparse

from icecube import icetray, dataio, dataclasses
from I3Tray import I3Units

from collections import OrderedDict
import itertools
import random

## Create ability to change settings from terminal ##
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input",type=str,default='Level5_IC86.2013_genie_numu.014640.00000?.i3.bz2',
                    dest="input_file", help="path and name of the input file")
parser.add_argument("-n", "--name",type=str,default='Level5_IC86.2013_genie_numu.014640.00000X',
                    dest="output_name",help="name for output file (no path)")
parser.add_argument("-r", "--reco",type=str,default="False",
                    dest="reco", help="True if using Level5p or have a pegleg reco")
parser.add_argument("--emax",type=float,default=60.0,
                    dest="emax",help="Max energy to keep, cut anything above")
parser.add_argument("--vertex",type=str, default="DC",
                    dest="vertex_name",help="Name of vertex cut to put on file")
parser.add_argument("--cleaned",type=str,default="False",
                    dest="cleaned", help="True if wanted to use SRTTWOfflinePulsesDC")
parser.add_argument("--true_name",type=str,default=None,
                    dest="true_name", help="Name of key for true particle info if you want to check with I3MCTree[0]")
args = parser.parse_args()
input_file = args.input_file
output_name = args.output_name
emax = args.emax
vertex_name = args.vertex_name
true_name = args.true_name
if args.reco == 'True' or args.reco == 'true':
    use_old_reco = True
    print("Expecting old reco values in files, pulling from pegleg frames")
else:
    use_old_reco = False
if args.cleaned == "True" or args.cleaned == "true":
    use_cleaned_pulses = True
else:
    use_cleaned_pulses = False


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

    #Look inside ice pulses and get stats on charges and time
    # DC = deep core which is certain strings/DOMs in IceCube
    store_string = []
    #IC_near_DC_strings = [17, 18, 19, 25, 26, 27, 28, 34, 35, 36, 37, 38, 44, 45, 46, 47, 54, 55, 56]
    IC_near_DC_strings = [26, 27, 35, 36, 37, 45, 46]
    ICstrings = len(IC_near_DC_strings)

    # Adding outter rings to sum into single string
    if num_rings > 0:
        if ICstrings == 7:
            ring_start = 2
            assert num_rings<5, "Only 6 total rings (starting at ring 2 in this case), so rings are out of range" 
        elif ICstrings == 19:
            ring_start = 3 
            assert num_rings<4, "Only 6 total rings (starting at ring 3 in this case), so rings are out of range" 
        else:
            assert False, "Only supports 7 of 19 IC strings to add a ring to"
        rings = create_ring_dict()

    DC_strings = [79, 80, 81, 82, 83, 84, 85, 86]

    # Current Information: sum charges, time first pulse, Time of last pulse, Charge weighted mean time of pulses, Charge weighted standard deviation of pulse times
    # Old information: sum charges, sum charge <500ns, sum charge <100ns, time first pulse, time when 20 % of charge, time when 50% charge, Time of last pulse, Charge weighted mean time of pulses, Charge weighted standard deviation of pulse times
    array_DC = numpy.zeros([len(DC_strings),60,5]) #9 #take only DOMs in main region, not veto layer
    array_IC_near_DC = numpy.zeros([len(IC_near_DC_strings),60,5]) #9
    initial_stats = numpy.zeros([4])
    num_pulses_per_dom = numpy.zeros([len(DC_strings),60,1])
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

    #Start by making all times negative shift time (distinguish null from 0)
    array_DC[...,1:] = -20000
    array_IC_near_DC[...,1:] = -20000

    for omkey, pulselist in ice_pulses:
        dom_index =  omkey.om-1
        string_val = omkey.string
        timelist = []
        chargelist = []

        DC_flag = False
        IC_near_DC_flag = False
        IC_ring_flag = False

        
        for pulse in pulselist:
            
            if string_val not in store_string:
                store_string.append(string_val)

            # Check IceCube near DeepCore DOMs
            if( (string_val in IC_near_DC_strings) and dom_index<60):
                string_index = IC_near_DC_strings.index(string_val)
                timelist.append(pulse.time)
                chargelist.append(pulse.charge)
                IC_near_DC_flag = True

            # Check DeepCore DOMS
            elif ( (string_val in DC_strings) and dom_index<60): #dom_index >=10
                string_index = DC_strings.index(string_val)
                timelist.append(pulse.time)
                chargelist.append(pulse.charge)
                DC_flag = True

            elif num_rings > 0:
                for ring_check in range(ring_start,ring_start+num_rings):
                    if ((string_val in ring[ring_check]) and dom_index<60):
                        string_index = ICstrings+(ring_check-ring_start) #Put all pulses in ring into additional string, so ICstrings+1 for first additional ring, ICstrings +2 for second additional ring
                        timelist.append(pulse.time)
                        chargelist.append(pulse.charge)
                        IC_ring_flag = True


            else:
                count_outside +=1
                charge_outside += pulse.charge
                
        if DC_flag == True or IC_near_DC_flag == True:

            charge_array = numpy.array(chargelist)
            time_array = numpy.array(timelist)
            time_array = [ (t_value - shift_time_by) for t_value in time_array ]
            time_shifted = [ (t_value - time_array[0]) for t_value in time_array ]
            time_shifted = numpy.array(time_shifted)
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
            charge_array = numpy.array(charge_array_in_window)
            time_array = numpy.array(time_array_in_window)
            assert len(charge_array)==len(time_array), "Mismatched pulse time and charge"
            if len(charge_array) == 0:
                continue

            #Original Stats
            count_inside += len(chargelist)
            charge_inside += sum(chargelist)

            # Check that pulses are sorted
            for i_t,time in enumerate(time_array):
                assert time == sorted(time_array)[i_t], "Pulses are not pre-sorted!"

            # Find time when 20% and 50% of charge has hit DOM
            sum_charge = numpy.cumsum(chargelist)
            flag_20p = False
            for sum_index,current_charge in enumerate(sum_charge):
                if charge_array[-1] == 0:
                    time_20p = 0
                    time_50p = 0
                    break
                if current_charge/float(charge_array[-1]) > 0.2 and flag_20p == False:
                    time_20p = sum_index
                    flag_20p = True
                if current_charge/float(charge_array[-1]) > 0.5:
                    time_50p = sum_index
                    break            
            
            # Charge weighted mean and stdev
            weighted_avg_time = numpy.average(time_array,weights=charge_array)
            weighted_std_time = numpy.sqrt( numpy.average((time_array - weighted_avg_time)**2, weights=charge_array) )

            #print(dom_index,string_val,string_index,DC_flag,IC_near_DC_flag,len(chargelist),sum(chargelist), time_array[0],time_array[-1],weighted_avg_time,weighted_std_time)

        if DC_flag == True:
            array_DC[string_index,dom_index,0] = sum(chargelist)
            array_DC[string_index,dom_index,1] = time_array[0]
            array_DC[string_index,dom_index,2] = time_array[-1]
            array_DC[string_index,dom_index,3] = weighted_avg_time
            array_DC[string_index,dom_index,4] = weighted_std_time
            #array_DC[string_index,dom_index,1] = sum(charge_array[mask_500])
            #array_DC[string_index,dom_index,2] = sum(charge_array[mask_100])
            #array_DC[string_index,dom_index,4] = time_array[time_20p]
            #array_DC[string_index,dom_index,5] = time_array[time_50p]
            num_pulses_per_dom[string_index,dom_index,0] = len(chargelist)
        
        if IC_near_DC_flag == True:
            array_IC_near_DC[string_index,dom_index,0] = sum(chargelist)
            array_IC_near_DC[string_index,dom_index,1] = time_array[0]
            array_IC_near_DC[string_index,dom_index,2] = time_array[-1]
            array_IC_near_DC[string_index,dom_index,3] = weighted_avg_time
            array_IC_near_DC[string_index,dom_index,4] = weighted_std_time
            #array_IC_near_DC[string_index,dom_index,1] = sum(charge_array[mask_500])
            #array_IC_near_DC[string_index,dom_index,2] = sum(charge_array[mask_100])
            #array_IC_near_DC[string_index,dom_index,4] = time_array[time_20p]
            #array_IC_near_DC[string_index,dom_index,5] = time_array[time_50p]

    #print(count_outside, charge_outside, charge_outside/(sum(array_DC[:,:,0].flatten())+charge_outside))

    initial_stats[0] = count_outside
    initial_stats[1] = charge_outside
    initial_stats[2] = count_inside
    initial_stats[3] = charge_inside

    return array_DC, array_IC_near_DC, initial_stats, num_pulses_per_dom, trigger_time, num_extra_DC_triggers, ICstrings

def read_files(filename_list, drop_fraction_of_tracks=0.00, drop_fraction_of_cascades=0.00):
    """
    Read list of files, make sure they pass L5 cuts, create truth labels
    Receives:
        filename_list = list of strings, filenames to read data from
        drop_fraction_of_tracks = how many track events to drop
        drop_fraction_of_cascades = how many cascade events to drop
                                --> track & cascade not evenly simulated
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
    not_in_DC = 0
    isOther_count = 0

    for event_file_name in filename_list:
        print("reading file: {}".format(event_file_name))
        event_file = dataio.I3File(event_file_name)

        while event_file.more():
            try:
                frame = event_file.pop_physics()
            except:
                continue
            
            if frame["I3EventHeader"].sub_event_stream != "InIceSplit":
                continue

            # ALWAYS USE EVENTS THAT PASSES CLEANING!
            #if use_cleaned_pulses:
            try:
                cleaned = frame["SRTTWOfflinePulsesDC"]
            except:
                continue

            # some truth labels (we do *not* have these in real data and would like to figure out what they are)
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
            #track_length = frame["trueMuon"].length
            isTrack = frame['I3MCWeightDict']['InteractionType']==1.   # it is a cascade with a trac
            isCascade = frame['I3MCWeightDict']['InteractionType']==2. # it is just a cascade
            isCC = frame['I3MCWeightDict']['InteractionType']==1.
            isNC = frame['I3MCWeightDict']['InteractionType']==2.
            isOther = not isCC and not isNC
            
            if use_old_reco:

                if not frame.Has('IC86_Dunkman_L6_PegLeg_MultiNest8D_NumuCC'):
                    continue
                reco_nu = frame['IC86_Dunkman_L6_PegLeg_MultiNest8D_NumuCC']
                reco_energy = reco_nu.energy
                reco_time = reco_nu.time
                reco_zenith = reco_nu.dir.zenith
                reco_azimuth = reco_nu.dir.azimuth
                reco_x = reco_nu.pos.x
                reco_y = reco_nu.pos.y
                reco_z = reco_nu.pos.z

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
            
            # Decide how many track events to keep
            if isTrack and random.random() < drop_fraction_of_tracks:
                continue

            # Decide how many cascade events to kep
            if isCascade and random.random() < drop_fraction_of_cascades:
                continue

            # Only look at "low energy" events for now
            if nu_energy > emax:
                continue
            
            # Cut to only use events with true vertex in DeepCore
            if vertex_name == "IC19":
                #print("Using IC19 radius for cuts")
                radius = 300
            if vertex_name == "DC":
                #print("Using DC only for cuts")
                radius = 90
            x_origin = 54
            y_origin = -36
            shift_x = nu_x - x_origin
            shift_y = nu_y - y_origin
            z_val = nu_z
            radius_calculation = numpy.sqrt(shift_x**2+shift_y**2)
            if( radius_calculation > radius or z_val > 192 or z_val < -505 ):
                not_in_DC += 1
                continue

            
            DC_array, IC_near_DC_array,initial_stats,num_pulses_per_dom, trig_time, extra_triggers, ICstrings  = get_observable_features(frame)

            # Check if there were multiple SMT3 triggers or no SMT3 triggers
            # Skip event if so
            if extra_triggers > 0 or trig_time == None:
                continue

            # regression variables
            # OUTPUT: [ nu energy, nu zenith, nu azimuth, nu time, nu x, nu y, nu z, track length (0 for cascade), isTrack, flavor, type (anti = 1), isCC]

            output_labels.append( numpy.array([ float(nu_energy), float(nu_zenith), float(nu_azimuth), float(nu_time), float(nu_x), float(nu_y), float(nu_z), float(track_length), float(isTrack), float(neutrino_type), float(particle_type), float(isCC) ]) )

            if use_old_reco:
                output_reco_labels.append( numpy.array([ float(reco_energy), float(reco_zenith), float(reco_azimuth), float(reco_time), float(reco_x), float(reco_y), float(reco_z) ]) )

            
            output_features_DC.append(DC_array)
            output_features_IC.append(IC_near_DC_array)
            output_initial_stats.append(initial_stats)
            output_num_pulses_per_dom.append(num_pulses_per_dom)
            output_trigger_times.append(trig_time)
    
        print("Got rid of %i events not in DC so far"%not_in_DC)
        print("Got rid of %i events classified as other so far"%isOther_count)


        # close the input file once we are done
        del event_file

    output_features_DC=numpy.asarray(output_features_DC)
    output_features_IC=numpy.asarray(output_features_IC)
    output_labels=numpy.asarray(output_labels)
    output_initial_stats=numpy.asarray(output_initial_stats)
    output_num_pulses_per_dom=numpy.asarray(output_num_pulses_per_dom)
    output_trigger_times = numpy.asarray(output_trigger_times)
    if use_old_reco:
        output_reco_labels=numpy.asarray(output_reco_labels)

    print("Got rid of %i events not in DC in total"%not_in_DC)

    return output_features_DC, output_features_IC, output_labels, output_reco_labels, output_initial_stats, output_num_pulses_per_dom, output_trigger_times, ICstrings

#Construct list of filenames
import glob

file_name = input_file

event_file_names = sorted(glob.glob(file_name))
assert event_file_names,"No files loaded, please check path."

#Call function to read and label files
#Currently set to ONLY get track events, no cascades!!! #
features_DC, features_IC, labels, reco_labels, initial_stats, num_pulses_per_dom, output_trigger_times, ICstrings = read_files(event_file_names, drop_fraction_of_tracks=0.0,drop_fraction_of_cascades=0.0)

print(features_DC.shape)

#Save output to hdf5 file
output_path = "/mnt/scratch/micall12/training_files/" + output_name + "_lt" + str(int(emax)) + "_vertex" + vertex_name + "_IC" + str(ICstrings) + ".hdf5"
f = h5py.File(output_path, "w")
f.create_dataset("features_DC", data=features_DC)
f.create_dataset("features_IC", data=features_IC)
f.create_dataset("labels", data=labels)
if use_old_reco:
    f.create_dataset("reco_labels", data=reco_labels)
f.create_dataset("initial_stats", data=initial_stats)
f.create_dataset("num_pulses_per_dom", data=num_pulses_per_dom)
f.create_dataset("trigger_times",data=output_trigger_times)
f.close()

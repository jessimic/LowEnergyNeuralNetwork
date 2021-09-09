#############################
# Read IceCube files and create input feature arrays
#   get_observable_features = access specific data from i3 files for cnn
##############################

import numpy as np
from icecube import icetray, dataclasses

def get_observable_features(frame,low_window=-500,high_window=4000,use_cleaned_pulses=True,charge_min=0.25):
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
            if a_charge < charge_min:
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
                if charge < charge_min:
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


        initial_stats[0] = count_outside
        initial_stats[1] = charge_outside
        initial_stats[2] = count_inside
        initial_stats[3] = charge_inside
        
    return array_DC, array_IC_near_DC, initial_stats, num_pulses_per_dom, trigger_time, num_extra_DC_triggers, ICstrings, clean_pulses_8_or_more


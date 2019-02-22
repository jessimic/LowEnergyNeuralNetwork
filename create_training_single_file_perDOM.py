#############################
# Read IceCube files and create training file (hdf5)
#   Modified from code written by Claudio Kopper
#   get_observable_features = access data from IceCube files
#   read_files = read in files and add truth labels
#   Can take 1 or multiple files
#   Input:
#       -i input: name of input file, include path
#       -n name: name for output file, automatically puts in my scratch
##############################

## ASSUMPTIONS: All strings >=79 are DeepCore

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
                    dest="input_file", help="name of the input file")
parser.add_argument("-n", "--name",type=str,default='Level5_IC86.2013_genie_numu.014640.00000X',
                    dest="output_name",help="name for output file (no path)")
args = parser.parse_args()
input_file = args.input_file
output_name = args.output_name

def get_observable_features(frame):
    """
    Load observable features from IceCube files
    Receives:
        frame = IceCube object type from files
    Returns:
        observable_features: Observables dictionary
    """

    ice_pulses = dataclasses.I3RecoPulseSeriesMap.from_frame(frame,'InIcePulses')

    #Look inside ice pulses and get stats on charges and time
    # DC = deep core which is certain strings/DOMs in IceCube
    store_string = []
    IC_near_DC_strings = [26, 27, 35, 36, 37, 45, 46]
    DC_strings = [79, 80, 81, 82, 83, 84, 85, 86]

    # Current Information: sum charges, time first pulse, Time of last pulse, Charge weighted mean time of pulses, Charge weighted standard deviation of pulse times
    # Old information: sum charges, sum charge <500ns, sum charge <100ns, time first pulse, time when 20 % of charge, time when 50% charge, Time of last pulse, Charge weighted mean time of pulses, Charge weighted standard deviation of pulse times
    array_DC = numpy.zeros([len(DC_strings),60,5]) #9
    array_IC_near_DC = numpy.zeros([len(IC_near_DC_strings),60,5]) #9
    count_outside = 0
    charge_outside = 0

    for omkey, pulselist in ice_pulses:
        dom_index =  omkey.om-1
        string_val = omkey.string
        timelist = []
        chargelist = []
        DC_flag = False
        IC_near_DC_flag = False

        for pulse in pulselist:
            
            if string_val not in store_string:
                store_string.append(string_val)

            # Populate IceCube near DeepCore strings
            if( (string_val in IC_near_DC_strings) and dom_index<60):
                string_index = IC_near_DC_strings.index(string_val)
                timelist.append(pulse.time)
                chargelist.append(pulse.charge)
                IC_near_DC_flag = True

            # Populate DeepCore strings
            elif ( (string_val in DC_strings) and dom_index<60):
                string_index = DC_strings.index(string_val)
                timelist.append(pulse.time)
                chargelist.append(pulse.charge)
                DC_flag = True

            else:
                count_outside +=1
                charge_outside += pulse.charge
                
        if DC_flag == True or IC_near_DC_flag == True:


            charge_array = numpy.array(chargelist)
            time_array = numpy.array(timelist)
            time_shifted = [ (t_value - time_array[0]) for t_value in time_array ]
            time_shifted = numpy.array(time_shifted)
            mask_500 = time_shifted<500
            mask_100 = time_shifted<100

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

    print(count_outside, charge_outside, charge_outside/(sum(array_DC[:,:,0].flatten())+charge_outside))

    return array_DC, array_IC_near_DC 

def read_files(filename_list, drop_fraction_of_tracks=0.88, drop_fraction_of_cascades=0.00):
    """
    Read list of files, make sure they pass L5 cuts, create truth labels
    Receives:
        filename_list = list of strings, filenames to read data from
        drop_fraction_of_tracks = how many track events to drop
        drop_fraction_of_cascades = how many cascade events to drop
                                --> track & cascade not evenly simulated
    Returns:
        output_features = dict from observable features, passed to here
        output_labels = dict with output labels
    """
    output_features_DC = []
    output_features_IC = []
    output_labels = []

    for event_file_name in filename_list:
        print("reading file: {}".format(event_file_name))
        event_file = dataio.I3File(event_file_name)

        while event_file.more():
            frame = event_file.pop_physics()

            # some truth labels (we do *not* have these in real data and would like to figure out what they are)
            truth_labels = dict(
                neutrino = frame["trueNeutrino"],
                muon = frame['trueMuon'],
                cascade = frame['trueCascade'],
                nu_x = frame["trueNeutrino"].pos.x,
                nu_y = frame["trueNeutrino"].pos.y,
                nu_z = frame["trueNeutrino"].pos.z,
                nu_zenith = frame["trueNeutrino"].dir.zenith,
                nu_azimuth = frame["trueNeutrino"].dir.azimuth,
                nu_energy = frame["trueNeutrino"].energy,
                nu_time = frame["trueNeutrino"].time,
                track_length = frame["trueMuon"].length,
                isTrack = frame['I3MCWeightDict']['InteractionType']==1.,   # it is a cascade with a track
                isCascade = frame['I3MCWeightDict']['InteractionType']==2., # it is just a cascade
                isOther = frame['I3MCWeightDict']['InteractionType']!=1. and frame['I3MCWeightDict']['InteractionType']!=2., # it is something else (should not happen)
            )

            # input file sanity check: this should not print anything since "isOther" should always be false
            if truth_labels['isOther']:
                print(frame['I3MCWeightDict'])

            # Decide how many track events to keep
            if truth_labels['isTrack'] and random.random() < drop_fraction_of_tracks:
                continue

            # Decide how many cascade events to kep
            if truth_labels['isCascade'] and random.random() < drop_fraction_of_cascades:
                continue

            # Only look at "low energy" events for now
            if truth_labels["nu_energy"] > 60.0:
                                continue
            
            # Cut to only use events with true vertex in DeepCore
            radius = 90
            x_origin = 54
            y_origin = -36
            shift_x = truth_labels["nu_x"] - x_origin
            shift_y = truth_labels["nu_y"] - y_origin
            z_val = truth_labels["nu_z"]
            radius_calculation = numpy.sqrt(shift_x**2+shift_y**2)
            if( radius_calculation > radius or z_val > 192 or z_val < -505 ):
                continue

            
            # regression variables
            output_labels.append( numpy.array([ float(truth_labels['nu_energy']),float(truth_labels['nu_zenith']),float(truth_labels['nu_azimuth']),float(truth_labels['nu_time']),float(truth_labels['track_length']),float(truth_labels['nu_x']),float(truth_labels['nu_y']),float(truth_labels['nu_z']) ]) )

            DC_array, IC_near_DC_array = get_observable_features(frame)
            
            output_features_DC.append(DC_array)
            output_features_IC.append(IC_near_DC_array)


        # close the input file once we are done
        del event_file

    output_features_DC=numpy.asarray(output_features_DC)
    output_features_IC=numpy.asarray(output_features_IC)
    output_labels=numpy.asarray(output_labels)

    return output_features_DC, output_features_IC, output_labels

#Construct list of filenames
import glob

file_name = input_file

event_file_names = sorted(glob.glob(file_name))
assert event_file_names,"No files loaded, please check path."

#Call function to read and label files
#Currently set to ONLY get track events, no cascades!!! #
features_DC, features_IC, labels = read_files(event_file_names, drop_fraction_of_tracks=0.0,drop_fraction_of_cascades=1.0)

print(features_DC.shape)

#Save output to hdf5 file
output_path = "/mnt/scratch/micall12/training_files/" + output_name + ".hdf5"
f = h5py.File(output_path, "w")
f.create_dataset("features_DC", data=features_DC)
f.create_dataset("features_IC", data=features_IC)
f.create_dataset("labels", data=labels)
f.close()

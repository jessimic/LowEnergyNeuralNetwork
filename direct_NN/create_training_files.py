#############################
# Read IceCube files and create training file (hdf5)
#   Modified from code written by Claudio Kopper
#   get_observable_features = access data from IceCube files
#   read_files = read in files and add truth labels
##############################

import numpy
import h5py

from icecube import icetray, dataio, dataclasses
from I3Tray import I3Units

from collections import OrderedDict
import itertools
import random

def get_observable_features(frame):
    """
    Load observable features from IceCube files
    Receives:
        frame = IceCube object type from files
    Returns:
        observable_features: Observables dictionary
    """

    l4_vars = frame["IC86_Dunkman_L4"]
    l5_vars = frame["IC86_Dunkman_L5"]
    
    ice_pulses = dataclasses.I3RecoPulseSeriesMap.from_frame(frame,'InIcePulses')

    #Look inside ice pulses and get stats on charges and time
    # DC = deep core which is certain strings/DOMs in IceCube
    DCcharge = 0
    timelist = []
    chargelist = []
    DC_timelist = []
    DC_chargelist = []
    store_string = []
    for omkey, pulselist in ice_pulses:
        for pulse in pulselist:
            timelist.append(pulse.time)
            chargelist.append(pulse.charge)
            #Check deep core
            if omkey.string not in store_string:
                store_string.append(omkey.string)
            if(omkey.string==36 and omkey.om >= 42 and omkey.om<=60) or (omkey.string>=79 and omkey.om >= 14 and omkey.om<=60):
                DC_timelist.append(pulse.time)
                DC_chargelist.append(pulse.charge)

    direct_charge = frame['IC86_Dunkman_L6_SANTA_DirectCharge']
    direct_doms = frame['IC86_Dunkman_L6_SANTA_DirectDOMs']
    
    direct_pulses = dataclasses.I3RecoPulseSeriesMap.from_frame(frame,'IC86_Dunkman_L6_SANTA_DirectPulses')
    num_direct_pulses = 0
    for omkey, pulselist in direct_pulses:
        for pulse in pulselist:
            num_direct_pulses += 1


    observable_features_dict = OrderedDict(
        
        C2QR3 = l5_vars['C2QR3'],
        QR3 = l5_vars['QR3'],
        bdt_score = l5_vars['bdt_score'],
        cog_q1_rho = l5_vars['cog_q1_rho'],
        cog_q1_z = l5_vars['cog_q1_z'],
        linefit_speed = l5_vars['linefit_speed'],
        linefit_zenith = l5_vars['linefit_zenith'],
        num_hit_doms = l5_vars['num_hit_doms'],
        separation = l5_vars['separation'],
        spe11_zenith = l5_vars['spe11_zenith'],
        total_charge = l5_vars['total_charge'],
        z_spread = l5_vars['z_spread'],

        cog_sigma_time = l4_vars['cog_sigma_time'],
        cog_sigma_z = l4_vars['cog_sigma_z'],
        dcc_veto_charge = l4_vars['dcc_veto_charge'],
        first_hlc_rho = l4_vars['first_hlc_rho'],
        first_hlc_z = l4_vars['first_hlc_z'],
        interval = l4_vars['interval'],
        result_invertedVICH = l4_vars['result_invertedVICH'],
        rt_fid_charge = l4_vars['rt_fid_charge'],
        vich_charge = l4_vars['vich_charge'],

        direct_charge = direct_charge.value,
        direct_DOMs = direct_doms.value,
        direct_pulses = num_direct_pulses,
        
        #time_range = max(timelist)-min(timelist),
        #charge_total = sum(chargelist),
        #number_pulses = len(chargelist),
        #charge_max = max(chargelist),
        #charge_median = numpy.median(chargelist),
        #number_strings = len(store_string),
        #charge_min = min(chargelist),
        #charge_range = max(chargelist)-min(chargelist),
    
        #DC_time_range = max(DC_timelist)-min(DC_timelist),
        #DC_number_pulses = len(DC_chargelist),
        #DC_charge_total = sum(DC_chargelist),
        #DC_charge_max = max(DC_chargelist),
        #DC_charge_min = min(DC_chargelist),
        #DC_charge_range = max(DC_chargelist)-min(DC_chargelist),
        
    )
    

    keys = [k for k in observable_features_dict.keys()]
    observable_features = numpy.array(
        [tuple(v for v in observable_features_dict.values())],
        dtype=numpy.dtype(
            {'names': keys,
             'formats': list(itertools.repeat(numpy.float32, len(keys))),
            }
        )
    )

    #print("ARRAY_np:", observable_features.view((numpy.float32, len(observable_features.dtype.names))))
    #print("ARRAY:",    observable_features)

    return observable_features    

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
    output_features = None
    output_labels = []

    for event_file_name in filename_list:
        print("reading file: {}".format(event_file_name))
        event_file = dataio.I3File(event_file_name)

        while event_file.more():
            frame = event_file.pop_physics()

            ###Level 5 cuts### (specific to these input files)
            # leave only "Level 6" events, skip all others
            if frame.Has("IC86_Dunkman_L3"):
                pass_L3 = (frame['IC86_Dunkman_L3'].value)
            else:
                pass_L3 = False
            if not pass_L3:
                continue

            if "IC86_Dunkman_L4" not in frame:
                continue

            pass_L4_1 = (frame['IC86_Dunkman_L4']['result'] == 1.)
            if not pass_L4_1:
                continue

            if "IC86_Dunkman_L5" not in frame:
                continue

            pass_L5 = (frame['IC86_Dunkman_L5']['bdt_score'] >= 0.1)
            if not pass_L5:
                continue
                
            ###End Check Level 5 cuts####

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

            
            # multiple categories
            # output_labels.append( numpy.array([float(truth_labels['nu_x']), float(truth_labels['nu_y']),float(truth_labels['nu_z']), \
            #                        float(truth_labels['nu_zenith']),float(truth_labels['nu_azimyth']),\
            #                        float(truth_labels['nu_energy']),float(truth_labels['nu_time']),float(truth_labels['track_length']) ]) )
            #Let's start regression with ONE category
            output_labels.append( numpy.array([ float(truth_labels['nu_energy']) ]) )

            observable_features = get_observable_features(frame)
            if output_features is None:
                output_features = observable_features
            else:
                output_features = numpy.append(output_features, observable_features)


        # close the input file once we are done
        del event_file

    output_features=numpy.asarray(output_features)
    output_labels=numpy.asarray(output_labels)

    return output_features, output_labels

#Construct list of filenames
import glob

event_file_names = sorted(glob.glob("/mnt/research/IceCube/jpandre/Matt/level5/numu/14640/Level5_IC86.2013_genie_numu.014640.000???.i3.bz2"))
assert event_file_names,"No files loaded, please check path."

#Call function to read and label files
#Currently set to ONLY get track events, no cascades!!! #
features, labels = read_files(event_file_names, drop_fraction_of_tracks=0.0,drop_fraction_of_cascades=1.0)

#Save output to hdf5 file
f = h5py.File("Level5_IC86.2013_genie_numu.014640.000XXX_calc.hdf5", "w")
f.create_dataset("features", data=features)
f.create_dataset("labels", data=labels)
f.close()

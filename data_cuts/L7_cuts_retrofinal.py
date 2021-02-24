import math, os, copy

import numpy as np

from icecube import dataclasses
from icecube import icetray
from icecube.icetray import I3Units
from icecube import lilliput
import icecube.lilliput.segments
# icetray.logging.console() #TODO Breaks some stuff, think about this
from icecube.oscNext.tools.data_quality import check_object_exists
from icecube.oscNext.selection.globals import *
from icecube.oscNext.selection.oscNext_cuts import oscNext_cut
from icecube.oscNext.selection.oscNext_L6 import RETRO_RECO_PARTICLE_KEY, RETRO_FIT_METHOD, check_retro_reco_success



#
# Globals
#

# Define all output frame objects
L7_HDF5_KEYS = []

# Cut
L7_HDF5_KEYS.append( L7_CUT_BOOL_KEY )

# STV
L7_STV_LE_PREFIX = "L7_STV_LE"
L7_STV_LE_PROB_MUON_KEY = L7_STV_LE_PREFIX + "_Pm"
L7_HDF5_KEYS.extend([ L7_STV_LE_PROB_MUON_KEY, ])

# Track hits
L7_TRACK_HITS_PREFIX = "L7_TrackHits"
L7_TRACK_HITS_COMP_CHARGE_KEY = L7_TRACK_HITS_PREFIX + "_CompCharge"
L7_TRACK_HITS_COMP_HITS_KEY = L7_TRACK_HITS_PREFIX + "_CompHits"
L7_HDF5_KEYS.extend([ L7_TRACK_HITS_COMP_CHARGE_KEY, L7_TRACK_HITS_COMP_HITS_KEY ])

# Corridor cut (narrow) + angular correlation
L7_NARROW_CORRIDOR_CUT_PULSES_KEY = "L7_NarrowCorridorCutPulses"
L7_NARROW_CORRIDOR_CUT_COUNT_KEY = "L7_NarrowCorridorCutCount"
L7_NARROW_CORRIDOR_CUT_TRACK_KEY  = "L7_NarrowCorridorCutTrack"
L7_HDF5_KEYS.extend([ L7_NARROW_CORRIDOR_CUT_COUNT_KEY, L7_NARROW_CORRIDOR_CUT_TRACK_KEY ])
L7_HDF5_KEYS.extend([ 
    L7_NARROW_CORRIDOR_CUT_PULSES_KEY, 
    L7_NARROW_CORRIDOR_CUT_PULSES_KEY+"HitMultiplicity", 
    L7_NARROW_CORRIDOR_CUT_PULSES_KEY+"HitStatistics", 
    L7_NARROW_CORRIDOR_CUT_PULSES_KEY+"TimeCharacteristics",
])
L7_NARROW_CORRIDOR_TRACK_ANGLE_DIFF_KEY = L7_NARROW_CORRIDOR_CUT_TRACK_KEY + "_" + RETRO_RECO_PARTICLE_KEY + "_angles"
L7_HDF5_KEYS.append( L7_NARROW_CORRIDOR_TRACK_ANGLE_DIFF_KEY )

# Cuts/classifiers
L7_MUON_CLASSIFIER_MODEL_FILE = 'L7_classifier_v2.joblib'
L7_MUON_MODEL_PREDICTION_KEY = "L7_MuonClassifier_ProbNu"
L7_DATA_QUALITY_CUT_BOOL_KEY = "L7_data_quality_cut"
L7_MUON_CUT_BOOL_KEY = "L7_muon_cut"
L7_NOISE_CUT_BOOL_KEY = "L7_noise_cut"
L7_HDF5_KEYS.extend([L7_MUON_MODEL_PREDICTION_KEY, L7_DATA_QUALITY_CUT_BOOL_KEY, L7_MUON_CUT_BOOL_KEY,L7_NOISE_CUT_BOOL_KEY ])

# PID
L7_PID_CLASSIFIER_MODEL_FILE = 'pid_model_train_26FEB20_ready-for-processing_unosc-honda_5vars.joblib'
L7_PID_MODEL_PREDICTION_KEY = "L7_PIDClassifier_ProbTrack"
L7_HDF5_KEYS.append(L7_PID_MODEL_PREDICTION_KEY)

# Coincident muons
L7_COINCIDENT_REJECTION_VARS_KEY = "L7_CoincidentMuon_Variables"
L7_COINCIDENT_REJECTION_BOOL_KEY = "L7_CoincidentMuon_bool"
L7_HDF5_KEYS.extend([L7_COINCIDENT_REJECTION_VARS_KEY, L7_COINCIDENT_REJECTION_BOOL_KEY])

# Some containment stuff
L7_RETRO_RECO_RHO36_KEY = "L7_retro_crs_prefit__rho36"
L7_GRECO_CONTAINMENT_KEY = "L7_greco_containment"
L7_HDF5_KEYS.extend([L7_RETRO_RECO_RHO36_KEY, L7_GRECO_CONTAINMENT_KEY])

# Keep track of best SANTA fit
SANTA_BEST_KEY = "L6_SANTA_sel_Particle"
L7_HDF5_KEYS.append(SANTA_BEST_KEY)

# Compare recos
L7_SANTA_RETRO_ANGLE_DIFF_KEY = "L7_santa_retro_angles"
L7_HDF5_KEYS.append(L7_SANTA_RETRO_ANGLE_DIFF_KEY)

# Post-processed retro reco variables
#TODO replace 'retro_crs_prefit' with RETRO_FIT_METHOD
for sign in [ "p", "n", "tot" ] :
    L7_HDF5_KEYS.extend([
        "L7_retro_crs_prefit__azimuth_sigma_%s"%sign,
        "L7_retro_crs_prefit__zenith_sigma_%s"%sign,
        "L7_retro_crs_prefit__x_sigma_%s"%sign,
        "L7_retro_crs_prefit__y_sigma_%s"%sign,
        "L7_retro_crs_prefit__z_sigma_%s"%sign,
        "L7_retro_crs_prefit__time_sigma_%s"%sign,
        "L7_retro_crs_prefit__cascade_energy_sigma_%s"%sign,
        "L7_retro_crs_prefit__track_energy_sigma_%s"%sign,
        "L7_retro_crs_prefit__energy_sigma_%s"%sign,
        "L7_retro_crs_prefit__zero_dllh_sigma_%s"%sign,
])
L7_HDF5_KEYS.extend([
    "L7_max_postproc_llh_over_nstring",
    "L7_max_postproc_llh_over_nch",
    "L7_max_postproc_llh_over_nch_retro",
    "L7_nchannel_used_in_retro",
])

# Final reco variables
L7_FINAL_RECO_PREFIX = "L7_reconstructed"

L7_FINAL_RECO_X_KEY = L7_FINAL_RECO_PREFIX + "_vertex_x"
L7_FINAL_RECO_Y_KEY = L7_FINAL_RECO_PREFIX + "_vertex_y"
L7_FINAL_RECO_Z_KEY = L7_FINAL_RECO_PREFIX + "_vertex_z"
L7_FINAL_RECO_RHO36_KEY = L7_FINAL_RECO_PREFIX + "_vertex_rho36"
L7_FINAL_RECO_TIME_KEY = L7_FINAL_RECO_PREFIX + "_time"
L7_FINAL_RECO_ZENITH_KEY = L7_FINAL_RECO_PREFIX + "_zenith"
L7_FINAL_RECO_AZIMUTH_KEY = L7_FINAL_RECO_PREFIX + "_azimuth"
L7_FINAL_RECO_TRACK_LENGTH_KEY = L7_FINAL_RECO_PREFIX + "_track_length"
L7_FINAL_RECO_TRACK_ENERGY_KEY = L7_FINAL_RECO_PREFIX + "_track_energy"
L7_FINAL_RECO_CASCADE_ENERGY_KEY = L7_FINAL_RECO_PREFIX + "_cascade_energy"
L7_FINAL_RECO_EM_CASCADE_ENERGY_KEY = L7_FINAL_RECO_PREFIX + "_em_cascade_energy"
L7_FINAL_RECO_TOTAL_ENERGY_KEY = L7_FINAL_RECO_PREFIX + "_total_energy"

L7_HDF5_KEYS.extend([
    L7_FINAL_RECO_X_KEY,
    L7_FINAL_RECO_Y_KEY,
    L7_FINAL_RECO_Z_KEY,
    L7_FINAL_RECO_RHO36_KEY,
    L7_FINAL_RECO_TIME_KEY,
    L7_FINAL_RECO_ZENITH_KEY,
    L7_FINAL_RECO_AZIMUTH_KEY,
    L7_FINAL_RECO_TRACK_LENGTH_KEY,
    L7_FINAL_RECO_TRACK_ENERGY_KEY,
    L7_FINAL_RECO_EM_CASCADE_ENERGY_KEY,
    L7_FINAL_RECO_CASCADE_ENERGY_KEY,
    L7_FINAL_RECO_TOTAL_ENERGY_KEY,
])

# Photon speed
for suffix in ["PhotonSpeedMetrics", "PhotonDisplacement", "PhotonTimeTaken", "PhotonSpeed"] :
    for prefix in ["AllPhotons", "PhysicalPhotonsOnly"] :
        L7_HDF5_KEYS.append( "L7_%s_%s" % (prefix, suffix) )

#
# Reco
#

def oscNext_L7_compute_final_reconstructed_values(frame, cleaned_pulses):
    '''
    Comoute the final reconsturcted values we will use for analysis.

    We use RetroReco as the final level reconstruction, and apply a number of improved conversion
    factors and correction factors to get good agreement with the truth parameters.

    Also do a bunch of other post-processing such as:
      - Computing reduced LLH
      - Extract uncertainties on reconstructed quantities
      - Compute derived quantities like rho36
    '''

    from icecube.dataclasses import I3Double, I3MapStringDouble
    from icecube.oscNext.frame_objects.geom import calc_rho_36
    from icecube.oscNext.frame_objects.reco import convert_retro_reco_energy_to_neutrino_energy
    from retro.i3processing.retro_recos_to_i3files import const_en2len, const_en_to_gms_en, GMS_LEN2EN

    #
    # Compute reco rho36
    #

    retro_rho36 = I3MapStringDouble()

    for key in ['mean','median','upper_bound','lower_bound','max']:
        x = frame['retro_crs_prefit__x'][key]
        y = frame['retro_crs_prefit__y'][key]
        z = frame['retro_crs_prefit__z'][key]
        r36 = calc_rho_36(x=x, y=y)
        retro_rho36[key] = r36

    frame[L7_RETRO_RECO_RHO36_KEY] = retro_rho36


    #
    # Track length/energy
    #

    #
    # To get the length, we have to reverse-compute it from the
    # level6 track_energy estimate that is already available in the frame
    # This energy was computed using a constant light-emission approximation
    # so the length is simply the energy of the track divided by a constant 
    # factor of 0.22 GeV/m
    #

    track_energy_const = frame["retro_crs_prefit__track_energy"]["median"]
    track_length = const_en2len(track_energy_const)


    #
    # Overall energy
    #

    # The cascade energy fitted by retro reco assumed an EM cascade 
    frame['L7_reconstructed_em_cascade_energy'] = I3Double(frame["retro_crs_prefit__cascade_energy"]["median"])

    # Some additional steps are required to get the final energy values from the RetroReco fits
    #   1) Convert from EM energy to hadronic-equivalent energy 
    #   2) Apply some scaling factors to the hadronic cascade energy and the track length to better match the truth params on average
    #   3) Compute track energy again using the corrected length, and also now using the GMS tables which work better than the constant energy loss approximatiin 
    #   4) Sum the track and cascade energy to get the total energy
    cascade_energy, track_energy, total_energy, track_length = convert_retro_reco_energy_to_neutrino_energy(
        em_cascade_energy=frame['L7_reconstructed_em_cascade_energy'].value,
        track_length=track_length,
    )

    # Write final variable to frame
    frame[L7_FINAL_RECO_TRACK_LENGTH_KEY]         = I3Double(track_length)
    frame[L7_FINAL_RECO_CASCADE_ENERGY_KEY] = I3Double(cascade_energy)
    frame[L7_FINAL_RECO_TRACK_ENERGY_KEY]   = I3Double(track_energy)
    frame[L7_FINAL_RECO_TOTAL_ENERGY_KEY]   = I3Double(total_energy)


    #
    # Store the other reconstructed params
    #

    # Directly using the retro reco value for all other params
    frame[L7_FINAL_RECO_X_KEY]        = I3Double(frame['retro_crs_prefit__x']['median'])
    frame[L7_FINAL_RECO_Y_KEY]        = I3Double(frame['retro_crs_prefit__y']['median'])
    frame[L7_FINAL_RECO_Z_KEY]        = I3Double(frame['retro_crs_prefit__z']['median'])
    frame[L7_FINAL_RECO_RHO36_KEY]    = I3Double(frame[L7_RETRO_RECO_RHO36_KEY]['median'])
    frame[L7_FINAL_RECO_ZENITH_KEY]   = I3Double(frame['retro_crs_prefit__zenith']['median'])
    frame[L7_FINAL_RECO_AZIMUTH_KEY]  = I3Double(frame['retro_crs_prefit__azimuth']['median'])
    frame[L7_FINAL_RECO_TIME_KEY]     = I3Double(frame['retro_crs_prefit__time']['median'])


    #
    # Compute the llh-based error
    #

    # Get the uncertainties on the RetroReco reconstructed quantities

    for qty in ['azimuth','zenith','x','y','z','cascade_energy','track_energy','energy','time','zero_dllh']: 

        ub = frame['retro_crs_prefit__%s'%(qty)]['upper_bound']
        lb = frame['retro_crs_prefit__%s'%(qty)]['lower_bound']
        md = frame['retro_crs_prefit__%s'%(qty)]['median']

        p1s = ub - md # Positive 1 sigma
        n1s = md - lb # Negative 1 sigma
        w1s = p1s + n1s # Full 1 sigma width (-1sigma -> +1sigma)

        frame['L7_retro_crs_prefit__%s_sigma_p'%(qty)] = I3Double(p1s)
        frame['L7_retro_crs_prefit__%s_sigma_n'%(qty)] = I3Double(n1s)
        frame['L7_retro_crs_prefit__%s_sigma_tot'%(qty)] = I3Double(w1s)


    #
    # Compute the Reduced Goodness-of-fit
    #

    # Want to reduce compute the redued LLH value from RetroReco
    # This mans dividng it by the umber of elements used in the reco, e.g. num DOMs
    # Need to re-produce the pulse pre-cleaning RetroReco does in order to get the right value

    if frame.Has(cleaned_pulses):

        try: pulse_serie = frame[cleaned_pulses].apply(frame)
        except: pulse_serie = frame[cleaned_pulses]

        nch = 0
        nch_retro = 0
        nstring_triggered = []

        
        for dom in pulse_serie.keys():
        
            # just making sure we don't catch stuff from iceTop...
            if dom.om<=60:

                nch  += len(pulse_serie[dom]) > 0
                nstring_triggered.append(dom.string)

                #
                # Apply the precleaning that Retro Reco uses
                #

                # Quantize the charge in bins of 0.05
                quanta = 0.05
                quantized_charge = np.array([(np.float64(x.charge) // quanta) * quanta + quanta / 2. for x in pulse_serie[dom]])
                nch_retro += sum(quantized_charge>=0.3)!=0 # only add the channel if there is 1 pulse above 0.3 pe or more

        GOF = frame['retro_crs_prefit__max_postproc_llh'].value
        nstrings = float(len(np.unique(nstring_triggered)))
        frame['L7_max_postproc_llh_over_nstring'] = I3Double(GOF/nstrings)
        frame['L7_max_postproc_llh_over_nch'] = I3Double(GOF/float(nch))
        frame['L7_nchannel_used_in_retro'] = I3Double(float(nch_retro))
        if nch_retro<=0:
            frame['L7_max_postproc_llh_over_nch_retro'] = I3Double(np.NaN)
        else:
            frame['L7_max_postproc_llh_over_nch_retro'] = I3Double(GOF/float(nch_retro))

    else:
        frame['L7_max_postproc_llh_over_nstring'] = I3Double(np.NaN)
        frame['L7_max_postproc_llh_over_nch'] = I3Double(np.NaN)
        frame['L7_nchannel_used_in_retro'] = I3Double(np.NaN)
        frame['L7_max_postproc_llh_over_nch_retro'] = I3Double(np.NaN)

def oscNext_L7( 
    tray, name, 
    uncleaned_pulses,
    cleaned_pulses,
    reco_particle=RETRO_RECO_PARTICLE_KEY,
    debug=False,
):
    '''
    This is the main oscNext L7 traysegment
    '''
    if debug:
        icetray.set_log_level(icetray.I3LogLevel.LOG_DEBUG)
    else:
        icetray.set_log_level(icetray.I3LogLevel.LOG_ERROR)

    #
    # Data quality
    #
    
    # Check the required pulse series are present
    tray.Add(check_object_exists,object_key=uncleaned_pulses)
    tray.Add(check_object_exists,object_key=cleaned_pulses)

    #
    # Apply cuts on input data
    #

    # Only keep frames passing L6
    tray.Add(oscNext_cut, "L6_cut", processing_level=6)

    # Only keep frames where retro reco was successful
    #TODO This is now part of L6 but some files were run before this
    tray.Add(check_retro_reco_success, "check_retro_reco_success")
    

    #
    # Compute L7 variables
    #

    # Get final reco values
    tray.Add(
        oscNext_L7_compute_final_reconstructed_values,
        'oscNext_L7_compute_final_reconstructed_values',
        cleaned_pulses=cleaned_pulses,
    )

    # Find coincident events (unsimulated)
    #tray.Add(
    #    oscNext_L7_CoincidentCut,
    #    'oscNext_L7_CoincidentCut',
    #    uncleaned_pulses=uncleaned_pulses,
    #    cleaned_pulses=cleaned_pulses,
    #)

    # Compare to older samples
    tray.Add(
        oscNext_L7_compute_comparisons_to_older_samples,
        'oscNext_L7_compute_comparisons_to_older_samples',
    )

    # Compare different reconstructions
    tray.Add(
        oscNext_L7_compute_reconstruction_comparisons,
        'oscNext_L7_compute_reconstruction_comparisons',
    )

    # Calculate STV Variables
    #tray.Add(
    #    oscNext_L7_STV, 
    #    "oscNext_L7_STV_InfTrack_DC", 
     #   uncleaned_pulses=uncleaned_pulses,
     #   cleaned_pulses=cleaned_pulses,
     #   reco_particle=reco_particle,
        reco_shape=dataclasses.I3Particle.InfiniteTrack, # Primary is the default shape of retro-reco particles
        include_DeepCore = True,
    )

    # Corridor cut
    # Compute a version of the corridor cut with much narroweer corridor definitions compared to L5
    # This is more like what was done in DRAGON/GRECO
    tray.Add(
        oscNext_L7_CorridorCuts,
        'oscNext_L7_NarrowCorrCuts',
        uncleaned_pulses=uncleaned_pulses,
        cleaned_pulses=cleaned_pulses,
        reco_particle=reco_particle,
    )

    # Photon speeds
    tray.Add(
        oscNext_L7_photon_speed_metrics,
        "oscNext_L7_photon_speed_metrics",
        cleaned_pulses=cleaned_pulses, 
    )

    # PID
    tray.Add(
        oscNext_L7_compute_pid, 
        "oscNext_L7_PID", 
    )


    #
    # L7 cut
    #

    # Compute cut
    tray.Add(
        oscNext_L7_compute_cut, 
        "oscNext_L7_cut", 
    )


    #
    # Done
    #

    # Can dump frame for debugging
    if False :
        tray.Add("Dump","Dump")

    # Easier reading
    if debug:
        def add_line_break(frame):
            print "\n*********************\n"
        tray.Add(add_line_break)

    return
    return

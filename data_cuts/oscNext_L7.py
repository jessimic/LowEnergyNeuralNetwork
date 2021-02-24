'''
The oscNext level 7 event selection traysegment.
This is the final stage of atmospheric muon removal, using the final 
level reconstructions as inputs.

Etienne Bourbeau, Tom Stuttard
'''


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



def compute_greco_style_containment(rho36, z):
    '''
    Compute the containment cuts from the GRECO sample
    '''
    greco_boxcut = np.array(z<=-4.4*rho36+166.).astype(int)
    greco_r36cut = np.array(rho36<140.).astype(int)
    greco_zcut   = np.array(z<-230.).astype(int)

    return greco_boxcut*greco_r36cut*greco_zcut


def oscNext_L7_compute_comparisons_to_older_samples(frame):
    '''
    Add some comparisons to methods used in older samples
    '''

    from icecube.icetray import I3Bool

    #
    # GRECO
    #

    # Compute final level containment cut from GRECO
    greco_containment = compute_greco_style_containment(
        rho36=frame[L7_FINAL_RECO_RHO36_KEY].value,
        z=frame[L7_FINAL_RECO_Z_KEY].value,
    )

    frame[L7_GRECO_CONTAINMENT_KEY] = I3Bool(greco_containment)


def oscNext_L7_compute_reconstruction_comparisons(frame):
    '''
    Compare RetroReco and SANTA
    '''

    from icecube.dataclasses import I3Direction
    from icecube.oscNext.frame_objects.geom import angular_comparison


    #
    # Compute the angle between Retro Santa recostructed directions
    #

    # Get best SANTA fit
    santa_bestfit = frame[SANTA_BEST_KEY] if frame.Has(SANTA_BEST_KEY) else None

    # Get directions to compare
    santa_bestfit_dir = santa_bestfit.dir if santa_bestfit is not None else I3Direction() # Null if nothing is there
    retro_bestfit_dir = I3Direction(frame[L7_FINAL_RECO_ZENITH_KEY].value, frame[L7_FINAL_RECO_AZIMUTH_KEY].value)

    # Compare
    frame[L7_SANTA_RETRO_ANGLE_DIFF_KEY] = angular_comparison(santa_bestfit_dir, retro_bestfit_dir)



def check_reco_status(frame,reco_particle) :
    '''
    Function to check if reconstruction was successful
    '''

    from icecube.dataclasses import I3Particle
    reco_success = False

    if reco_particle in frame :
        reco_success = frame[reco_particle].fit_status == 0

    return reco_success



def Reset_retro_reco_shape(frame,reco_particle=None,new_shape=None,new_name=None):

    from icecube import dataclasses

    if frame.Has(reco_particle):

        P = frame[reco_particle].__copy__()

        icetray.logging.log_debug('Reset_retro_reco_shape: '+'Old Shape: %s'%P.shape)

        assert isinstance(new_shape,dataclasses.I3Particle.ParticleShape), 'ERROR: new shape is not an instance of dataclasses.ParticleShape'
        P.shape=new_shape
        icetray.logging.log_debug('Reset_retro_reco_shape: New Shape: %s'%P.shape)

        if new_name is None:
            frame['Reshaped_'+reco_particle] = P

        else:
            assert isinstance(new_name,str),"ERROR: new_name must be a string"

            frame[new_name] = P
            icetray.logging.log_debug('Reset_retro_reco_shape: New Shape: %i ---> %s'%(frame[new_name].shape,new_name))



def check_that_shape_has_changed(frame,particle_name=None):

    if frame.Has(particle_name):
        P = frame[particle_name]
        icetray.logging.log_debug('check_that_shape_has_changed: New shape: %s'%P.shape)



#
# Coicident events
#

def calculate_coincident_cut_containment(frame, pulsesname, outputname):
    '''Calculate containment variables for a given hit map.
    These variables are used for co-incident muon rejection.
    '''
    # taken from http://wiki.icecube.wisc.edu/index.php/Deployment_order
    ic86 = [21, 29, 39, 38, 30, 40, 50, 59, 49, 58, 67, 66, 74, 73, 65, 72, 78, 48, 57, 47,
            46, 56, 63, 64, 55, 71, 70, 76, 77, 75, 69, 60, 68, 61, 62, 52, 44, 53, 54, 45,
            18, 27, 36, 28, 19, 20, 13, 12, 6, 5, 11, 4, 10, 3, 2, 83, 37, 26, 17, 8, 9, 16,
            25, 85, 84, 82, 81, 86, 35, 34, 24, 15, 23, 33, 43, 32, 42, 41, 51, 31, 22, 14,
            7, 1, 79, 80]
    deep_core_strings = [79, 80, 81, 82, 83, 84, 85, 86]
    outer_strings = [31, 41, 51, 60, 68, 75, 76, 77, 78, 72, 73, 74, 67, 59, 50,
                     40, 30, 21, 13, 6, 5, 4, 3, 2, 1, 7, 14, 22]
    vars = dataclasses.I3MapStringDouble()
    vars['n_top15'] = 0.
    vars['n_top10'] = 0.
    vars['n_top5'] = 0.
    vars['z_travel_top15'] = 0.
    vars['n_outer'] = 0.
    if type(frame[pulsesname]) == dataclasses.I3RecoPulseSeriesMap:
        hit_map = frame.Get(pulsesname)
    elif type(frame[pulsesname]) == dataclasses.I3RecoPulseSeriesMapMask:
        hit_map = frame[pulsesname].apply(frame)
    omgeo = frame['I3Geometry'].omgeo
    z_pulses = []
    t_pulses = []
    for om in hit_map.keys():
        if om.string in ic86 and om.string not in deep_core_strings:
            if om.om <= 15:
                vars['n_top15'] += 1.
                z_pulses.append(omgeo[om].position.z)
                t_pulses.append(np.min([p.time for p in hit_map[om]]))
            if om.om <= 10:
                vars['n_top10'] += 1.
            if om.om <= 5:
                vars['n_top5'] += 1.
        if om.string in outer_strings:
            vars['n_outer'] += 1.
    z_pulses = np.array(z_pulses)
    z_pulses = z_pulses[np.argsort(t_pulses)]
    if len(z_pulses) >= 4:
        len_quartile = np.floor(len(z_pulses)/4)
        mean_first_quartile = np.mean(z_pulses[:int(len_quartile)])
        vars['z_travel_top15'] = np.mean(z_pulses - mean_first_quartile)
    frame[outputname] = vars
    return True


@icetray.traysegment
def oscNext_L7_CoincidentCut(
    tray,
    name,
    uncleaned_pulses,
    cleaned_pulses,
):
    '''
    Calculate co-incident muon rejection variables and a boolean corresponding to
    optimized cut values.
    '''

    tray.AddModule(
        calculate_coincident_cut_containment, "containment_vars_cleaned",
        pulsesname=cleaned_pulses,
        outputname=L7_COINCIDENT_REJECTION_VARS_KEY,
    )

    def coincident_cut(frame) :
        passed_cut = frame[L7_COINCIDENT_REJECTION_VARS_KEY]['z_travel_top15'] >= 0.
        passed_cut *= frame[L7_COINCIDENT_REJECTION_VARS_KEY]['n_outer'] < 8
        frame[L7_COINCIDENT_REJECTION_BOOL_KEY] = icetray.I3Bool(passed_cut)

    tray.Add(coincident_cut, "L7_coincident_cut")



@icetray.traysegment
def oscNext_L7_STV( 
    tray,
    name, 
    uncleaned_pulses,
    cleaned_pulses,
    reco_particle,
    reco_shape=dataclasses.I3Particle.Primary, # Primary is the default shape of retro-reco particles
    include_DeepCore = False,

) :
    '''
    Tray segment to run the STV cuts. S
    Sub functions from STV are stored in frame_objects/stv_variables.py
    '''

    from icecube.oscNext.frame_objects.stv_variables import GetPhotonicsService, NSegmentVector, CleanTH, CleanSTV


    # Steering
    Percent = 0.01 # % of the Max expected charge, for finding time window where lay compatible hits
    NSeg = 1 #Number of segments in the track
    R = 150. #Search radius around the track 

    # Load phtonics tables
    #TODO Sometimes see a photonics spline call failure error when running this segment, needs investigating
    #TODO Which tables should we use?
    inf_muon_service = GetPhotonicsService(service_type="inf_muon")


    reshaped_reco_particle = 'Reshaped_inclDC_%s_particle_shape_%s_'%(str(include_DeepCore),reco_shape.name)+reco_particle
    # Reset the reco particle's shape (this will change the particle hypothesis)
    tray.Add(Reset_retro_reco_shape,reco_particle=reco_particle,
                                    new_shape=reco_shape,
                                    new_name = reshaped_reco_particle)

    tray.Add(check_that_shape_has_changed,particle_name=reshaped_reco_particle)


    # Needed for STV and TH
    tray.AddModule(
        NSegmentVector,
        "NSegmentVectorTH_"+reshaped_reco_particle+"_"+str(NSeg),
        FitName=reshaped_reco_particle,
        N=NSeg,
    )

    tray.Add(
        "TrackHits",
        "TH_"+reshaped_reco_particle+"_"+str(NSeg),
        Pulses=uncleaned_pulses,
        Photonics_Service=inf_muon_service,
        Percent=Percent,
        DeepCore=include_DeepCore,     # If True, include hits from DeepCore
        Fit=reshaped_reco_particle,
        Particle_Segments=reshaped_reco_particle+"_"+str(NSeg)+"_segments",
        Min_CAD_Dist=R,
    )


         
    tray.Add(
        "StartingTrackVetoLE",
        "STV_"+reshaped_reco_particle+"_"+str(NSeg),
        Pulses=uncleaned_pulses,
        Photonics_Service=inf_muon_service,
        Miss_Prob_Thresh=1.1, # A cut is systematically applied for events with a Prob < Miss_Prob_Thresh
        Fit=reshaped_reco_particle,
        Particle_Segments=reshaped_reco_particle+"_"+str(NSeg)+"_segments",
        Distance_Along_Track_Type='cherdat',
        Supress_Stochastics=False,
        Cascade = True,
        Norm = False,
        Min_CAD_Dist=R,
    )

    tray.Add(
        CleanTH,
        "CleanTH_"+reshaped_reco_particle,
        Fit=reshaped_reco_particle,
        Pulses=uncleaned_pulses,
        Name = reshaped_reco_particle,  #Output name
    )
   
    tray.Add(
        CleanSTV,
        "CleanSTV_"+reshaped_reco_particle,
        Fit=reshaped_reco_particle,
        Pulses=uncleaned_pulses,
        Name = 'L7_STV_LE_InfTrack',#L7_STV_LE_PREFIX+reshaped_reco_particle, # Name of th STV LE probability frame object (will have _Pm)
    )


@icetray.traysegment
def oscNext_L7_CorridorCuts( 
    tray,
    name, 
    uncleaned_pulses,
    cleaned_pulses,
    reco_particle,

) :
    #
    # Corridor cut (narrow)
    #

    # Compute a version of the corridor cut with much tighter settings than used at L5
    # This is more like LEESARD/DRAGON samples

    from icecube.veto_tools.CorridorCut import CorridorCut
    from icecube.oscNext.frame_objects.pulses import pulse_info
    from icecube.oscNext.selection.oscNext_L5 import AngleCorrelation

    # Run the module
    tray.AddModule(
        CorridorCut,
        "L7_NarrowCorridorCut",
        InputPulseSeries = uncleaned_pulses,
        #SANTAFit = "", #TODO?
        NoiselessPulseSeries = cleaned_pulses,
        OutputPulseSeries    = L7_NARROW_CORRIDOR_CUT_PULSES_KEY,
        HitCounter           = L7_NARROW_CORRIDOR_CUT_COUNT_KEY,
        OutputTrack          = L7_NARROW_CORRIDOR_CUT_TRACK_KEY,
        Radius               = 75. * I3Units.m, # LEESARD/DRAGON-like settings
        WindowMinus          = -150. * I3Units.ns, # LEESARD/DRAGON-like settings
        WindowPlus           = +250. * I3Units.ns, # LEESARD/DRAGON-like settings
        ZenithSteps          = 0.02,
    )
    
    # Calc hit info for the corridor pulses
    tray.Add( pulse_info, "oscNext_pulse_info_%s"%L7_NARROW_CORRIDOR_CUT_PULSES_KEY, pulses=L7_NARROW_CORRIDOR_CUT_PULSES_KEY  )

    '''
    # Calculating correlation between corridor track and reconstructed direction
    tray.AddModule(
        AngleCorrelation, 
        particleAname = L7_NARROW_CORRIDOR_CUT_TRACK_KEY, 
        particleBname = RETRO_RECO_PARTICLE_KEY, 
        outdictname   = L7_NARROW_CORRIDOR_TRACK_ANGLE_DIFF_KEY
    )
    '''
    #TODO also recompute wide angle correlation with retro reco direction??

    return



@icetray.traysegment
def oscNext_L7_photon_speed_metrics(
    tray,
    name,
    cleaned_pulses, 
) :
    '''
    Compute the superluminal photon variables for oscNext L7
    '''

    from icecube.oscNext.frame_objects.photons import photon_speed_metrics, SPEED_OF_LIGHT_ICE

    # Get the vertex
    vertex_x    = L7_FINAL_RECO_X_KEY
    vertex_y    = L7_FINAL_RECO_Y_KEY
    vertex_z    = L7_FINAL_RECO_Z_KEY
    vertex_time = L7_FINAL_RECO_TIME_KEY
    
    # Use cleaned pulses
    pulses = cleaned_pulses

    # Choose the superluminal threshold
    superluminal_threshold = SPEED_OF_LIGHT_ICE

    # Trying both all pulses (useful for noise) and only those with physical speeds (useful for track ID)
    for trim_unphysical, key in zip([False, True], ["AllPhotons", "PhysicalPhotonsOnly"]) :

        # Add module
        tray.Add(
            photon_speed_metrics,
            "photon_speed_metrics_%s"%key,
            pulses=pulses, 
            vertex_x=vertex_x,
            vertex_y=vertex_y,
            vertex_z=vertex_z,
            vertex_time=vertex_time,
            output_prefix="L7_%s"%key,
            superluminal_threshold=superluminal_threshold,
            trim_unphysical=trim_unphysical,
        )



#
# Other stuff
#

@icetray.traysegment
def oscNext_L7_compute_cut( tray, name, ):
    '''
    Calculate the cut(s) to apply at this processing level
    '''

    from icecube.oscNext.tools.classifier import I3Classifier

    #
    # Atmospheric muon classifier
    #

    # Add a muon classifier to the tray
    muon_model_file = os.path.expandvars(os.path.join(CLASSIFIER_MODEL_DIR, L7_MUON_CLASSIFIER_MODEL_FILE))
    muon_classifier = I3Classifier(model_file=muon_model_file, class_key="neutrino", output_key=L7_MUON_MODEL_PREDICTION_KEY)
    tray.Add(muon_classifier,"oscNext_L7_muon_classifier")


    #
    # Make the overall cut
    #

    # Create function defining the cut
    def overall_cut(frame) :

        # Data quality
        # Remove coincident events (not simulated)
        data_quality_cut = frame[L7_COINCIDENT_REJECTION_BOOL_KEY].value

        # Muons
        # Cut on classifier prediction (0.7 gives a muon rate that is ~ the same as the nutau rate)
        muon_cut = frame[L7_MUON_MODEL_PREDICTION_KEY].value >= 0.7

        # Noise
        noise_cut = True #TODO

        # Put it all together and add to frame
        frame[L7_DATA_QUALITY_CUT_BOOL_KEY] = icetray.I3Bool(data_quality_cut)
        frame[L7_MUON_CUT_BOOL_KEY] = icetray.I3Bool(muon_cut)
        frame[L7_NOISE_CUT_BOOL_KEY] = icetray.I3Bool(noise_cut)
        frame[L7_CUT_BOOL_KEY] = icetray.I3Bool(data_quality_cut & muon_cut & noise_cut)

    # Add to the tray
    tray.Add(overall_cut,"L7_overall_cut")

    return



@icetray.traysegment
def oscNext_L7_compute_pid( tray, name, ):
    '''
    Calculate the Particle ID
    '''

    from icecube.oscNext.tools.classifier import I3Classifier

    # Add the track vs cascade classifier to the tray
    pid_model_file = os.path.expandvars(os.path.join(CLASSIFIER_MODEL_DIR, L7_PID_CLASSIFIER_MODEL_FILE))
    pid_classifier = I3Classifier(model_file=pid_model_file, class_key="tracks", output_key=L7_PID_MODEL_PREDICTION_KEY)
    tray.Add(pid_classifier, "oscNext_L7_pid_classifier")
    
    return


#
# Top-level traysegment
#

@icetray.traysegment
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

    # Make the L3 data quality cuts
    # Update on 19th August 2019: This applies SLOP and flaring DOM cuts which have not been made before
    #TODO Can remove this in the future once have re-processed everything (e.g. because it will then have already been run at L3)
    from icecube.level3_filter_lowen.LowEnergyL3TraySegment import DataQualityCuts
    tray.AddSegment( DataQualityCuts, "L3_DataQualityCuts", pulses=cleaned_pulses, make_cut=True, If=lambda frame: "Data_quality_bool" not in frame  )


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
    tray.Add(
        oscNext_L7_CoincidentCut,
        'oscNext_L7_CoincidentCut',
        uncleaned_pulses=uncleaned_pulses,
        cleaned_pulses=cleaned_pulses,
    )

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
    tray.Add(
        oscNext_L7_STV, 
        "oscNext_L7_STV_InfTrack_DC", 
        uncleaned_pulses=uncleaned_pulses,
        cleaned_pulses=cleaned_pulses,
        reco_particle=reco_particle,
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

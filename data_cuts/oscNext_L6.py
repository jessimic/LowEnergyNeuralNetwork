'''
The oscNext level 6 event selection traysegment.
This is where final level reconstruction is done.

Tom Stuttard, Philipp Eller
'''

#TODO Make this L6 or remove enforce integer numbering of processing levels in master

import math, resource

from icecube import dataclasses
from icecube import icetray
from icecube.icetray import I3Units
from icecube import lilliput
import icecube.lilliput.segments

from icecube.oscNext.tools.data_quality import check_object_exists
from icecube.oscNext.selection.globals import *
from icecube.oscNext.selection.oscNext_cuts import oscNext_cut

from icecube.oscNext.selection.oscNext_L5 import L5_SPE_FIT_KEY
from icecube.oscNext.selection.oscNext_direct_L6 import L6_DIRECT_HDF5_KEYS, oscNext_direct_L6



#
# Globals
#

# Define retro reco fit method
RETRO_FIT_METHOD = "crs_prefit"

# Define all output frame objects
L6_HDF5_KEYS = []

# Cut
L6_HDF5_KEYS.append( L6_CUT_BOOL_KEY )

# Add the L6 direct keys
L6_HDF5_KEYS.extend(L6_DIRECT_HDF5_KEYS)

# RetroReco keys
RETRO_RECO_PREFIX = "retro_%s__" % RETRO_FIT_METHOD
# Particles...
RETRO_RECO_PARTICLE_KEY = RETRO_RECO_PREFIX + "median__neutrino"
RETRO_RECO_CASCADE_KEY = RETRO_RECO_PREFIX + "median__cascade"
RETRO_RECO_TRACK_KEY = RETRO_RECO_PREFIX + "median__track"
# Underlying varaibles #TODO do I actually need these, or are they already contained in the I3Particle instances?
RETRO_RECO_CASCADE_ENERGY_KEY = RETRO_RECO_PREFIX + "cascade_energy"
RETRO_RECO_TRACK_ENERGY_KEY = RETRO_RECO_PREFIX + "track_energy"
RETRO_RECO_ENERGY_KEY = RETRO_RECO_PREFIX + "energy"
RETRO_RECO_AZIMUTH_KEY = RETRO_RECO_PREFIX + "azimuth"
RETRO_RECO_ZENITH_KEY = RETRO_RECO_PREFIX + "zenith"
RETRO_RECO_X_KEY = RETRO_RECO_PREFIX + "x"
RETRO_RECO_Y_KEY = RETRO_RECO_PREFIX + "y"
RETRO_RECO_Z_KEY = RETRO_RECO_PREFIX + "z"
RETRO_RECO_TIME_KEY = RETRO_RECO_PREFIX + "time"
RETRO_RECO_TRACK_AZIMUTH_KEY = RETRO_RECO_PREFIX + "track_azimuth"
RETRO_RECO_TRACK_ZENITH_KEY = RETRO_RECO_PREFIX + "track_zenith"
# Fit info...
RETRO_RECO_FIT_STATUS_KEY = RETRO_RECO_PREFIX + "fit_status"
RETRO_RECO_ITERATIONS_KEY = RETRO_RECO_PREFIX + "iterations"
RETRO_RECO_LLH_STD_KEY = RETRO_RECO_PREFIX + "llh_std"
RETRO_RECO_LOWER_DLL_KEY = RETRO_RECO_PREFIX + "lower_dll"
RETRO_RECO_MAX_LLH_KEY = RETRO_RECO_PREFIX + "max_llh"
RETRO_RECO_MAX_POSTPROC_LLH_KEY = RETRO_RECO_PREFIX + "max_postproc_llh"
RETRO_RECO_NO_IMPROVEMENT_COUNTER_KEY = RETRO_RECO_PREFIX + "no_improvement_counter"
RETRO_RECO_NUM_FAILURES_KEY = RETRO_RECO_PREFIX + "num_failures"
RETRO_RECO_NUM_LLH_KEY = RETRO_RECO_PREFIX + "num_llh"
RETRO_RECO_NUM_MUTATION_SUCCESSES_KEY = RETRO_RECO_PREFIX + "num_mutation_successes"
RETRO_RECO_NUM_SIMPLEX_SUCCESSES_KEY = RETRO_RECO_PREFIX + "num_simplex_successes"
RETRO_RECO_RUN_TIME_KEY = RETRO_RECO_PREFIX + "run_time"
RETRO_RECO_STOPPING_FLAG_KEY = RETRO_RECO_PREFIX + "stopping_flag"
RETRO_RECO_UPPER_DLLH_KEY = RETRO_RECO_PREFIX + "upper_dllh"
RETRO_RECO_VERTEX_STD_KEY = RETRO_RECO_PREFIX + "vertex_std"
RETRO_RECO_VERTEX_STD_MET_AT_ITER_KEY = RETRO_RECO_PREFIX + "vertex_std_met_at_iter"
RETRO_RECO_ZERO_DLLH_KEY = RETRO_RECO_PREFIX + "zero_dllh"

L6_HDF5_KEYS.extend([
    RETRO_RECO_PARTICLE_KEY,
    RETRO_RECO_CASCADE_KEY,
    RETRO_RECO_TRACK_KEY,
    RETRO_RECO_CASCADE_ENERGY_KEY,
    RETRO_RECO_TRACK_ENERGY_KEY,
    RETRO_RECO_ENERGY_KEY,
    RETRO_RECO_AZIMUTH_KEY,
    RETRO_RECO_ZENITH_KEY,
    RETRO_RECO_X_KEY,
    RETRO_RECO_Y_KEY,
    RETRO_RECO_Z_KEY,
    RETRO_RECO_TIME_KEY,
    RETRO_RECO_TRACK_AZIMUTH_KEY,
    RETRO_RECO_TRACK_ZENITH_KEY,
    RETRO_RECO_FIT_STATUS_KEY,
    RETRO_RECO_ITERATIONS_KEY,
    RETRO_RECO_LLH_STD_KEY,
    RETRO_RECO_LOWER_DLL_KEY,
    RETRO_RECO_MAX_LLH_KEY,
    RETRO_RECO_MAX_POSTPROC_LLH_KEY,
    RETRO_RECO_NO_IMPROVEMENT_COUNTER_KEY,
    RETRO_RECO_NUM_FAILURES_KEY,
    RETRO_RECO_NUM_LLH_KEY,
    RETRO_RECO_NUM_MUTATION_SUCCESSES_KEY,
    RETRO_RECO_NUM_SIMPLEX_SUCCESSES_KEY,
    RETRO_RECO_RUN_TIME_KEY,
    RETRO_RECO_STOPPING_FLAG_KEY,
    RETRO_RECO_UPPER_DLLH_KEY,
    RETRO_RECO_VERTEX_STD_KEY,
    RETRO_RECO_VERTEX_STD_MET_AT_ITER_KEY,
    RETRO_RECO_ZERO_DLLH_KEY,
])

#
# Tray segments
#

@icetray.traysegment
def oscNext_RetroReco( tray, name, cleaned_pulses, gcd_file ):
    '''
    Run RetroReco
    '''

    #TODO Use SANTA as seed
    #TODO Steer choice of pulses, pre-cleaning, etc

    from retro import const
    from retro.reco import Reco

    # Define tables settings
    tables_path = "/cvmfs/icecube.opensciencegrid.org/data/photon-tables/retro/SpiceLea/tilt_on_anisotropy_on_noazimuth_ic80_dc60_histats"

    dom_tables_kw = {
        'norm_version':'binvol2.5', 
        'force_no_mmap': False, 
        'gcd': gcd_file,
        'dom_tables_kind': 'ckv_templ_compr',
        'compute_t_indep_exp': True,
        'num_phi_samples': None,
        'template_library': tables_path+'/ckv_dir_templates.npy',
        'use_sd_indices': const.ALL_STRS_DOMS, 
        'ckv_sigma_deg': None,
        'no_noise': False,
        'dom_tables_fname_proto': tables_path+'/cl{cluster_idx}',
    }

    tdi_tables_kw = {
        'mmap': None, 
        'tdi': None,
    }

    # Pulse charge is quantized from 2017 onwards (in detector data)
    # Need to match it
    hit_charge_quant = 0.05

    # Pulses with very low charge do not agree well between data and MC
    # See e.g. https://wiki.icecube.wisc.edu/index.php/File:Discriminator_mod.jpg
    # Enforcing a minimum pulse charges to cut away those we don't model well
    # We do not have the "discriminator scaling" mentioned in the plot applied
    # in oscNext 2018/19 MC, so choosing 0.25 rather than 0.2.
    min_hit_charge = 0.25

    # Define minimum number of pulses required to reconstruct
    # This is both because the 8D fit needs ~8 inputs (ignoring unhit DOMs as "inputs"),
    # and also to address some data-MC diagreement in these vents (probably due to bad 
    # modelling of coincident noise events)
    min_num_pulses = 8

    # Instantiate Retro reco object
    # Tables will be read into memory here, which uses a lot of memory. Report on this.
    mem_before_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    retro_reco = Reco( dom_tables_kw=dom_tables_kw, tdi_tables_kw=tdi_tables_kw, )
    retro_memory_gb = 1.e-6 * ( resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - mem_before_kb )
    print("RetroReco memory footprint (mostly tables) : %0.3g GB" % retro_memory_gb)

    # Add to the tray
    tray.Add(
        _type=retro_reco,
        _name="retro_reco",
        methods=RETRO_FIT_METHOD,
        seeding_recos=[L5_SPE_FIT_KEY, "LineFit_DC"],
        point_estimator="median",
        triggers=["I3TriggerHierarchy"],
        additional_keys=None,
        reco_pulse_series_name=cleaned_pulses,
        filter=('len(event["hits"]) >= %i'%min_num_pulses),
        hit_charge_quant=hit_charge_quant,
        min_hit_charge=min_hit_charge,
    )


def check_retro_reco_success(frame) :
    '''
    Check that retro reco ran successfully
    '''

    # Check that retro fit wass succesful
    fit_success = ( RETRO_RECO_FIT_STATUS_KEY in frame ) and frame[RETRO_RECO_FIT_STATUS_KEY] == 0

    # Remove frames where the reco did not produce a viable particle
    # This happens when the noise hypothesis is able to describe all charge
    # Cut based on the number of iterations
    if fit_success :
        if frame[RETRO_RECO_ITERATIONS_KEY] < 10 :
            fit_success = False

    return fit_success


def compute_L6_cut(frame):
    '''
    Compute the actual reco stage cut boolean
    '''

    # Check if at least one final level reco was run and was successful
    # Note that retro does some pre-cleaning of pulses/events, so some events are NOT reco'd due to this
    # santa_fit_success = False #TODO
    retro_fit_success = check_retro_reco_success(frame)

    # Build the final cut
    # keep_event = santa_fit_success or retro_fit_success #TODO
    keep_event = retro_fit_success

    # Add to frame
    frame[L6_CUT_BOOL_KEY] = icetray.I3Bool(keep_event)
    

@icetray.traysegment
def oscNext_L6( tray, name, uncleaned_pulses, cleaned_pulses, gcd_file ):
    '''
    Main traysegment for running the final oscNext recos
    '''


    #
    # Apply L5 cut
    #

    # Only keep frames passing L5
    # This is actually also run by `oscNext_direct_L6` so could ingore this,
    # but doubling up for safety in case that tray segment changes in the future
    tray.Add( oscNext_cut, "L5_cut_high_stats", processing_level=5 ) # Differentiating fro "L5_cut" also run as part of verification sample L6, which is also called by this code


    #
    # Run recos
    #

    # SANTA (via the oscNext direct L6 segment)
    tray.Add(
        oscNext_direct_L6, 
        "oscNext_direct_L6",
        uncleaned_pulses=uncleaned_pulses,
        cleaned_pulses=cleaned_pulses,
        santa_pulses='L5_SANTA_DirectPulses', #TODO import variable
    )

    #TODO L6_SANTA_sel_Particle, L6_SANTA_FitType

    # Retro
    tray.Add(
        oscNext_RetroReco, 
        "oscNext_RetroReco",
        cleaned_pulses=cleaned_pulses,
        gcd_file=gcd_file,
    )


    #
    # Done
    #

    # Add the cut
    tray.Add(
        compute_L6_cut, 
        "compute_L6_cut",
    )

    # Can dump frame for debugging
    if False :
        tray.Add("Dump","Dump")

    return

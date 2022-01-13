import numpy as np
import h5py
from icecube import icetray, dataio, dataclasses, simclasses, recclasses, sim_services
from I3Tray import I3Units
import argparse
import glob

def read_i3_files(filenames_list, variable_list,save_cnn_input=False,sim_type=None):
    output_cnn = []
    output_labels = []
    output_reco_labels = []
    output_weights = []
    output_info = []
    if save_cnn_input:
        input_features_DC = []
        input_features_IC = []

    else:
        input_features_DC = None
        input_features_IC = None

    max_files = len(filenames_list)
    if max_files > 10:
        ten_percent = int(max_files/10)

    no_retro = 0
    given_sim_type = sim_type

    for count, event_file_name in enumerate(filenames_list):
        event_file = dataio.I3File(event_file_name)
        if given_sim_type is None:
            if "muongun" in event_file_name:
                sim_type = "muongun"
            elif "noise" in event_file_name:
                sim_type = "noise"
            elif "genie" in event_file_name:
                sim_type = "genie"
            else:
                sim_type = None
            #print("No sim_type given, guessing %s is %s"%(event_file_name,sim_type))

        for frame in event_file:
            if frame.Stop == icetray.I3Frame.Physics:

                if save_cnn_input:
                    from get_observables import get_observable_features
                    DC_array, IC_near_DC_array, initial_stats, pulses_per_dom, trig_time, extra_triggers, ICstrings, clean_pulses_8_or_more = get_observable_features(frame)
                    input_features_DC.append(DC_array)
                    input_features_IC.append(IC_near_DC_array)

                #CNN Prediction
                cnn_prediction = None
                for var in variable_list:
                    predict_var = np.array([float(frame['FLERCNN_%s'%var].value)])
                    if cnn_prediction is None:
                        cnn_prediction = predict_var
                    else:
                        cnn_prediction = np.concatenate((cnn_prediction,predict_var))
                output_cnn.append(cnn_prediction)

                #Truth 
                tree = frame["I3MCTree"]
                if sim_type == "muongun":
                    nu = tree[1]
                    isCC = 2
                    isTrack = 2
                    isOther = 2
                else:
                    nu = tree[0]
                    isCC = frame['I3MCWeightDict']['InteractionType']==1.
                nu_x = nu.pos.x
                nu_y = nu.pos.y
                nu_z = nu.pos.z
                nu_zenith = nu.dir.zenith
                nu_azimuth = nu.dir.azimuth
                nu_energy = nu.energy
                nu_time = nu.time
                

                #Setting isTrack and track_length
                if ((nu.type == dataclasses.I3Particle.NuMu or nu.type == dataclasses.I3Particle.NuMuBar) and isCC):
                    isTrack = True
                    if frame["I3MCTree"][1].type == dataclasses.I3Particle.MuMinus or frame["I3MCTree"][1].type == dataclasses.I3Particle.MuPlus:
                        track_length = frame["I3MCTree"][1].length
                    else:
                        print("Second particle in MCTree not muon for numu CC? Skipping event...")
                        continue
                else:
                    isTrack = False
                    isCascade = True
                    track_length = 0
                
                if sim_type == "muongun":
                    if nu.type == dataclasses.I3Particle.MuMinus or nu.type == dataclasses.I3Particle.MuPlus:
                        track_length = frame["I3MCTree"][1].length
                        neutrino_type = 13
                        particle_type = 1
                    else:
                        print("Do not know first particle type in MCTree, should be muon for muongun sim, skipping this event")
                        continue
                else:
                    #Particle ID & neutrino/antineutrino
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

                # Find EM equivalent energy
                total_daughter_energy = 0
                em_equiv_daughter_energy = 0
                for particle in tree.get_daughters(nu.id):
                    #elif (str(particle.type) == "311" or str(particle.type) == "-311" or str(particle.type) == "O16Nucleus" or str(particle.type) == "1000080160"): #THERE IS A BETTER WAY TO FIX THIS, THIS WILL CAUSE ERRORS IN THE EM equiv SCALE calculation!!!!
                    # Do not scale neutrinos 
                    if (particle.type == dataclasses.I3Particle.NuMu or particle.type == dataclasses.I3Particle.NuMuBar \
                    or particle.type == dataclasses.I3Particle.NuE or particle.type == dataclasses.I3Particle.NuEBar\
                    or particle.type == dataclasses.I3Particle.NuTau or particle.type == dataclasses.I3Particle.NuTauBar):
                        EM_equivalent_scale = 0
                    # Do not scale mu or tau
                    elif (particle.type == dataclasses.I3Particle.MuPlus or particle.type == dataclasses.I3Particle.MuMinus \
                    or particle.type == dataclasses.I3Particle.TauPlus or particle.type == dataclasses.I3Particle.TauMinus):
                        EM_equivalent_scale = 1.0
                    else:
                        EM_equivalent_scale = sim_services.ShowerParameters(particle.type, particle.energy).emScale

                    total_daughter_energy += particle.energy
                    em_equiv_daughter_energy += particle.energy*EM_equivalent_scale


                output_labels.append( np.array([ float(nu_energy), float(nu_zenith), float(nu_azimuth), float(nu_time), float(nu_x), float(nu_y), float(nu_z), float(track_length), float(isTrack), float(neutrino_type), float(particle_type), float(isCC), float(nu_zenith), float(total_daughter_energy), float(em_equiv_daughter_energy) ]) )

                #Retro Reco
                try:
                    reco_energy = frame['L7_reconstructed_total_energy'].value
                    reco_zenith = frame['L7_reconstructed_zenith'].value
                    reco_z = frame['L7_reconstructed_vertex_z'].value
                    reco_x = frame['L7_reconstructed_vertex_x'].value
                    reco_y = frame['L7_reconstructed_vertex_y'].value
                    reco_r = frame['L7_reconstructed_vertex_rho36'].value
                    reco_time = frame['L7_reconstructed_time'].value
                    reco_azimuth = frame['L7_reconstructed_azimuth'].value
                    reco_length = frame['L7_reconstructed_track_length'].value
                    reco_casc_energy = frame['L7_reconstructed_cascade_energy'].value
                    reco_track_energy = frame['L7_reconstructed_track_energy'].value
                    reco_em_casc_energy = frame['L7_reconstructed_em_cascade_energy'].value
                except:
                    no_retro +=1
                    reco_energy = np.nan
                    reco_zenith = np.nan
                    reco_z = np.nan 
                    reco_x = np.nan 
                    reco_y = np.nan 
                    reco_r = np.nan 
                    reco_time = np.nan
                    reco_azimuth = np.nan
                    reco_length = np.nan
                    reco_casc_energy = np.nan
                    reco_track_energy = np.nan
                    reco_em_casc_energy = np.nan
                try:
                    reco_pid_full = frame['L7_PIDClassifier_FullSky_ProbTrack'].value
                    reco_pid_up = frame['L7_PIDClassifier_Upgoing_ProbTrack'].value
                except:
                    reco_pid_full = np.nan
                    reco_pid_up = np.nan
                try:
                    reco_iterations = frame['retro_crs_prefit__iterations'].value
                except:
                    reco_iterations = np.nan

                output_reco_labels.append( np.array([ float(reco_energy), float(reco_zenith), float(reco_azimuth), float(reco_time), float(reco_x), float(reco_y), float(reco_z), float(reco_length), float(reco_track_energy), float(reco_casc_energy), float(reco_em_casc_energy), float(reco_zenith), float(reco_pid_full), float(reco_pid_up), reco_iterations ]) )

                #Additional Info
                try:
                    coin_muon = frame['L7_CoincidentMuon_bool'].value > 0
                except:
                    coin_muon = np.nan
                try:
                    prob_nu = frame['L7_MuonClassifier_Upgoing_ProbNu'].value
                except:
                    prob_nu = np.nan
                try:
                    prob_nu2 = frame['L7_MuonClassifier_FullSky_ProbNu'].value
                except:
                    prob_nu2 = np.nan
                try:
                    n_top15 = frame['L7_CoincidentMuon_Variables']['n_top15']
                except:
                    n_top15 = np.nan
                try:
                    n_outer = frame['L7_CoincidentMuon_Variables']['n_outer']
                except:
                    n_outer = np.nan
                try:
                    fit_success = ( "retro_crs_prefit__fit_status" in frame ) and (frame["retro_crs_prefit__fit_status"] == 0)
                except:
                    fit_success = np.nan
                    
                #if prob_nu < 0.3:
                #    print("Prob Upgoing: %f"%prob_nu)
                #if prob_nu2 < 0.3:
                #    print("Prob All: %f"%prob_nu2)

                true_ndoms = frame['IC2018_LE_L3_Vars']['NchCleaned']
                HLC_vertex = frame['L4_first_hlc'].pos
                noise_class = frame['L4_NoiseClassifier_ProbNu'].value
                nhit_doms = frame['L5_SANTA_DirectPulsesHitMultiplicity'].n_hit_doms
                # Check for 8 or more hits
                cleaned_ice_pulses = dataclasses.I3RecoPulseSeriesMap.from_frame(frame,'SRTTWOfflinePulsesDC')
                count_cleaned_pulses = 0
                clean_pulses_8_or_more = False
                for omkey, pulselist in cleaned_ice_pulses:
                    #if clean_pulses_8_or_more == True:
                    #    break
                    for pulse in pulselist:

                        a_charge = pulse.charge

                        #Cut any pulses < 0.25 PE
                        if a_charge < 0.25:
                            continue

                        #Count number pulses > 0.25 PE in event
                        count_cleaned_pulses +=1
                        #if count_cleaned_pulses >=8:
                        #    clean_pulses_8_or_more = True
                        #    break
                try:
                    nDOMs_CNN = frame['FLERCNN_nDOM'].value
                except:
                    nDOMs_CNN = 0
                output_info.append( np.array([ float(coin_muon), float(prob_nu), float(true_ndoms), fit_success, float( noise_class), float(nhit_doms), float(n_top15), float(n_outer), float(prob_nu2), clean_pulses_8_or_more, float(HLC_vertex.x), float(HLC_vertex.y), float(HLC_vertex.z), int(nDOMs_CNN) ]))

                #Weights
                if sim_type == "data":
                    output_weights.append( np.nan )
                else:
                    weights = frame['I3MCWeightDict']
                    header = frame["I3EventHeader"]
                    if sim_type == "muongun":
                        output_weights.append( np.array([ float(header.run_id), float(header.sub_run_id), float(header.event_id), float(weights["num_events"]), float(weights["raw_weight"]), float(weights["power_law_offset"]),float(weights["power_law_index"]), float(weights["prob_passing_KDE"]), float(weights["weight"]) ]) )
                    else:
                        output_weights.append( np.array([ float(header.run_id), float(header.sub_run_id), float(header.event_id), float(weights["NEvents"]), float(weights["OneWeight"]), float(weights["GENIEWeight"]),float(weights["PowerLawIndex"]), float(weights["gen_ratio"]), float(weights["weight"]) ]) )                

        count +=1
        if (max_files > 10) and (count%ten_percent == 0):
            print("Progress Percent: %i"%(count/max_files*100))

    if output_cnn is None:
        output_cnn.append(np.nan)
        print("DID NOT SAVE ANY CNN VALUES!!")
    output_cnn=np.asarray(output_cnn,dtype=float)
    output_labels=np.asarray(output_labels)
    output_reco_labels=np.asarray(output_reco_labels)
    output_info = np.asarray(output_info)
    output_weights = np.asarray(output_weights)

    if no_retro == output_labels.shape[0]:
        print("Didnt save any retro values")

    return output_cnn, output_labels, output_reco_labels, output_info, output_weights, input_features_DC, input_features_IC

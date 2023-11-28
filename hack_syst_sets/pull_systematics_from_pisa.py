import h5py
import argparse
import os, sys
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import matplotlib.colors as colors

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input",nargs='+',
                    dest="input_file", help="path and names of the input file")
parser.add_argument("--syst_list",nargs='+',default=["0000"],
                    dest="syst_list", help="path and name of the input file")
parser.add_argument("-o", "--outdir",type=str,default='/mnt/home/micall12/LowEnergyNeuralNetwork/output_plots/',
                    dest="output_dir", help="path of ouput file")
parser.add_argument("-n", "--savename",default=None,
                    dest="savename", help="additional directory to save in")
parser.add_argument("--retro_file",default=False,action='store_true',
                    dest="retro_file",help="flag if reading retro reco (not FLERCNN) pisa file")
args = parser.parse_args()

input_file_list = args.input_file
syst_list = args.syst_list
cnn_file = not args.retro_file
save_folder_name = args.output_dir + "/"
if args.savename is not None:
    save_folder_name += args.savename + "/"
    if os.path.isdir(save_folder_name) != True:
            os.mkdir(save_folder_name)
print("Saving to %s"%save_folder_name)
print("Expecting input from cnn: %s"%cnn_file)

#CUT values
cut1 = {}
cut1['r'] = 200
cut1['zmin'] = -495
cut1['zmax'] = -225
cut1['coszen'] = 0.3
cut1['emin'] = 5
cut1['emax'] = 100
cut1['mu'] = 0.2
cut1['nDOM'] = 7
cut1['time'] = 14500
numu_files1 = 1518
nue_files1 = 602
muon_files1 = 17999
nutau_files1 = 334

nu_keys = ['nue_cc', 'nue_nc', 'nuebar_cc', 'nuebar_nc', 'numu_cc', 'numu_nc', 'numubar_cc', 'numubar_nc', 'nutau_cc', 'nutau_nc', 'nutaubar_cc', 'nutaubar_nc']
saved_particles = []
particle = {}

syst_counter = 0
f = {}
for input_file in input_file_list:
	syst_set = syst_list[syst_counter]
	syst_counter += 1
	print("Reading file %s, labeling syst %s"%(input_file,syst_set))
	f[syst_set] = h5py.File(input_file, "r")
	particle[syst_set] = {}
	if "genie" in input_file:
		for nu_key in nu_keys:
			print("Reading %s"%nu_key)
			particle[syst_set][nu_key] = f[syst_set][nu_key]
            if syst_counter == 0:
                saved_particles.append(nu_key)
    else:
        print("Only reads in genie/neutrino files!")

#Name to pull from pisa fill
cnn_vars = ['FLERCNN_energy', 'FLERCNN_coszen', 'FLERCNN_vertex_x', 'FLERCNN_vertex_y', 'FLERCNN_vertex_z']

true_vars = ['MCInIcePrimary.energy', 'MCInIcePrimary.dir.coszen', 'MCInIcePrimary.pos.x', 'MCInIcePrimary.pos.y', 'MCInIcePrimary.pos.z']

retro_vars = ['L7_reconstructed_total_energy', 'L7_reconstructed_coszen', 'L7_reconstructed_vertex_x', 'L7_reconstructed_vertex_y', 'L7_reconstructed_vertex_z'] #'L7_reconstructed_azimuth', 'L7_reconstructed_cascade_energy', 'L7_reconstructed_time', 'L7_reconstructed_track_energy', 'L7_reconstructed_track_length', 'L7_reconstructed_vertex_rho36', 'L7_reconstructed_zenith', 'L7_MuonClassifier_ProbNu', 'L7_NarrowCorridorCutPulsesHitStatistics.z_min', 'L7_PIDClassifier_ProbTrack']

if cnn_file:
    ProbNu="FLERCNN_BDT_ProbNu"
    ProbTrack="FLERCNN_prob_track"
else:
    ProbNu="L7_MuonClassifier_Upgoing_ProbNu"
    ProbTrack="L7_PIDClassifier_Upgoing_ProbTrack"

base_save = ["energy", "coszenith", "x", "y", "z"]
true = {}
reco = {}
more_info = {}
mask = {}
weights = {}
for syst_set in syst_list:
    true[syst_set] = {}
    reco[syst_set] = {}
    more_info[syst_set] = {}
    mask[syst_set] = {}
    weights[syst_set] = {}

    sum_of_events = 0
	

    for par_index in range(len(saved_particles)):
        particle_name = saved_particles[par_index]
        particle_here = particle[syst_set][particle_name]
        true_energy = np.array(particle_here['MCInIcePrimary.energy'])
        size = len(true_energy)
        sum_of_events += size
        print(syst_set, particle_name, size, sum_of_events)	

#Save main variables: energy, coszen, x, y, z
        for var in range(len(base_save)):
            #print(base_save[var], saved_particles[par_index],true_vars[var],cnn_vars[var])
			
            #Handle cases for muon and noise where true values not saved
            if particle_name == 'muon':
                if var >= 2:
                    true_var = np.full((size),None)
                else:
                    true_var = np.array(particle_here[true_vars[var]])
            elif particle_name == 'noise':
                true_var = np.full((size),None)
            else:
                true_var = np.array(particle_here[true_vars[var]])
            if cnn_file:
                reco_var = np.array(particle_here[cnn_vars[var]]) #All paricled reco-ed
            else:
                reco_var = np.array(particle_here[retro_vars[var]]) #All paricled reco-ed

        if par_index == 0:
            true[syst_set][base_save[var]] = true_var
            reco[syst_set][base_save[var]] = reco_var
        else:
            true[syst_set][base_save[var]] = np.concatenate((true[syst_set][base_save[var]], true_var))
            if base_save[var] == "energy":
                print(len(true[syst_set]["energy"]),"size")
            reco[syst_set][base_save[var]] = np.concatenate((reco[syst_set][base_save[var]], reco_var))

        #Seperate out variables by CC and NC (and muons vs. neutrinos)
        if "cc" in particle_name:
            pdg = particle_here['MCInIcePrimary.pdg_encoding']
            check_isCC = np.ones((size))
            if "numu" in particle_name: #NuMu CC
                check_isTrack = np.ones((size))
                print(particle_name,"TRACK")
            else:
                check_isTrack = np.zeros((size))
            if "nutau" in particle_name: #NuTau CC
                deposited_energy = true_energy - (true_energy*(np.ones((size))-particle_here['I3GENIEResultDict.y'])*0.5) #Estimate energy carried away by secondary nutau (~0.5), subtract that from true_energy to get deposited
            else:
                deposited_energy = true_energy
        elif "nc" in particle_name: #All NC
            pdg = particle_here['MCInIcePrimary.pdg_encoding']
            check_isCC = np.zeros((size))
            check_isTrack = np.zeros((size))
            deposited_energy = np.multiply(true_energy, particle_here['I3GENIEResultDict.y']) #Inelasticity, has fraction of true energy that remains (subtracted out the daughter lepton energy)
        elif "muon" in particle_name:
            pdg =  np.ones((size))*13
            check_isCC = np.ones((size))*2
            check_isTrack = np.ones((size))
            deposited_energy = true_energy
        elif "noise" in particle_name:
            pdg =  np.ones((size))*88
            check_isCC = np.ones((size))*2
            check_isTrack = np.zeros((size))
            deposited_energy = true_energy
        else:
            print("UNKNOWN FILE/PARTICLE")

        #Save other variables by name
        check_prob_mu = np.ones((size)) - np.array(particle_here[ProbNu])
        if par_index == 0:
            weights[syst_set] = particle_here['ReferenceWeight'] #np.ones((size))
            #print("NOT WEIGHTED")

            true[syst_set]['PID'] = pdg
            true[syst_set]['isTrack'] = check_isTrack
            true[syst_set]['isCC'] = check_isCC
            true[syst_set]['run_id'] = np.array(particle_here['I3EventHeader.run_id'])
            true[syst_set]['subrun_id'] = np.array(particle_here['I3EventHeader.sub_run_id'])
            true[syst_set]['event_id'] =  np.array(particle_here['I3EventHeader.event_id'])
            true[syst_set]['deposited_energy'] =  deposited_energy

            reco[syst_set]['prob_track'] = np.array(particle_here[ProbTrack])
            reco[syst_set]['prob_nu'] = np.array(particle_here[ProbNu])
            reco[syst_set]['prob_mu'] = check_prob_mu
            if cnn_file:
                reco[syst_set]['nDOMs'] = np.array(particle_here['FLERCNN_nDOM'])
            else:
                reco[syst_set]['time'] = np.array(particle_here['L7_reconstructed_time'])

            more_info[syst_set]['coin_muon'] = np.array(particle_here['L7_CoincidentMuon_bool'])
            more_info[syst_set]['noise_class'] = np.array(particle_here['L4_NoiseClassifier_ProbNu'])
            more_info[syst_set]['nhit_doms'] = np.array(particle_here['L5_SANTA_DirectPulsesHitMultiplicity.n_hit_doms'])
            more_info[syst_set]['n_top15'] = np.array(particle_here['L7_CoincidentMuon_Variables.n_top15'])
            more_info[syst_set]['n_outer'] = np.array(particle_here['L7_CoincidentMuon_Variables.n_outer'])
        else:
            weights[syst_set] = np.concatenate((weights[syst_set], particle_here['ReferenceWeight'])) #np.ones((size))
            #print("NOT WEIGHTED")

            true[syst_set]['isCC'] = np.concatenate((true[syst_set]['isCC'], check_isCC))
            true[syst_set]['isTrack'] = np.concatenate((true[syst_set]['isTrack'], check_isTrack))
            true[syst_set]['PID'] = np.concatenate((true[syst_set]['PID'], pdg))
            true[syst_set]['run_id'] = np.concatenate((true[syst_set]['run_id'],np.array(particle_here['I3EventHeader.run_id'])))
            true[syst_set]['subrun_id'] = np.concatenate((true[syst_set]['subrun_id'], np.array(particle_here['I3EventHeader.sub_run_id'])))
            true[syst_set]['event_id'] =  np.concatenate((true[syst_set]['event_id'], np.array(particle_here['I3EventHeader.event_id'])))
            true[syst_set]['deposited_energy'] =  np.concatenate((true[syst_set]['deposited_energy'], deposited_energy))

            reco[syst_set]['prob_track'] = np.concatenate((reco[syst_set]['prob_track'], np.array(particle_here[ProbTrack])))
            reco[syst_set]['prob_nu'] = np.concatenate((reco[syst_set]['prob_nu'], np.array(particle_here[ProbNu])))
            reco[syst_set]['prob_mu'] = np.concatenate((reco[syst_set]['prob_mu'], check_prob_mu))
            if cnn_file:
                reco[syst_set]['nDOMs'] = np.concatenate((reco[syst_set]['nDOMs'], np.array(particle_here['FLERCNN_nDOM'])))
            else:
                reco[syst_set]['time'] = np.concatenate((reco[syst_set]['time'],np.array(particle_here['L7_reconstructed_time'])))

            more_info[syst_set]['coin_muon'] = np.concatenate((more_info[syst_set]['coin_muon'], np.array(particle_here['L7_CoincidentMuon_bool'])))
            more_info[syst_set]['noise_class'] = np.concatenate((more_info[syst_set]['noise_class'], np.array(particle_here['L4_NoiseClassifier_ProbNu'])))
            more_info[syst_set]['nhit_doms'] = np.concatenate((more_info[syst_set]['nhit_doms'], np.array(particle_here['L5_SANTA_DirectPulsesHitMultiplicity.n_hit_doms'])))
            more_info[syst_set]['n_top15'] = np.concatenate((more_info[syst_set]['n_top15'], np.array(particle_here['L7_CoincidentMuon_Variables.n_top15'])))
            more_info[syst_set]['n_outer'] =  np.concatenate((more_info[syst_set]['n_outer'], np.array(particle_here['L7_CoincidentMuon_Variables.n_outer'])))


    #PID identification
    muon_mask_test = abs(true[syst_set]['PID']) == 13
    true[syst_set]['isMuon'] = np.array(muon_mask_test,dtype=bool)
    numu_mask_test = abs(true[syst_set]['PID']) == 14
    true[syst_set]['isNuMu'] = np.array(numu_mask_test,dtype=bool)
    nue_mask_test = abs(true[syst_set]['PID']) == 12
    true[syst_set]['isNuE'] = np.array(nue_mask_test,dtype=bool)
    nutau_mask_test = abs(true[syst_set]['PID']) == 16
    true[syst_set]['isNuTau'] = np.array(nutau_mask_test,dtype=bool)
    nu_mask = np.logical_or(np.logical_or(numu_mask_test, nue_mask_test), nutau_mask_test)
    true[syst_set]['isNu'] = np.array(nu_mask,dtype=bool)

    #Calculated variables
    #print(len(true[syst_set]['r']), sum([true[syst_set]['isNu']]))
    true[syst_set]['r'] = np.zeros((len(true[syst_set]['x']))) 
    x_origin = np.ones((len(true[syst_set]['x'])))*46.290000915527344
    y_origin = np.ones((len(true[syst_set]['y'])))*-34.880001068115234
    true[syst_set]['r'][true[syst_set]['isNu']] = np.sqrt( (true[syst_set]['x'][true[syst_set]['isNu']] - x_origin[true[syst_set]['isNu']])**2 + (true[syst_set]['y'][true[syst_set]['isNu']] - y_origin[true[syst_set]['isNu']])**2 ) #only calculate for neutrino events, which have non-None inputs
    reco[syst_set]['r'] = np.sqrt( (reco[syst_set]['x'] - x_origin)**2 + (reco[syst_set]['y'] - y_origin)**2 )
    #weights_squared[syst_set] = weights[syst_set]*weights[syst_set]
    together = [str(i) + str(j) + str(k) for i, j, k in zip(true[syst_set]['run_id'], true[syst_set]['subrun_id'], true[syst_set]['event_id'])]
    true[syst_set]['full_ID'] = np.array(together,dtype=int )
    true[syst_set]['isNC'] = np.logical_not(true[syst_set]['isCC'])
    true[syst_set]['isCascade'] = np.logical_not(true[syst_set]['isTrack'])

    #RECO masks
    mask[syst_set]['Energy'] = np.logical_and(reco[syst_set]['energy'] >= cut1['emin'], reco[syst_set]['energy'] <= cut1['emax'])
    mask[syst_set]['Zenith'] = reco[syst_set]['coszenith'] <= cut1['coszen']
    mask[syst_set]['R'] = reco[syst_set]['r'] < cut1['r']
    mask[syst_set]['Z'] = np.logical_and(reco[syst_set]['z'] > cut1['zmin'], reco[syst_set]['z'] < cut1['zmax'])
    mask[syst_set]['Vertex'] = np.logical_and(mask[syst_set]['R'], mask[syst_set]['Z'])
    mask[syst_set]['ProbMu'] = reco[syst_set]['prob_mu'] <= cut1['mu']
    mask[syst_set]['Reco'] = np.logical_and(mask[syst_set]['ProbMu'], np.logical_and(mask[syst_set]['Zenith'], np.logical_and(mask[syst_set]['Energy'], mask[syst_set]['Vertex'])))
    if cnn_file:
        mask[syst_set]['DOM'] = reco[syst_set]['nDOMs'] >= cut1['nDOM']
    else:
        mask[syst_set]['time'] = reco[syst_set]['time'] < cut1['time']
    mask[syst_set]['RecoNoEn'] = np.logical_and(mask[syst_set]['ProbMu'], np.logical_and(mask[syst_set]['Zenith'], mask[syst_set]['Vertex']))
    mask[syst_set]['RecoNoZenith'] = np.logical_and(mask[syst_set]['ProbMu'], np.logical_and(mask[syst_set]['Energy'], mask[syst_set]['Vertex']))
    mask[syst_set]['RecoNoZ'] = np.logical_and(mask[syst_set]['ProbMu'], np.logical_and(mask[syst_set]['Zenith'], np.logical_and(mask[syst_set]['Energy'], mask[syst_set]['R'])))
    mask[syst_set]['RecoNoR'] = np.logical_and(mask[syst_set]['ProbMu'], np.logical_and(mask[syst_set]['Zenith'], np.logical_and(mask[syst_set]['Energy'], mask[syst_set]['Z'])))
    mask[syst_set]['All'] = true[syst_set]['energy'] > 0
    true[syst_set]['All'] = true[syst_set]['energy'] > 0

    mask[syst_set]['Noise'] = mask[syst_set]['All'] #more_info[syst_set]['noise_class'] > 0.95
    mask[syst_set]['nhit'] = more_info[syst_set]['nhit_doms'] > 2.5
    mask[syst_set]['ntop']= more_info[syst_set]['n_top15'] < 0.5
    mask[syst_set]['nouter'] = more_info[syst_set]['n_outer'] < 7.5
    mask[syst_set]['CoinHits'] = np.logical_and(np.logical_and(mask[syst_set]['nhit'], mask[syst_set]['ntop']), mask[syst_set]['nouter'])
    if cnn_file:
        mask[syst_set]['MC'] = np.logical_and(np.logical_and(mask[syst_set]['CoinHits'],mask[syst_set]['Noise']),mask[syst_set]['DOM'])
    else:
        mask[syst_set]['MC'] = np.logical_and(np.logical_and(mask[syst_set]['CoinHits'],mask[syst_set]['Noise']),mask[syst_set]['time'])

    #Combined Masks
    mask[syst_set]['Analysis'] = np.logical_and(mask[syst_set]['MC'], mask[syst_set]['Reco'])
    mask[syst_set]['AnalysisNoEn'] = np.logical_and(mask[syst_set]['MC'], mask[syst_set]['RecoNoEn'])
    mask[syst_set]['AnalysisNoZen'] = np.logical_and(mask[syst_set]['MC'], mask[syst_set]['RecoNoZenith'])
    mask[syst_set]['AnalysisNoDOM'] = np.logical_and(np.logical_and(mask[syst_set]['CoinHits'],mask[syst_set]['Noise']),mask[syst_set]['Reco'])

    #print("Events file [syst_set]: %i, NuMu Rate: %.2e"%(len(true[syst_set]['energy']),sum(weights[syst_set][true[syst_set]['isNuMu']])))


from PlottingFunctions import plot_bin_slices
print(syst_list)
syst_bins = 20
save_output = {}
for i in range(0,len(syst_list)):

    print("Printing systematics set:", syst_set)
    print("Events in file: %i, Events after analysis cut: %i"%(len(true[syst_set]['energy']),sum(mask[syst_set]['Analysis'])))
    syst_set = syst_list[i]
    save_output[syst_set] = {}
    save_output[syst_set]["numucc"] = {}
    save_output[syst_set]["nuecc"] = {}
    save_output[syst_set]["numucc"]["energy"] = {}
    save_output[syst_set]["nuecc"]["energy"] = {}
    save_output[syst_set]["numucc"]["coszen"] = {}
    save_output[syst_set]["nuecc"]["coszen"] = {}
    sample="CC"
    if cnn_file:
        retro_flag = ""
    else:
        retro_flag="retro_"
    """

    sample_here = np.logical_and(true[syst_set]['isNuMu'],true[syst_set]['isCC'])
    mask_here = np.logical_and(mask[syst_set]['AnalysisNoEn'],sample_here)
    flavor="NuMu"
    print("%s %s energy"%(flavor, sample))
    save_output[syst_set]["numucc"]["energy"]["median"], save_output[syst_set]["numucc"]["energy"]["err_to"], save_output[syst_set]["numucc"]["energy"]["err_from"] = plot_bin_slices(true[syst_set]["energy"][mask_here],
                        reco[syst_set]["energy"][mask_here],
                        weights=weights[syst_set][mask_here],
                        use_fraction = True, bins=syst_bins,
                        min_val=5, max_val=100,
                        print_bins=True,
                        save=True, savefolder=save_folder_name,
                        save_name=syst_set,
                        variable="energy", units="(GeV)",
                        cnn_name="CNN",variable_type="True",
                        flavor=flavor,sample=sample,legend="upper right") 

    sample_here = np.logical_and(true[syst_set]['isNuE'],true[syst_set]['isCC'])
    mask_here = np.logical_and(mask[syst_set]['AnalysisNoEn'],sample_here)
    flavor="NuE"
    print("%s %s energy"%(flavor, sample))
    save_output[syst_set]["nuecc"]["energy"]["median"], save_output[syst_set]["nuecc"]["energy"]["err_to"], save_output[syst_set]["nuecc"]["energy"]["err_from"] = plot_bin_slices(true[syst_set]["energy"][mask_here],
                        reco[syst_set]["energy"][mask_here],
                        weights=weights[syst_set][mask_here],
                        use_fraction = True, bins=syst_bins,
                        min_val=5, max_val=100,
                        print_bins=True,
                        save=True, savefolder=save_folder_name,
                        save_name=syst_set,
                        variable="energy", units="(GeV)",
                        cnn_name="CNN",variable_type="True",
                        flavor=flavor,sample=sample,legend="upper right") 
"""
    sample_here = np.logical_and(true[syst_set]['isNuMu'],true[syst_set]['isCC'])
    mask_here = np.logical_and(mask[syst_set]['AnalysisNoZen'],sample_here)
    flavor="NuMu"
    print("%s %s cosine zenith"%(flavor, sample))
    save_output[syst_set]["numucc"]["coszen"]["median"], save_output[syst_set]["numucc"]["coszen"]["err_to"], save_output[syst_set]["numucc"]["coszen"]["err_from"] = plot_bin_slices(true[syst_set]["coszenith"][mask_here],
                        reco[syst_set]["coszenith"][mask_here],
                        weights=weights[syst_set][mask_here],
                        use_fraction = False, bins=syst_bins,
                        min_val=-1, max_val=0.3,
                        print_bins=True,
                        save=True, savefolder=save_folder_name,
                        save_name=syst_set,
                        variable="cosine zenith", units="",
                        cnn_name="CNN",variable_type="True",
                        flavor=flavor,sample=sample,legend="upper right") 
    sample_here = np.logical_and(true[syst_set]['isNuE'],true[syst_set]['isCC'])
    mask_here = np.logical_and(mask[syst_set]['AnalysisNoZen'],sample_here)
    flavor="NuE"
    print("%s %s cosine zenith"%(flavor, sample))
    save_output[syst_set]["nuecc"]["coszen"]["median"], save_output[syst_set]["nuecc"]["coszen"]["err_to"], save_output[syst_set]["nuecc"]["coszen"]["err_from"] = plot_bin_slices(true[syst_set]["coszenith"][mask_here],
                        reco[syst_set]["coszenith"][mask_here],
                        weights=weights[syst_set][mask_here],
                        use_fraction = False, bins=syst_bins,
                        min_val=-1, max_val=0.3,
                        print_bins=True,
                        save=True, savefolder=save_folder_name,
                        save_name=syst_set,
                        variable="cosine zenith", units="",
                        cnn_name="CNN",variable_type="True",
                        flavor=flavor,sample=sample,legend="upper right")
"""
    save_file_name=retro_flag + "energy_median_noEcut_%s"%(syst_set)
    with open("%s/%s.txt"%(save_folder_name,save_file_name),'w') as o:
        o.write("NuMuCC Med\t NuMuCC Q3\t NuMuCC Q1\t NuECC Med\t NuECC Q3\t NuECC Q1\n")
        for i in range(syst_bins):
            energy_numucc = save_output[syst_set]["numucc"]["energy"]
            energy_nuecc = save_output[syst_set]["nuecc"]["energy"]
            o.write("%f\t %f\t %f\t %f\t %f\t %f\n"%(energy_numucc["median"][i], energy_numucc["err_to"][i],energy_numucc["err_from"][i], energy_nuecc["median"][i], energy_nuecc["err_to"][i],energy_nuecc["err_from"][i]))
        o.close()
    """
    save_file_name=retro_flag + "coszen_median_noEcut_%s"%(syst_set)
    with open("%s/%s.txt"%(save_folder_name,save_file_name),'w') as o:
        o.write("NuMuCC Med\t NuMuCC Q3\t NuMuCC Q1\t NuECC Med\t NuECC Q3\t NuECC Q1\n")
        for i in range(syst_bins):
            coszen_numucc = save_output[syst_set]["numucc"]["coszen"]
            coszen_nuecc = save_output[syst_set]["nuecc"]["coszen"]
            o.write("%f\t %f\t %f\t %f\t %f\t %f\n"%(coszen_numucc["median"][i], coszen_numucc["err_to"][i],coszen_numucc["err_from"][i], coszen_nuecc["median"][i], coszen_nuecc["err_to"][i],coszen_nuecc["err_from"][i]))
        o.close()
   """
    #PID
    if cnn_file:	
        reco_track = reco[syst_set]['prob_track'] >= 0.55
        reco_cascade = reco[syst_set]['prob_track'] <= 0.25
        reco_mixed = np.logical_and(reco[syst_set]['prob_track'] > 0.25, reco[syst_set]['prob_track'] < 0.55)
    else:
        reco_track = reco[syst_set]['prob_track'] >= 0.85
        reco_cascade = reco[syst_set]['prob_track'] <= 0.5
        reco_mixed = np.logical_and(reco[syst_set]['prob_track'] > 0.5, reco[syst_set]['prob_track'] < 0.85)

    #Weights convert to mHz
    factor=1000
    save_file_name=retro_flag + "pid_mHz_%s"%(syst_set)
    with open("%s/%s.txt"%(save_folder_name,save_file_name),'w') as o:
        print("PID\t Track\t Mixed\t Cascade\n")
        o.write("PID\t Track\t Mixed\t Cascade\n")

        sample_here = np.logical_and(true[syst_set]['isNuE'],true[syst_set]['isCC'])
        mask_here = np.logical_and(sample_here, mask[syst_set]['Analysis'])
        track_here = np.logical_and(reco_track,mask_here)
        mixed_here = np.logical_and(reco_mixed,mask_here)
        cascade_here = np.logical_and(reco_cascade,mask_here)
        print("NuECC\t %f\t %f\t %f\n"%(sum(weights[syst_set][track_here])*factor,sum(weights[syst_set][mixed_here])*factor, sum(weights[syst_set][cascade_here])*factor))
        o.write("NuECC\t %f\t %f\t %f\n"%(sum(weights[syst_set][track_here])*factor,sum(weights[syst_set][mixed_here])*factor, sum(weights[syst_set][cascade_here])*factor))

        sample_here = np.logical_and(true[syst_set]['isNuMu'],true[syst_set]['isCC'])
        mask_here = np.logical_and(sample_here, mask[syst_set]['Analysis'])
        track_here = np.logical_and(reco_track,mask_here)
        mixed_here = np.logical_and(reco_mixed,mask_here)
        cascade_here = np.logical_and(reco_cascade,mask_here)
        print("NuMuCC\t %f\t %f\t %f\n"%(sum(weights[syst_set][track_here])*factor,sum(weights[syst_set][mixed_here])*factor, sum(weights[syst_set][cascade_here])*factor))
        o.write("NuMuCC\t %f\t %f\t %f\n"%(sum(weights[syst_set][track_here])*factor,sum(weights[syst_set][mixed_here])*factor, sum(weights[syst_set][cascade_here])*factor))

        sample_here = np.logical_and(true[syst_set]['isNuTau'],true[syst_set]['isCC'])
        mask_here = np.logical_and(sample_here, mask[syst_set]['Analysis'])
        track_here = np.logical_and(reco_track,mask_here)
        mixed_here = np.logical_and(reco_mixed,mask_here)
        cascade_here = np.logical_and(reco_cascade,mask_here)
        print("NuTauCC\t %f\t %f\t %f\n"%(sum(weights[syst_set][track_here])*factor,sum(weights[syst_set][mixed_here])*factor, sum(weights[syst_set][cascade_here])*factor))
        o.write("NuTauCC\t %f\t %f\t %f\n"%(sum(weights[syst_set][track_here])*factor,sum(weights[syst_set][mixed_here])*factor, sum(weights[syst_set][cascade_here])*factor))

        sample_here = np.logical_and(true[syst_set]['isNu'],true[syst_set]['isNC'])
        mask_here = np.logical_and(sample_here, mask[syst_set]['Analysis'])
        track_here = np.logical_and(reco_track,mask_here)
        mixed_here = np.logical_and(reco_mixed,mask_here)
        cascade_here = np.logical_and(reco_cascade,mask_here)
        print("NuNC\t %f\t %f\t %f\n"%(sum(weights[syst_set][track_here])*factor, sum(weights[syst_set][mixed_here])*factor, sum(weights[syst_set][cascade_here])*factor))
        o.write("NuNC\t %f\t %f\t %f\n"%(sum(weights[syst_set][track_here])*factor, sum(weights[syst_set][mixed_here])*factor, sum(weights[syst_set][cascade_here])*factor))

        sample_here = true[syst_set]['isNu']
        mask_here = np.logical_and(sample_here, mask[syst_set]['Analysis'])
        track_here = np.logical_and(reco_track,mask_here)
        mixed_here = np.logical_and(reco_mixed,mask_here)
        cascade_here = np.logical_and(reco_cascade,mask_here)
        print("TotalNu\t %f\t %f\t %f\n"%(sum(weights[syst_set][track_here])*factor,sum(weights[syst_set][mixed_here])*factor, sum(weights[syst_set][cascade_here])*factor))
        o.write("TotalNu\t %f\t %f\t %f\n"%(sum(weights[syst_set][track_here])*factor,sum(weights[syst_set][mixed_here])*factor, sum(weights[syst_set][cascade_here])*factor))
        o.close()

#Plot
from PlottingFunctions import plot_distributions
from PlottingFunctions import plot_2D_prediction
from PlottingFunctions import plot_single_resolution
from PlottingFunctions import plot_bin_slices
from PlottingFunctions import plot_rms_slices
from PlottingFunctionsClassification import my_confusion_matrix

save=True
if save ==True:
    print("Saving to %s"%save_folder_name)
save_base_name = save_folder_name

var_names = ["Energy", "Cosine Zenith", "Z Position", "Radius", "X End", "Y End", "Z End", "R End", "X Position", "Y Postition"]
units = ["(GeV)", "", "(m)", "(m)", "(m)", "(m)", "(m)", "(m)", "(m)", ]
minvals = [5, -1, cut1['zmin'], 0, -200, -200, -450, 0, -200, -200]
maxvals = [100, cut1['coszen'], cut1['zmax'], cut1['r'], 200, 200, -200, 300, 200, 200]
res_ranges =  [100, 1, 75, 100, 100, 100, 100, 100, 100, 100]
frac_res_ranges = [2, 2, 0.5, 1, 2, 2, 2, 2, 2, 2]
binss = [95, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
binned_fracs = [True, False, False, False, False, False, False, False, False, False, False]
syst_bins = [95, 100, 100, 100, 100, 100, 100, 100, 100, 100]
cut_mins = [cut1['emin'], None, cut1['zmin'], None, -1000, -1000, cut1['zmax'], cut1['r'], -1000, -1000]
cut_maxs = [cut1['emax'], cut1['coszen'], cut1['zmax'], cut1['r'], None, None, cut1['zmin'], None, None, None]
#Names of mask keys to Use/hold back when plotting
keynames = ['Energy', 'Zenith', 'Z', 'R', 'All', 'All', 'All', 'All', 'All', 'All', 'All']
masknames = ['RecoNoEn', 'RecoNoZenith', 'RecoNoZ', 'RecoNoR', 'Reco', 'Reco', 'Reco', 'Reco', 'Reco', 'Reco']

#check_depo = true1['energy'] - true1['deposited_energy']
#near_zero = check_depo < 1e-3
#print("Check deposited different than true 1: %i"%sum(near_zero))
#if input_file2 is not None:
#	check_depo2 = true2['energy'] - true2['deposited_energy']
#	near_zero2 = check_depo2 < 1e-3
#	print("Check deposited different than true 2: %i"%sum(near_zero2))

name1 = "CNN"
name2 = "Likelihood"
logmax = 10**1.5
bins_log = 10**np.linspace(0,1.5,100)

variable_names = ['energy', 'coszenith', 'z', 'r', 'x_end', 'y_end', 'z_end', 'r_end', 'x', 'y']
flavors = ["NuMu", "NuE", "NuTau", "Nu", "Muon", "Nu", "All", "Nu", "Nu"]
selects = ["CC", "CC", "CC", "NC", "All", "All", "All", "Track", "Cascade"]
############## CHANGE THESE LINES ##############
variable_index_list = [0] #[1,2,3] #[0] #[1,2,3] #chose variable from list above
check_index_list = [0,1] #[-2, -1] #[0,1,2,3] #corresponds to flavor/select index
cut_or = False #use for ending cuts, want below min OR above max
energy_type = "True" #"EM Equiv" or Deposited or True

print_rates = True
make_distributions = True
make_log_zoom = False
make_2d_hist = True
make_2d_hist_vs_reco = False
make_resolution = True
make_bin_slice = True
make_bin_slice_vs_reco = False
make_confusion = False
make_PID = False
make_muon = False
##################################################

all_remaining1 = mask1['Analysis']


sample_mask1 = true1['isNu']
if sum(sample_mask1) > 0:
	check1 = np.logical_and(sample_mask1, mask1['AnalysisNoDOM'])
	final1 = np.logical_and(sample_mask1, mask1['Analysis'])
	print("NU CUT NDOM", sum(weights1[final1])/sum(weights1[check1]))

if print_rates:
    print("Flavor", "Type", "Num events (after)", "Rate (after)", "Fraction Of Sample")
    for check_set in range(0,7):
        
        flavor = flavors[check_set]
        sample = selects[check_set]
        if flavor == "All":
            flavor_key = flavor
        else:
            flavor_key = "is%s"%flavor
        if sample =="CC" or sample == "NC":
            select = "is%s"%sample
        else:
            select = sample
        
        sample_mask1 = np.logical_and(true1[flavor_key],true1[select])
        final1 = np.logical_and(sample_mask1, mask1['Analysis'])
        print("%s\t %s\t %i\t %.3e\t %.3f"%(flavor, sample, sum(final1), sum(weights1[final1]), sum(weights1[final1])/sum(weights1[mask1['Analysis']])))
        
        if input_file2 is not None:
            sample_mask2 = np.logical_and(true2[flavor_key],true2[select])
            final2 = np.logical_and(sample_mask2, mask2['Analysis'])
            print("%s\t %s\t %i\t %.3e\t %.3f"%(flavor, sample, sum(final2), sum(weights2[final2]), sum(weights2[final2])/sum(weights2[mask2['Analysis']])))
            
            #Find shared events
            shared_events = len(set(true1['full_ID'][final1]) & set(true2['full_ID'][final2]))
            unique_set1 = len(true1['full_ID'][final1]) - shared_events
            unique_set2 = len(true2['full_ID'][final2]) - shared_events
            print("%s\t %s\t %i\t %i\t %i\t %.3f\t %.3f"%(flavor, sample, shared_events, unique_set1, unique_set2, unique_set1/(shared_events+unique_set1),  unique_set1/(shared_events+unique_set2)))

if make_muon:
#NEED BINARY PROBMU
    percent_save = my_confusion_matrix(true1['isNu'], mask1['ProbMu'], weights1,
                    mask=mask1['Analysis'],title="%s Muon Cut"%name1,
                    save=save,save_folder_name=save_folder_name)

    if input_file2 is not None:
        percent_save = my_confusion_matrix(true2['isNu'], mask2['ProbMu'], 
                    weights2,
                    mask=mask2['Analysis'],title="%s Muon Cut"%name2,
                    save=save,save_folder_name=save_folder_name)

if make_PID:
#NEED BINARY PROB TRACK
    mask1_here = np.logical_and(mask1['Analysis'], true1['isNu'])
    percent_save = my_confusion_matrix(true1['isTrack'], true1['prob_track'],                              weights1,label0="Cascade",label1="Track",
                    mask=mask1_here,title="%s PID Cut"%name1,
                    save=save,save_folder_name=save_folder_name)
    if input_file2 is not None:
        mask2_here = np.logical_and(mask2['Analysis'], true2['isNu'])
        percent_save = my_confusion_matrix(true2['isTrack'], true2['prob_track'],
                    weights2,label0="Cascade",label1="Track",
                    mask=mask2_here,title="%s Muon Cut"%name2,
                    save=save,save_folder_name=save_folder_name)

for variable_index in variable_index_list:
    for check_set in check_index_list:
        
        flavor = flavors[check_set]
        sample = selects[check_set]
        if flavor == "All":
            flavor_key = flavor
        else:
            flavor_key = "is%s"%flavor
        if sample =="CC" or sample == "NC" or sample == "Track" or sample == "Cascade":
            select = "is%s"%sample
        else:
            select = sample
        
        var_type="True" #labeling purposes
        if energy_type == "EM Equiv":
            use_em = 'em_equiv_'
            if variable_index == 0:
                var_type="EM Equiv"
        elif energy_type == "Deposited":
            use_em = 'deposited_'
            if variable_index == 0:
                var_type="Deposited"
        else:
            use_em = ''
        print("Plotting %s %s against %s energy"%(flavor, sample,var_type))

        save_folder_name = save_base_name + "/%s%s_%s%s/"%(use_em,variable_names[variable_index],flavor,sample)
        if os.path.isdir(save_folder_name) != True:
            os.mkdir(save_folder_name)
        print("saving to %s"%save_folder_name)

        variable_name = variable_names[variable_index]
        minval = minvals[variable_index]
        maxval = maxvals[variable_index]
        bins = binss[variable_index]
        binned_frac = binned_fracs[variable_index]
        syst_bin = syst_bins[variable_index]
        plot_name = var_names[variable_index]
        plot_units = units[variable_index]
        res_range = res_ranges[variable_index]
        frac_res_range = frac_res_ranges[variable_index]
        cut_min = cut_mins[variable_index]
        cut_max = cut_maxs[variable_index]
        keyname = keynames[variable_index]
        maskname = masknames[variable_index]

        sample_mask1 = np.logical_and(true1[flavor_key],true1[select])
        full_mask1 = np.logical_and(sample_mask1, mask1['Analysis'])
        minus_var_mask1 = np.logical_and(np.logical_and(sample_mask1, mask1['MC']), mask1[maskname])
        if input_file2 is not None:
            sample_mask2 = np.logical_and(true2[flavor_key],true2[select])
            full_mask2 = np.logical_and(sample_mask2, mask2['Analysis'])
            minus_var_mask2 = np.logical_and(np.logical_and(sample_mask2, mask2['MC']), mask2[maskname])

        print("using %s"%(use_em + variable_name))
        true1_value = true1[use_em + variable_name][minus_var_mask1]
        reco1_value = reco1[variable_name][minus_var_mask1]
        weights1_value = weights1[minus_var_mask1]
        true1_value_fullAnalysis = true1[use_em + variable_name][full_mask1]
        reco1_value_fullAnalysis = reco1[variable_name][full_mask1]
        weights1_value_fullAnalysis = weights1[full_mask1]
        true1_energy_fullAnalysis = true1[use_em + 'energy'][full_mask1]
        if cut_min is not None:
            if cut_max is not None:
                if cut_or:
                    true1_binary = np.logical_or(true1[variable_name][minus_var_mask1] > cut_min, true1[variable_name][minus_var_mask1] < cut_max)
                    reco1_binary = np.logical_or(reco1[variable_name][minus_var_mask1] > cut_min, reco1[variable_name][minus_var_mask1] < cut_max)
                    print(variable_name, "Checking > ", cut_min, " OR < ", cut_max)
                else:
                    true1_binary = np.logical_and(true1[variable_name][minus_var_mask1] > cut_min, true1[variable_name][minus_var_mask1] < cut_max)
                    reco1_binary = np.logical_and(reco1[variable_name][minus_var_mask1] > cut_min, reco1[variable_name][minus_var_mask1] < cut_max)
                    print(variable_name, "Checking > ", cut_min, " AND < ", cut_max)
            else:
                true1_binary = true1[variable_name][minus_var_mask1] > cut_min
                reco1_binary = reco1[variable_name][minus_var_mask1] > cut_min
                print(variable_name, "Checking > ", cut_min)
        else:
            true1_binary = true1[variable_name][minus_var_mask1] < cut_max
            reco1_binary = reco1[variable_name][minus_var_mask1] < cut_max
            print(variable_name, "Checking < ", cut_max)

        print(true1_binary[:10],reco1_binary[:10])
        #print(sum(weights1_value_fullAnalysis)/sum(weights1[true1['isCC']]))
        print(true1_value[:10], reco1_value[:10])

        if input_file2 is not None:
            true2_value = true2[use_em + variable_name][minus_var_mask2]
            reco2_value = reco2[variable_name][minus_var_mask2]
            true2_value_fullAnalysis = true2[use_em + variable_name][full_mask2]
            reco2_value_fullAnalysis = reco2[variable_name][full_mask2]
            weights2_value = weights2[minus_var_mask2]
            weights2_value_fullAnalysis = weights2[full_mask2]
            true2_energy_fullAnalysis = true2[use_em + 'energy'][full_mask2]
            if cut_min is not None:
                if cut_max is not None:
                    true2_binary = np.logical_and(true2[variable_name][minus_var_mask2] > cut_min, true2[variable_name][minus_var_mask2] < cut_max)
                    reco2_binary = np.logical_and(reco2[variable_name][minus_var_mask2] > cut_min, reco2[variable_name][minus_var_mask2] < cut_max)
                    print(variable_name, "Checking > ", cut_min, " AND < ", cut_max)
                else:
                    true2_binary = true2[variable_name][minus_var_mask2] > cut_min
                    reco2_binary = reco2[variable_name][minus_var_mask2] > cut_min
                    print(variable_name, "Checking > ", cut_min)
            else:
                true2_binary = true2[variable_name][minus_var_mask2] < cut_max
                reco2_binary = reco2[variable_name][minus_var_mask2] < cut_max
                print(variable_name, "Checking < ", cut_max)
            #print(sum(weights2_value_fullAnalysis)/sum(weights2[true2['isCC']]))
            print(true2_value[:10], reco2_value[:10])
        else:
            true2_value = None 
            reco2_value = None
            true2_value_fullAnalysis = None
            reco2_value_fullAnalysis = None
            weights2_value = None
            weights2_value_fullAnalysis = None
            true2_energy = None
            true2_energy_fullAnalysis = None

        if make_log_zoom:
            plt.figure(figsize=(10,7))
            plt.hist(true1_value, color="green",label="true",
                bins=bins_log,range=[minval,logmax],
                weights=weights1_value,alpha=0.5)
            plt.hist(reco1_value, color="blue",label="CNN",
                bins=bins_log,range=[minval,logmax],
                weights=weights1_value,alpha=0.5)
            plt.xscale('log')
            plt.title("Energy Distribution Weighted for %s events"%len(true1_value),fontsize=25)
            plt.xlabel("Energy (GeV)",fontsize=20)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.axvline(5,linewidth=3,linestyle="--",color='k',label="Cut at 5 GeV")
            plt.legend(loc='upper left',fontsize=15)
            plt.savefig("%s/%sLogEnergyDist_ZoomInLE.png"%(save_folder_name,name1.replace(" ","")))

            plt.figure(figsize=(10,7))
            plt.hist(true2_value, color="green",label="true",
                bins=bins_log,range=[minval,logmax],
                weights=weights2_value,alpha=0.5)
            plt.hist(reco2_value, color="blue",label="CNN",
                bins=bins_log,range=[minval,logmax],
                weights=weights2_value,alpha=0.5)
            plt.xscale('log')
            plt.title("Energy Distribution Weighted for %s events"%len(true2_value),fontsize=25)
            plt.xlabel("Energy (GeV)",fontsize=20)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.axvline(5,linewidth=3,linestyle="--",color='k',label="Cut at 5 GeV")
            plt.legend(loc='upper left',fontsize=15)
            plt.savefig("%s/%sLogEnergyDist_ZoomInLE.png"%(save_folder_name,name2.replace(" ","")))

        if make_distributions:
            plot_distributions(true1_value_fullAnalysis,
                            reco1_value_fullAnalysis,
                            weights=weights1_value_fullAnalysis,
                            save=save, savefolder=save_folder_name,
                            cnn_name = name1, variable=plot_name,
                            units= plot_units,
                            minval=minval,maxval=maxval,
                            bins=bins,true_name=energy_type)

            if input_file2 is not None:
                plot_distributions(true2_value_fullAnalysis,
                            old_reco=reco2_value_fullAnalysis,
                            weights=weights2_value_fullAnalysis,
                            save=save, savefolder=save_folder_name,
                            reco_name = name2, variable=plot_name, 
                            units= plot_units,
                            minval=minval,maxval=maxval,
                            bins=bins,true_name=energy_type)

        if make_2d_hist:
            switch = False
            plot_2D_prediction(true1_value, reco1_value,
                            weights=weights1_value,\
                            save=save, savefolder=save_folder_name,
                            bins=bins, switch_axis=switch,
                            variable=plot_name, units=plot_units, reco_name=name1,
                            flavor=flavor,sample=sample,variable_type=energy_type)


            plot_2D_prediction(true1_value, reco1_value,
                            weights=weights1_value,\
                            save=save, savefolder=save_folder_name,
                            bins=bins,switch_axis=switch,\
                            minval=minval, maxval=maxval, axis_square=True,\
                            variable=plot_name, units=plot_units, reco_name=name1,
                            flavor=flavor,sample=sample,variable_type=energy_type)

            if input_file2 is not None:
                plot_2D_prediction(true2_value, reco2_value,
                            weights=weights2_value,
                            save=save, savefolder=save_folder_name,
                            bins=bins,switch_axis=switch,\
                            variable=plot_name, units=plot_units, reco_name=name2,
                            flavor=flavor,sample=sample,variable_type=energy_type)

                plot_2D_prediction(true2_value, reco2_value,
                            weights=weights2_value,
                            save=save, savefolder=save_folder_name,
                            bins=bins,switch_axis=switch,\
                            minval=minval, maxval=maxval, axis_square=True,\
                            variable=plot_name, units=plot_units, reco_name=name2,
                            flavor=flavor,sample=sample,variable_type=energy_type)

        if make_2d_hist_vs_reco:
            switch = True
            plot_2D_prediction(true1_value, reco1_value,
                            weights=weights1_value,\
                            save=save, savefolder=save_folder_name,
                            bins=bins, switch_axis=switch,
                            variable=plot_name, units=plot_units, reco_name=name1)


            plot_2D_prediction(true1_value, reco1_value,
                            weights=weights1_value,\
                            save=save, savefolder=save_folder_name,
                            bins=bins,switch_axis=switch,\
                            minval=minval, maxval=maxval,
                            cut_truth=True, axis_square=True,\
                            variable=plot_name, units=plot_units, reco_name=name1)

            if input_file2 is not None:
                plot_2D_prediction(true2_value, reco2_value,
                            weights=weights2_value,
                            save=save, savefolder=save_folder_name,
                            bins=bins,switch_axis=switch,\
                            variable=plot_name, units=plot_units, reco_name=name2)
            
                plot_2D_prediction(true2_value, reco2_value,
                            weights=weights2_value,
                            save=save, savefolder=save_folder_name,
                            bins=bins,switch_axis=switch,\
                            minval=minval, maxval=maxval,
                            cut_truth=True, axis_square=True,\
                            variable=plot_name, units=plot_units, reco_name=name2)
        
        if make_resolution:
            #Resolution
            if input_file2 is None:
                use_old_reco = False
            else:
                use_old_reco = True
            plot_single_resolution(true1_value_fullAnalysis, reco1_value_fullAnalysis, 
                           weights=weights1_value_fullAnalysis,
                           old_reco_weights=weights2_value_fullAnalysis,
                           use_old_reco = use_old_reco,
                           old_reco = reco2_value_fullAnalysis,
                           old_reco_truth=true2_value_fullAnalysis,\
                           minaxis=-res_range, maxaxis=res_range, bins=bins,\
                           save=save, savefolder=save_folder_name,\
                           variable=plot_name, units=plot_units, reco_name=name1,
                           flavor=flavor,sample=sample)

            plot_single_resolution(true1_value_fullAnalysis, reco1_value_fullAnalysis,
                            weights=weights1_value_fullAnalysis,
                            old_reco_weights=weights2_value_fullAnalysis,\
                            use_old_reco = use_old_reco,
                            old_reco = reco2_value_fullAnalysis,
                            old_reco_truth=true2_value_fullAnalysis,\
                            minaxis=-frac_res_range, maxaxis=frac_res_range, 
                            bins=bins, use_fraction=True,\
                            save=save, savefolder=save_folder_name,\
                            variable=plot_name, units=plot_units, reco_name=name1,
                            flavor=flavor,sample=sample)

        if make_bin_slice:
            #Bin Slices
            plot_bin_slices(true1_value, reco1_value, 
                        old_reco = reco2_value,
                        old_reco_truth=true2_value,
                        weights=weights1_value,
                        old_reco_weights=weights2_value,
                        use_fraction = binned_frac, bins=syst_bin, 
                        min_val=minval, max_val=maxval,
                        #ylim=[-0.6,1.3],
                        save=save, savefolder=save_folder_name,
                        variable=plot_name, units=plot_units, 
                        cnn_name=name1, reco_name=name2,variable_type=var_type,
                        flavor=flavor,sample=sample,legend="upper right") #add_contour=True

            plot_bin_slices(true1_value_fullAnalysis, reco1_value_fullAnalysis, 
                        energy_truth=true1_energy_fullAnalysis,
                        old_reco = reco2_value_fullAnalysis,
                        old_reco_truth=true2_value_fullAnalysis,
                        reco_energy_truth = true2_energy_fullAnalysis,
                        weights=weights1_value_fullAnalysis,
                        old_reco_weights=weights2_value_fullAnalysis,\
                        use_fraction = binned_frac, bins=syst_bin,
                        min_val=minvals[0], max_val=maxvals[0],\
                        save=save, savefolder=save_folder_name,
                        variable=plot_name, units=plot_units, 
                        cnn_name=name1, reco_name=name2,
                        variable_type=energy_type,
                        xvariable="%s Energy"%energy_type,xunits="(GeV)",
                        flavor=flavor,sample=sample,legend="outside")

        if make_bin_slice_vs_reco:
            plot_bin_slices(true1_value, reco1_value, 
                        old_reco = reco2_value,old_reco_truth=true2_value,
                        weights=weights1_value, old_reco_weights=weights2_value,\
                        use_fraction = binned_frac, bins=syst_bin, 
                        min_val=minval, max_val=maxval,\
                        save=save, savefolder=save_folder_name,
                        variable=plot_name, units=plot_units, 
                        cnn_name=name1, reco_name=name2,variable_type=var_type,
                        vs_predict=True,flavor=flavor,sample=sample)

        if make_confusion:
            from PlottingFunctionsClassification import my_confusion_matrix

            percent_save = my_confusion_matrix(true1_binary, reco1_binary, weights1_value,
                            mask=None,title="%s %s Cut"%(name1,plot_name),
                            label0="Outside Cut",label1="Inside Cut",
                            save=save,save_folder_name=save_folder_name)
            print("Reco1 Positive, True Positive: %.2f"%percent_save[2])
            print("Reco1 Negative, True Negative: %.2f"%percent_save[1])
            
            if input_file2 is not None:
                percent_save2 = my_confusion_matrix(true2_binary, reco2_binary,
                            weights2_value,ylabel="Retro Prediction",
                            label0="Outside Cut",label1="Inside Cut",
                            mask=None,title="%s %s Cut"%(name2,plot_name),
                            save=save,save_folder_name=save_folder_name)
                print("Reco2 Positive, True Positive: %.2f"%percent_save2[2])
                print("Reco2 Negative, True Negative: %.2f"%percent_save2[1])
"""

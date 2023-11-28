import h5py
import argparse
import os, sys
import numpy as np
import itertools

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import matplotlib.colors as colors

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input",nargs='+',
                    dest="input_file", help="path and names of the input files")
parser.add_argument("-i2", "--input2",nargs='+',default = [],
                    dest="input_file2", help="path and name of the second set of input files (optional)")
parser.add_argument("--set_names",nargs='+',default=["cnn","retro"],
                    dest="set_names", help="names for multiple sets")
parser.add_argument("-o", "--outdir",type=str,default='/mnt/home/micall12/LowEnergyNeuralNetwork/output_plots/',
                    dest="output_dir", help="path of ouput file")
parser.add_argument("-n", "--savename",default=None,
                    dest="savename", help="additional directory to save in")
args = parser.parse_args()

input_file_list = args.input_file
input_file_list2 = args.input_file2
if len(input_file_list2) == 0:
    input_file2 = None
set_names = args.set_names
save_folder_name = args.output_dir + "/"
if args.savename is not None:
    save_folder_name += args.savename + "/"
    if os.path.isdir(save_folder_name) != True:
            os.mkdir(save_folder_name)
print("Saving to %s"%save_folder_name)


#CUT values
cut = {}
cut["CNN"] = {}
cut["RETRO"] = {}
cut["CNN"]['r'] = 200
cut["CNN"]['zmin'] = -495
cut["CNN"]['zmax'] = -225
cut["CNN"]['coszen'] = 0.3
cut["CNN"]['emin'] = 5
cut["CNN"]['emax'] = 100
cut["CNN"]['mu'] = 0.2
cut["CNN"]['cnnmu'] = 0.045
cut["CNN"]['nDOM'] = 7
cut["CNN"]['track'] = 0.55
cut["CNN"]['cascade'] = 0.25

cut["RETRO"]['r'] = 300
cut["RETRO"]['zmin'] = -500
cut["RETRO"]['zmax'] = -200
cut["RETRO"]['coszen'] = 0.3
cut["RETRO"]['emin'] = 5
cut["RETRO"]['emax'] = 300
cut["RETRO"]['mu'] = 0.2
cut["RETRO"]['time'] = 14500
cut["RETRO"]['track'] = 0.85
cut["RETRO"]['cascade'] = 0.5

pretty_flavor_name = [r'$\nu_e$' + " CC", r'$\nu_\mu$' + " CC", r'$\nu_\tau$' + " CC", r'$\nu$' + " NC",  r'$\mu_{atm}$']
color = ["red","orange","blue","green","purple","gray","magenta", "brown", "lime", "mediumslateblue", "goldenrod", "cyan", "hotpink"]
marker = ["o","s","*","X","^","p","v","3","D","P","1", "H","2"]

nu_keys = ['nue_cc', 'nue_nc', 'nuebar_cc', 'nuebar_nc', 'numu_cc', 'numu_nc', 'numubar_cc', 'numubar_nc', 'nutau_cc', 'nutau_nc', 'nutaubar_cc', 'nutaubar_nc']
saved_particles = []
particle = {}
particle[set_names[0]] = {}

# Read input files
for input_file in input_file_list:
	print("Reading file %s"%input_file)
	f = h5py.File(input_file, "r")
	if "genie" in input_file:
		for nu_key in nu_keys:
			print("Reading %s"%nu_key)
			particle[set_names[0]][nu_key] = f[nu_key]
			saved_particles.append(nu_key)
	elif "muongun" in input_file:
		print("Reading muon")
		particle[set_names[0]]['muon'] = f['muon']
		saved_particles.append('muon')
	elif "noise" in input_file:
		print("Reading noise")
		particle[set_names[0]]['noise'] = f['noise']
		saved_particles.append('noise')
	else:
		print("Could not find simulation type in name!")
	#f.close()

if len(input_file_list2) > 0:
    particle[set_names[1]] = {}
    
    for input_file2 in input_file_list2:
        print("Reading file %s"%input_file2)
        f = h5py.File(input_file2, "r")
        if "genie" in input_file2:
            for nu_key in nu_keys:
                print("Reading %s"%nu_key)
                particle[set_names[1]][nu_key] = f[nu_key]
                assert nu_key in saved_particles, "Second files don't match neutrino keys/particles saved in first file"
                #saved_particles.append(nu_key)
        elif "muongun" in input_file2:
            print("Reading muon")
            particle[set_names[1]]['muon'] = f['muon']
            #saved_particles.append('muon')
            assert "muon" in saved_particles, "Second files don't match--muons saved here but not in first file"
        elif "noise" in input_file2:
            print("Reading noise")
            particle[set_names[1]]['noise'] = f['noise']
            #saved_particles.append('noise')
            assert "noise" in saved_particles, "Second files don't match--noise saved here but not in first file"
        else:
            print("Could not find simulation type in name!")

else:
    set_names = set_names[:1]

#Name to pull from pisa fill
cnn_vars = ['FLERCNN_energy', 'FLERCNN_coszen', 'FLERCNN_vertex_x', 'FLERCNN_vertex_y', 'FLERCNN_vertex_z']

true_vars = ['MCInIcePrimary.energy', 'MCInIcePrimary.dir.coszen', 'MCInIcePrimary.pos.x', 'MCInIcePrimary.pos.y', 'MCInIcePrimary.pos.z']

retro_vars = ['L7_reconstructed_total_energy', 'L7_reconstructed_coszen', 'L7_reconstructed_vertex_x', 'L7_reconstructed_vertex_y', 'L7_reconstructed_vertex_z'] 

#Names to save under here
true = {}
reco = {}
more_info = {}
mask = {}
weights = {}
weights_squared = {}
base_save = ["energy", "coszenith", "x", "y", "z"]

print(len(saved_particles),saved_particles)

check_cnn = []
for a_set in set_names:
    true[a_set] = {}
    reco[a_set] = {}
    more_info[a_set] = {}
    mask[a_set] = {}
    weights[a_set] = {}
    weights_squared[a_set] = {}
    sum_of_events = 0
    
    if "cnn" in a_set or "CNN" in a_set:
        cnn_file = True
        reco_name="CNN"
        check_cnn.append(cnn_file)
    else:
        cnn_file = False
        reco_name="RETRO"
        check_cnn.append(cnn_file)
    print("Do I think this is a cnn file?", cnn_file)
    if cnn_file:
        ProbNu="FLERCNN_BDT_ProbNu"
        ProbTrack="FLERCNN_prob_track"
    else:
        ProbNu="L7_MuonClassifier_Upgoing_ProbNu"
        ProbTrack="L7_PIDClassifier_Upgoing_ProbTrack"

    for par_index in range(len(saved_particles)):
        particle_name = saved_particles[par_index]
        particle_here = particle[a_set][particle_name]
        if particle_name != "noise":
            true_energy = np.array(particle_here['MCInIcePrimary.energy'])
        else:
            true_energy = np.array(particle_here[cnn_vars[0]])*0
        size = len(true_energy)
        sum_of_events += size
        print(a_set, particle_name, size, sum_of_events)	

        #Save main variables: energy, coszen, x, y, z
        for var in range(len(base_save)):
            
            #Handle cases for muon and noise where true values not saved
            if particle_name == 'muon':
                if var >= 2:
                    #true_var = np.full((size),None)
                    true_var = np.zeros((size))
                else:
                    true_var = np.array(particle_here[true_vars[var]])
            elif particle_name == 'noise':
                #true_var = np.full((size),None)
                true_var = np.zeros((size))
            else:
                true_var = np.array(particle_here[true_vars[var]])
            if cnn_file:
                reco_var = np.array(particle_here[cnn_vars[var]]) #All paricled reco-ed
            else:
                reco_var = np.array(particle_here[retro_vars[var]]) #All paricled reco-ed
                

            if par_index == 0:
                true[a_set][base_save[var]] = true_var
                reco[a_set][base_save[var]] = reco_var
            else:
                true[a_set][base_save[var]] = np.concatenate((true[a_set][base_save[var]], true_var))
                reco[a_set][base_save[var]] = np.concatenate((reco[a_set][base_save[var]], reco_var))

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
            weights[a_set] = particle_here['ReferenceWeight']

            true[a_set]['PID'] = pdg
            true[a_set]['isTrack'] = check_isTrack
            true[a_set]['isCC'] = check_isCC
            true[a_set]['run_id'] = np.array(particle_here['I3EventHeader.run_id'])
            true[a_set]['subrun_id'] = np.array(particle_here['I3EventHeader.sub_run_id'])
            true[a_set]['event_id'] =  np.array(particle_here['I3EventHeader.event_id'])
            true[a_set]['deposited_energy'] =  deposited_energy

            reco[a_set]['prob_track'] = np.array(particle_here[ProbTrack])
            reco[a_set]['prob_nu'] = np.array(particle_here[ProbNu])
            reco[a_set]['prob_mu'] = check_prob_mu
            if cnn_file:
                reco[a_set]['nDOMs'] = np.array(particle_here['FLERCNN_nDOM'])
                reco[a_set]['CNN_prob_mu'] = np.array(particle_here['FLERCNN_prob_muon_v3'])
            else:
                reco[a_set]['time'] = np.array(particle_here['L7_reconstructed_time'])

            more_info[a_set]['coin_muon'] = np.array(particle_here['L7_CoincidentMuon_bool'])
            more_info[a_set]['noise_class'] = np.array(particle_here['L4_NoiseClassifier_ProbNu'])
            more_info[a_set]['nhit_doms'] = np.array(particle_here['L5_SANTA_DirectPulsesHitMultiplicity.n_hit_doms'])
            more_info[a_set]['n_top15'] = np.array(particle_here['L7_CoincidentMuon_Variables.n_top15'])
            more_info[a_set]['n_outer'] = np.array(particle_here['L7_CoincidentMuon_Variables.n_outer'])
        else:
            weights[a_set] = np.concatenate((weights[a_set], particle_here['ReferenceWeight'])) #particle_here['I3MCWeightDict.OneWeight']))

            true[a_set]['isCC'] = np.concatenate((true[a_set]['isCC'], check_isCC))
            true[a_set]['isTrack'] = np.concatenate((true[a_set]['isTrack'], check_isTrack))
            true[a_set]['PID'] = np.concatenate((true[a_set]['PID'], pdg))
            true[a_set]['run_id'] = np.concatenate((true[a_set]['run_id'],np.array(particle_here['I3EventHeader.run_id'])))
            true[a_set]['subrun_id'] = np.concatenate((true[a_set]['subrun_id'], np.array(particle_here['I3EventHeader.sub_run_id'])))
            true[a_set]['event_id'] =  np.concatenate((true[a_set]['event_id'], np.array(particle_here['I3EventHeader.event_id'])))
            true[a_set]['deposited_energy'] =  np.concatenate((true[a_set]['deposited_energy'], deposited_energy))

            reco[a_set]['prob_track'] = np.concatenate((reco[a_set]['prob_track'], np.array(particle_here[ProbTrack])))
            reco[a_set]['prob_nu'] = np.concatenate((reco[a_set]['prob_nu'], np.array(particle_here[ProbNu])))
            reco[a_set]['prob_mu'] = np.concatenate((reco[a_set]['prob_mu'], check_prob_mu))
            if cnn_file:
                reco[a_set]['nDOMs'] = np.concatenate((reco[a_set]['nDOMs'], np.array(particle_here['FLERCNN_nDOM'])))
                reco[a_set]['CNN_prob_mu'] = np.concatenate((reco[a_set]['CNN_prob_mu'], np.array(particle_here['FLERCNN_prob_muon_v3'])))
            else:
                reco[a_set]['time'] = np.concatenate((reco[a_set]['time'],np.array(particle_here['L7_reconstructed_time'])))

            more_info[a_set]['coin_muon'] = np.concatenate((more_info[a_set]['coin_muon'], np.array(particle_here['L7_CoincidentMuon_bool'])))
            more_info[a_set]['noise_class'] = np.concatenate((more_info[a_set]['noise_class'], np.array(particle_here['L4_NoiseClassifier_ProbNu'])))
            more_info[a_set]['nhit_doms'] = np.concatenate((more_info[a_set]['nhit_doms'], np.array(particle_here['L5_SANTA_DirectPulsesHitMultiplicity.n_hit_doms'])))
            more_info[a_set]['n_top15'] = np.concatenate((more_info[a_set]['n_top15'], np.array(particle_here['L7_CoincidentMuon_Variables.n_top15'])))
            more_info[a_set]['n_outer'] =  np.concatenate((more_info[a_set]['n_outer'], np.array(particle_here['L7_CoincidentMuon_Variables.n_outer'])))


    #PID identification
    muon_mask_test = abs(true[a_set]['PID']) == 13
    true[a_set]['isMuon'] = np.array(muon_mask_test,dtype=bool)
    numu_mask_test = abs(true[a_set]['PID']) == 14
    true[a_set]['isNuMu'] = np.array(numu_mask_test,dtype=bool)
    nue_mask_test = abs(true[a_set]['PID']) == 12
    true[a_set]['isNuE'] = np.array(nue_mask_test,dtype=bool)
    nutau_mask_test = abs(true[a_set]['PID']) == 16
    true[a_set]['isNuTau'] = np.array(nutau_mask_test,dtype=bool)
    nu_mask = np.logical_or(np.logical_or(numu_mask_test, nue_mask_test), nutau_mask_test)
    true[a_set]['isNu'] = np.array(nu_mask,dtype=bool)

    #Calculated variables
    true[a_set]['r'] = np.zeros((len(true[a_set]['x']))) 
    x_origin = np.ones((len(true[a_set]['x'])))*46.290000915527344
    y_origin = np.ones((len(true[a_set]['y'])))*-34.880001068115234
    true[a_set]['r'] = np.sqrt( (true[a_set]['x'] - x_origin)**2 + (true[a_set]['y'] - y_origin)**2 )
    reco[a_set]['r'] = np.sqrt( (reco[a_set]['x'] - x_origin)**2 + (reco[a_set]['y'] - y_origin)**2 )
    weights_squared[a_set] = weights[a_set]*weights[a_set]
    together = [str(i) + str(j) + str(k) for i, j, k in zip(true[a_set]['run_id'], true[a_set]['subrun_id'], true[a_set]['event_id'])]
    true[a_set]['full_ID'] = np.array(together,dtype=int )
    true[a_set]['isNC'] = np.logical_not(true[a_set]['isCC'])
    true[a_set]['isCascade'] = np.logical_not(true[a_set]['isTrack'])

    #RECO masks
    mask[a_set]['Energy'] = np.logical_and(reco[a_set]['energy'] >= cut[reco_name]['emin'], reco[a_set]['energy'] <= cut[reco_name]['emax'])
    mask[a_set]['Zenith'] = reco[a_set]['coszenith'] <= cut[reco_name]['coszen']
    mask[a_set]['R'] = reco[a_set]['r'] < cut[reco_name]['r']
    mask[a_set]['Z'] = np.logical_and(reco[a_set]['z'] > cut[reco_name]['zmin'], reco[a_set]['z'] < cut[reco_name]['zmax'])
    mask[a_set]['Vertex'] = np.logical_and(mask[a_set]['R'], mask[a_set]['Z'])
    mask[a_set]['ProbMu'] = reco[a_set]['prob_mu'] <= cut[reco_name]['mu']
    mask[a_set]['ProbTrack'] = reco[a_set]['prob_track'] >= cut[reco_name]["track"]
    mask[a_set]['ProbCascade'] = reco[a_set]['prob_track'] <= cut[reco_name]["cascade"]
    mask[a_set]['Reco'] = np.logical_and(mask[a_set]['ProbMu'], np.logical_and(mask[a_set]['Zenith'], np.logical_and(mask[a_set]['Energy'], mask[a_set]['Vertex'])))
    if cnn_file:
        mask[a_set]['DOM'] = reco[a_set]['nDOMs'] >= cut[reco_name]['nDOM']
        #mask[a_set]['CNNProbMu'] = reco[a_set]['CNN_prob_mu'] >= (cut[reco_name]['mu'])
        reco[a_set]['CNN_prob_nu'] = 1 - reco[a_set]['CNN_prob_mu']
        mask[a_set]['CNNProbNu'] = reco[a_set]['CNN_prob_nu'] <= (1-cut[reco_name]['cnnmu'])
    else:
        mask[a_set]['time'] = reco[a_set]['time'] < cut[reco_name]['time']
    mask[a_set]['RecoNoEn'] = np.logical_and(mask[a_set]['ProbMu'], np.logical_and(mask[a_set]['Zenith'], mask[a_set]['Vertex']))
    mask[a_set]['RecoNoZenith'] = np.logical_and(mask[a_set]['ProbMu'], np.logical_and(mask[a_set]['Energy'], mask[a_set]['Vertex']))
    mask[a_set]['RecoNoZ'] = np.logical_and(mask[a_set]['ProbMu'], np.logical_and(mask[a_set]['Zenith'], np.logical_and(mask[a_set]['Energy'], mask[a_set]['R'])))
    mask[a_set]['RecoNoR'] = np.logical_and(mask[a_set]['ProbMu'], np.logical_and(mask[a_set]['Zenith'], np.logical_and(mask[a_set]['Energy'], mask[a_set]['Z'])))
    mask[a_set]['RecoNoVer'] = np.logical_and(mask[a_set]['ProbMu'], np.logical_and(mask[a_set]['Zenith'], mask[a_set]['Energy']))
    mask[a_set]['RecoNoMuCut'] = np.logical_and(mask[a_set]['Energy'], np.logical_and(mask[a_set]['Zenith'], mask[a_set]['Vertex']))
    mask[a_set]['All'] = true[a_set]['energy'] > 0
    true[a_set]['All'] = true[a_set]['energy'] > 0

    mask[a_set]['Noise'] = mask[a_set]['All'] #more_info[a_set]['noise_class'] > 0.95
    mask[a_set]['nhit'] = more_info[a_set]['nhit_doms'] > 2.5
    mask[a_set]['ntop']= more_info[a_set]['n_top15'] < 0.5
    mask[a_set]['nouter'] = more_info[a_set]['n_outer'] < 7.5
    mask[a_set]['CoinHits'] = np.logical_and(np.logical_and(mask[a_set]['nhit'], mask[a_set]['ntop']), mask[a_set]['nouter'])
    if cnn_file:
        mask[a_set]['MC'] = np.logical_and(np.logical_and(mask[a_set]['CoinHits'],mask[a_set]['Noise']),mask[a_set]['DOM'])
        last_cut = "DOM"
    else:
        mask[a_set]['MC'] = np.logical_and(np.logical_and(mask[a_set]['CoinHits'],mask[a_set]['Noise']),mask[a_set]['time'])
        last_cut = "time"

#Combined Masks
    mask[a_set]['Analysis'] = np.logical_and(mask[a_set]['MC'], mask[a_set]['Reco'])
    mask[a_set]['AnalysisNoDOM'] = np.logical_and(np.logical_and(mask[a_set]['CoinHits'],mask[a_set]['Noise']),mask[a_set]['Reco'])
    mask[a_set]['AnalysisNoNhit'] = np.logical_and(np.logical_and(np.logical_and(mask[a_set]['nouter'],mask[a_set]['ntop']),mask[a_set]['Reco']),mask[a_set][last_cut])
    mask[a_set]['AnalysisNoNtop'] = np.logical_and(np.logical_and(np.logical_and(mask[a_set]['nhit'],mask[a_set]['nouter']),mask[a_set]['Reco']),mask[a_set][last_cut])
    mask[a_set]['AnalysisNoNouter'] = np.logical_and(np.logical_and(np.logical_and(mask[a_set]['nhit'],mask[a_set]['ntop']),mask[a_set]['Reco']),mask[a_set][last_cut])

    print("Events file %s: %i, Events after analysis cut: %i"%(a_set,len(true[a_set]['energy']),sum(mask[a_set]['Analysis'])))


# Plotting
variable_names = ['energy', 'coszenith', 'z', 'r', 'x', 'y']
flavors = ["NuMu", "NuE", "NuTau", "Nu", "Muon", "Nu", "All", "Nu", "Nu"]
selects = ["CC", "CC", "CC", "NC", "All", "All", "All", "Track", "Cascade"]

############## CHANGE THESE LINES ##############
variable_index_list = [0,1]  #chose variable from variable_names above
check_index_list = [0,1,-4]  #corresponds to BOTH flavors & selects list above, e.g. 0 = NuMuCC
cut_or = False #use for ending cuts, want below min OR above max
energy_type = "True" #"EM Equiv" or Deposited or True

make_test_samples = True
make_distributions = True
make_2d_hist = True
make_bin_slice = True
make_PID = True
make_muon_confusion = False
make_muon = False
make_vertex=True
##################################################

# Starting Plots
from PlottingFunctions import plot_distributions
from PlottingFunctions import plot_2D_prediction
from PlottingFunctions import plot_single_resolution
from PlottingFunctions import plot_bin_slices
from PlottingFunctions import plot_rms_slices
from PlottingFunctionsClassification import my_confusion_matrix, ROC
from PlottingFunctionsClassification import plot_classification_hist

save=True
if save ==True:
    print("Saving to %s"%save_folder_name)
save_base_name = save_folder_name
set1 = set_names[0]
if len(set_names) > 1:
    set2 = set_names[1]
else:
    set2 = None

### Plot options and formatting for different variables
var_names = ["Energy", "Cosine Zenith", "Z Position", "Radius", "X End", "Y End", "Z End", "R End", "X Position", "Y Postition"]
title_names = ["Energy Resolution", "Cosine " + r'$\theta_{zen}$' + " Resolution", "Z Position", "Radius", "X End", "Y End", "Z End", "R End", "X Position", "Y Postition"]
units = ["(GeV)", "", "(m)", "(m)", "(m)", "(m)", "(m)", "(m)", "(m)", ]
minvals = [1, -1, cut["CNN"]['zmin'], 0, -200, -200, -450, 0, -200, -200]
maxvals = [100, cut["CNN"]['coszen'], cut["CNN"]['zmax'], cut["CNN"]['r'], 200, 200, -200, 300, 200, 200]
res_ranges =  [100, 1, 75, 100, 100, 100, 100, 100, 100, 100]
frac_res_ranges = [2, 2, 0.5, 1, 2, 2, 2, 2, 2, 2]
binss = [95, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
binned_fracs = [True, False, False, False, False, False, False, False, False, False, False]
#syst_bins = [95, 100, 100, 100, 100, 100, 100, 100, 100, 100]
syst_bins = [20, 20, 20, 20, 100, 100, 100, 100, 100, 100]
cut_mins = [cut["CNN"]['emin'], None, cut["CNN"]['zmin'], None, -1000, -1000, cut["CNN"]['zmax'], cut["CNN"]['r'], -1000, -1000]
cut_maxs = [cut["CNN"]['emax'], cut["CNN"]['coszen'], cut["CNN"]['zmax'], cut["CNN"]['r'], None, None, cut["CNN"]['zmin'], None, None, None]
#Names of mask keys to Use/hold back when plotting
keynames = ['Energy', 'Zenith', 'Z', 'R', 'All', 'All', 'All', 'All', 'All', 'All', 'All']
masknames = ['RecoNoEn', 'RecoNoZenith', 'RecoNoZ', 'RecoNoR', 'Reco', 'Reco', 'Reco', 'Reco', 'Reco', 'Reco'] #Don't apply final cut/mask for current variable plotting, else the edges are pulled due to cuts


name_dict = {}
name1 = set_names[0] #"CNN"
name2 = set_names[1] #"Retro"
name_dict[set1] = name1
if set2 is not None:
    name_dict[set2] = name2
logmax = 10**1.5
bins_log = 10**np.linspace(0,1.5,100)

sample_mask1 = true[set1]['isNu'] 
final1 = np.logical_and(sample_mask1, mask[set1]['Analysis'])
a_cut="All Analysis Cuts:"
no_cut = sum(weights[set1][sample_mask1])
with_cut = sum(weights[set1][final1])
print(a_cut, sum(sample_mask1), sum(final1), with_cut, no_cut, with_cut/no_cut)

#print("%s\t %s\t %i\t %i\t %.3e\t %.3f\t %.3f"%(set1, "isNu", sum(sample_mask1), sum(final1), sum(weights[set1][sample_mask1]), sum(weights[set1][final1]), sum(weights[set1][final1])/sum(weights[set1][mask[set1][sample_mask1]])))


######### MUON CLASSIFICATION PLOTS ################

if make_muon_confusion:
    for a_set in set_names:
        muon_key = "prob_nu" #BDT
        muon_mask_key = "ProbMu"

        print(a_set,muon_key,muon_mask_key,reco[a_set][muon_key][:10],mask[a_set][muon_mask_key][:10])
        print(a_set,"prob_nu","ProbMu",reco[a_set]["prob_nu"][:10],mask[a_set]["ProbMu"][:10])
        percent_save = my_confusion_matrix(true[a_set]['isNu'],
                    mask[a_set][muon_mask_key], weights[a_set],
                    mask=np.logical_and(mask[a_set]["RecoNoMuCut"], mask[a_set]["MC"]),
                    title="%s Muon Cut"%name_dict[a_set],
                    save=save,save_folder_name=save_folder_name)

if make_muon:
    for a_set in set_names:
        
        plot_classification_hist(true[a_set]['isNu'],
                        (reco[a_set]["prob_nu"]),
                        mask=np.logical_and(mask[a_set]["RecoNoMuCut"], mask[a_set]["MC"]),
                        mask_name="No Muon Cut", units="",bins=50,
                        weights=weights[a_set], log=True,save=save,
                        save_folder_name=save_folder_name,
                        name_prob1 = "Neutrino", name_prob0 = "Muon") 
        mask_here = np.logical_and(mask[a_set]["Analysis"], true[a_set]['isMuon'])
        if a_set == "CNN" or a_set == "cnn":
            plot_distributions(true[a_set]['energy'][mask_here],
                            reco[a_set]['energy'][mask_here],
                            weights=weights[a_set][mask_here],
                            save=save, savefolder=save_folder_name,
                            cnn_name = name_dict[a_set], variable="Muon Energy",
                            units= "(GeV)",
                            minval=5,maxval=100,
                            bins=95,true_name="")
            plot_classification_hist(true[set1]['isNu'],
                        (reco[set1]["CNN_prob_nu"]),
                        mask=np.logical_and(mask[set1]["RecoNoMuCut"], mask[set1]["MC"]),
                        mask_name="No Muon Cut", units="",bins=50,
                        weights=weights[set1], log=True,save=save,
                        save_folder_name=save_folder_name,savename="CNN_Muon_hist",
                        name_prob1 = "Neutrino", name_prob0 = "Muon") 
            plot_classification_hist(true[set1]['isNu'],
                        (reco[set1]["prob_nu"]),
                        mask=np.logical_and(mask[set1]["RecoNoMuCut"], mask[set1]["MC"]),
                        mask_name="No Muon Cut", units="",bins=50,
                        weights=weights[set1], log=True,save=save,
                        save_folder_name=save_folder_name,savename="BDT_Muon_hist",
                        name_prob1 = "Neutrino", name_prob0 = "Muon") 
            ROC(true[set1]['isNu'],reco[set1]["CNN_prob_nu"],
                        mask=np.logical_and(mask[set1]["RecoNoMuCut"], mask[set1]["MC"]),
                        mask_name="No Muon Cut",
                        reco=reco[set2]['prob_nu'],reco_truth=true[set2]['isNu'],
                        reco_mask=np.logical_and(mask[set2]["RecoNoMuCut"], mask[set2]["MC"]),
                        reco_name=name2,
                        save=save,save_folder_name=save_folder_name,
                        variable="Probability Neutrino (BDT)") 
            ROC(true[set1]['isNu'],reco[set1]["CNN_prob_nu"],
                        mask=np.logical_and(mask[set1]["RecoNoMuCut"], mask[set1]["MC"]),
                        mask_name="No Muon Cut",
                        reco=reco[set2]['prob_nu'],reco_truth=true[set2]['isNu'],
                        reco_mask=np.logical_and(mask[set2]["RecoNoMuCut"], mask[set2]["MC"]),
                        reco_name=name2,
                        save=save,save_folder_name=save_folder_name,
                        variable="Probability Neutrino (CNN)") 

    if len(set_names) > 1:
        print(a_set,reco[set1]["prob_nu"][:10])
        ROC(true[set1]['isNu'],reco[set1]["prob_nu"],
            mask=np.logical_and(mask[set1]["RecoNoMuCut"], mask[set1]["MC"]),
            reco=reco[set2]['prob_nu'],reco_truth=true[set2]['isNu'],
            reco_mask=np.logical_and(mask[set2]["RecoNoMuCut"], mask[set2]["MC"]),
            reco_name=name2,
            mask_name="No Muon Cut",
            save=save,save_folder_name=save_folder_name,
            variable="Probability Neutrino (BDT)") 
    else:
        ROC(true[set1]['isNu'],reco[set1]["prob_nu"],
            mask=np.logical_and(mask[set1]["RecoNoMuCut"], mask[set1]["MC"]),
            mask_name="No Muon Cut",
            save=save,save_folder_name=save_folder_name,
            variable="Probability Neutrino (BDT)") 

######### PID CLASSIFICATION PLOTS ################
if make_PID:
    for a_set in set_names:
        if "cnn" in a_set or "CNN" in a_set:
            print("I think", a_set, "is CNN")
            reco_name = "CNN"
        else:
            reco_name = "RETRO"
        mask_here = np.logical_and(mask[a_set]['Analysis'], true[a_set]['isNu'])
        plot_classification_hist(true[a_set]['isTrack'],
                        reco[a_set]['prob_track'],
                        mask=mask_here,
                        mask_name="Analysis Neutrinos", units="",bins=50,
                        weights=weights[a_set], log=True,save=save,
                        save_folder_name=save_folder_name,
                        name_prob1 = "Track", name_prob0 = "Cascade")
 
    if len(set_names) > 1:
        ROC(true[set1]['isTrack'],reco[set1]['prob_track'],
            mask=np.logical_and(mask[set1]['Analysis'], true[set1]['isNu']),
            reco=reco[set2]['prob_track'],
            reco_truth=true[set2]['isTrack'],
            reco_mask=np.logical_and(mask[set2]['Analysis'], true[set2]['isNu']),
            reco_name=name2,
            mask_name="Analysis Neutrinos",
            save=save,save_folder_name=save_folder_name,
            variable="Probability Track") 
    else:
        ROC(true[set1]['isTrack'],reco[set1]['prob_track'],
            mask=np.logical_and(mask[set1]['Analysis'], true[set1]['isNu']),
            mask_name="Analysis Neutrinos",
            save=save,save_folder_name=save_folder_name,
            variable="Probability Track") 

######### VERTEX CLASSIFICATION PLOTS ################
if make_vertex:
    # COMPARE EXACT CUTS, FOR SET 1 ONLY
    for a_set in set_names:
        true[a_set]['cut_R'] = true[a_set]['r'] < cut["CNN"]['r']
        true[a_set]['cut_Z'] = np.logical_and(true[a_set]['z'] > cut["CNN"]['zmin'], true[a_set]['z'] < cut["CNN"]['zmax'])
        true[a_set]['Vertex'] = np.logical_and(true[a_set]['cut_R'], true[a_set]['cut_Z'])
        reco_binary = mask[a_set]['Vertex']
        true_binary = true[a_set]['Vertex']
        mask_here = np.logical_and(mask[a_set]['RecoNoVer'], mask[a_set]['MC'])
        percent_save,save_percent_error = my_confusion_matrix(true_binary, 
                            reco_binary, weights[a_set],mask=mask_here,
                            title="%s %s Cut"%(name_dict[a_set],"Vertex"),
                            label0="Outside Cut",label1="Inside Cut",
                            save=save,save_folder_name=save_folder_name)
        print("Reco %s Positive, True Positive: %.2f"%(a_set,percent_save[2]))
        print("Reco %s Negative, True Negative: %.2f"%(a_set,percent_save[1]))

######## END CLASS PLOTS #############

#### CREATE TRUE ENERGY AND ZENITH DISTRIBUTION FOR TEST SAMPLE #########
if make_test_samples:
    plt.figure(figsize=(10,7))
    plt.title("Testing Energy Distributions",fontsize=25)
    minval_here = 1 
    maxval_here = 10000 
    bins = 10**(np.arange(0,4,0.1))
    selection_numucc = np.logical_and(true[set1]['isNuMu'], true[set1]['isCC'])
    selection_nuecc = np.logical_and(true[set1]['isNuE'], true[set1]['isCC'])
    plt.hist([true[set1]['energy'][selection_numucc],true[set1]['energy'][selection_nuecc]], bins=bins,color=[color[2],color[1]],range=[minval_here,maxval_here],weights=[weights[set1][selection_numucc]*1000,weights[set1][selection_nuecc]*1000],label=[pretty_flavor_name[1],pretty_flavor_name[0]],stacked=True);
    plt.ylabel("Rate (mHz)")
    plt.xlabel("True Neutrino Energy (GeV)",fontsize=20)
    #plt.yscale("log")
    plt.xscale("log")
    plt.legend(fontsize=20)
    plt.savefig("%s/TestingNuMuNuE_EnergyDistribution_%ito%i_xlog.png"%(save_folder_name,int(minval_here),int(maxval_here)),bbox_inches='tight')
    
    plt.figure(figsize=(10,7))
    plt.title("Testing Zenith Distributions",fontsize=25)
    minval_here = -1
    maxval_here = 1
    bins = 50
    selection_numucc = np.logical_and(true[set1]['isNuMu'], true[set1]['isCC'])
    selection_nuecc = np.logical_and(true[set1]['isNuE'], true[set1]['isCC'])
    plt.hist([true[set1]['coszenith'][selection_numucc],true[set1]['coszenith'][selection_nuecc]], bins=bins,color=[color[2],color[1]],range=[minval_here,maxval_here],weights=[weights[set1][selection_numucc]*1000,weights[set1][selection_nuecc]*1000],label=[pretty_flavor_name[1],pretty_flavor_name[0]],stacked=True);
    plt.ylabel("Rate (mHz)")
    plt.xlabel("True Cosine "+ r'$\theta_{zen}$',fontsize=20)
    plt.legend(fontsize=20)
    plt.savefig("%s/TestingNuMuNuE_CosZenDistribution_%ito%i.png"%(save_folder_name,int(minval_here),int(maxval_here)),bbox_inches='tight')


######## LOOP THRO VAR FOR REGRESSION PLOTS ############
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
        title_name = title_names[variable_index]

        sample_mask1 = np.logical_and(true[set1][flavor_key],true[set1][select])
        full_mask1 = np.logical_and(sample_mask1, mask[set1]['Analysis'])
        minus_var_mask1 = np.logical_and(np.logical_and(sample_mask1, mask[set1]['MC']), mask[set1][maskname]) #All analysis cuts EXCEPT for variable being plotting (e.g. no energy cut if plotting energy, but keeping all other cuts)

        if input_file2 is not None:
            sample_mask2 = np.logical_and(true[set2][flavor_key],true[set2][select])
            full_mask2 = np.logical_and(sample_mask2, mask[set2]['Analysis'])
            minus_var_mask2 = np.logical_and(np.logical_and(sample_mask2, mask[set2]['MC']), mask[set2][maskname])

        print("using %s"%(use_em + variable_name))
        true1_value = true[set1][use_em + variable_name][minus_var_mask1]
        reco1_value = reco[set1][variable_name][minus_var_mask1]
        weights1_value = weights[set1][minus_var_mask1]
        true1_value_fullAnalysis = true[set1][use_em + variable_name][full_mask1]
        reco1_value_fullAnalysis = reco[set1][variable_name][full_mask1]
        weights1_value_fullAnalysis = weights[set1][full_mask1]
        true1_energy_fullAnalysis = true[set1][use_em + 'energy'][full_mask1]

        print(true1_value[:10], reco1_value[:10])

        if input_file2 is not None:
            true2_value = true[set2][use_em + variable_name][minus_var_mask2]
            reco2_value = reco[set2][variable_name][minus_var_mask2]
            true2_value_fullAnalysis = true[set2][use_em + variable_name][full_mask2]
            reco2_value_fullAnalysis = reco[set2][variable_name][full_mask2]
            weights2_value = weights[set2][minus_var_mask2]
            weights2_value_fullAnalysis = weights[set2][full_mask2]
            true2_energy_fullAnalysis = true[set2][use_em + 'energy'][full_mask2]
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


        ######## STARTING PLOTS ##############

        if make_distributions:
            plot_distributions(truth=true1_value_fullAnalysis,
                            reco=reco1_value_fullAnalysis,
                            weights=weights1_value_fullAnalysis,
                            save=save, savefolder=save_folder_name,
                            cnn_name = name1,
                            xlog=binned_frac,
                            variable=plot_name, units= plot_units, 
                            minval=minval,maxval=maxval,
                            bins=bins,true_name=energy_type)

            if input_file2 is not None:
                plot_distributions(truth=true2_value_fullAnalysis,
                            old_reco=reco2_value_fullAnalysis,
                            weights=weights2_value_fullAnalysis,
                            save=save, savefolder=save_folder_name,
                            reco_name = name2,
                            xlog=binned_frac,
                            variable=plot_name, units= plot_units,
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

        if make_bin_slice:
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
                        flavor=flavor,sample=sample,legend="upper right",
                        title=title_name) #add_contour=True

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
                        flavor=flavor,sample=sample,legend="outside",
                        title=title_name)

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


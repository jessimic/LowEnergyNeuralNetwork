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
parser.add_argument("--input2",type=str,default=None,
                    dest="input_file2", help="path and name of the input file")
parser.add_argument("-o", "--outdir",type=str,default='/mnt/home/micall12/LowEnergyNeuralNetwork/output_plots/',
                    dest="output_dir", help="path of ouput file")
parser.add_argument("-n", "--savename",default=None,
                    dest="savename", help="additional directory to save in")
args = parser.parse_args()

input_file_list = args.input_file
input_file2 = args.input_file2
save_folder_name = args.output_dir + "/"
if args.savename is not None:
    save_folder_name += args.savename + "/"
    if os.path.isdir(save_folder_name) != True:
            os.mkdir(save_folder_name)
print("Saving to %s"%save_folder_name)

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
numu_files1 = 1518
nue_files1 = 602
muon_files1 = 17999
nutau_files1 = 334

cut2 = {}
cut2['r'] = 300
cut2['zmin'] = -500
cut2['zmax'] = -200
cut2['coszen'] = 0.3
cut2['emin'] = 5
cut2['emax'] = 300
cut2['mu'] = 0.01
cut2['nDOM'] = 7
numu_files2 = 1518
nue_files2 = 602
muon_files2 = 19991
nutau_files2 = 334

nu_keys = ['nue_cc', 'nue_nc', 'nuebar_cc', 'nuebar_nc', 'numu_cc', 'numu_nc', 'numubar_cc', 'numubar_nc', 'nutau_cc', 'nutau_nc', 'nutaubar_cc', 'nutaubar_nc']
saved_particles = []
particle = {}

for input_file in input_file_list:
	print("Reading file %s"%input_file)
	f = h5py.File(input_file, "r")
	if "genie" in input_file:
		for nu_key in nu_keys:
			print("Reading %s"%nu_key)
			particle[nu_key] = f[nu_key]
			saved_particles.append(nu_key)
	elif "muongun" in input_file:
		print("Reading muon")
		particle['muon'] = f['muon']
		saved_particles.append('muon')
	elif "noise" in input_file:
		print("Reading noise")
		particle['noise'] = f['noise']
		saved_particles.append('noise')
	else:
		print("Could not find simulation type in name!")
	#f.close()
	#del f

#Name to pull from pisa fill
cnn_vars = ['FLERCNN_energy', 'FLERCNN_coszen', 'FLERCNN_vertex_x', 'FLERCNN_vertex_y', 'FLERCNN_vertex_z']

true_vars = ['MCInIcePrimary.energy', 'MCInIcePrimary.dir.coszen', 'MCInIcePrimary.pos.x', 'MCInIcePrimary.pos.y', 'MCInIcePrimary.pos.z']

retro_vars = ['L7_reconstructed_total_energy', 'L7_reconstructed_zenith', 'L7_reconstructed_vertex_x', 'L7_reconstructed_vertex_y', 'L7_reconstructed_vertex_z'] #'L7_reconstructed_azimuth', 'L7_reconstructed_cascade_energy', 'L7_reconstructed_time', 'L7_reconstructed_track_energy', 'L7_reconstructed_track_length', 'L7_reconstructed_vertex_rho36', 'L7_reconstructed_zenith', 'L7_MuonClassifier_ProbNu', 'L7_NarrowCorridorCutPulsesHitStatistics.z_min', 'L7_PIDClassifier_ProbTrack']

#Names to save under here
true1 = {}
reco1 = {}
more_info1 = {}

base_save = ["energy", "coszenith", "x", "y", "z"]
sum_of_events = 0

for par_index in range(len(saved_particles)):
	particle_name = saved_particles[par_index]
	true_energy = np.array(particle[particle_name]['MCInIcePrimary.energy'])
	size = len(true_energy)
	sum_of_events += size
	print(particle_name, size, sum_of_events)	

	#Save main variables: energy, coszen, x, y, z
	for var in range(len(base_save)):
		#print(base_save[var], saved_particles[par_index],true_vars[var],cnn_vars[var])
		
		#Handle cases for muon and noise where true values not saved
		if particle_name == 'muon':
			if var >= 2:
				#true_var = np.full((size),None)
				true_var = np.zeros((size))
			else:
				true_var = np.array(particle[particle_name][true_vars[var]])
		elif particle_name == 'noise':
			#true_var = np.full((size),None)
			true_var = np.zeros((size))
		else:
			true_var = np.array(particle[particle_name][true_vars[var]])
		reco_var = np.array(particle[particle_name][cnn_vars[var]]) #All paricled reco-ed
			

		if par_index == 0:
			true1[base_save[var]] = true_var
			reco1[base_save[var]] = reco_var
		else:
			true1[base_save[var]] = np.concatenate((true1[base_save[var]], true_var))
			reco1[base_save[var]] = np.concatenate((reco1[base_save[var]], reco_var))

	#Seperate out variables by CC and NC (and muons vs. neutrinos)
	if "cc" in particle_name:
		pdg = particle[particle_name]['MCInIcePrimary.pdg_encoding']
		#true1['r'] = np.sqrt( (true1['x'] - x1_origin)**2 + (true1['y'] - y1_origin)**2 )
		check_isCC = np.ones((size))
		if "numu" in particle_name: #NuMu CC
			check_isTrack = np.ones((size))
			print(particle_name,"TRACK")
		else:
			check_isTrack = np.zeros((size))
		if "nutau" in particle_name: #NuTau CC
			deposited_energy = true_energy - (true_energy*(np.ones((size))-particle[particle_name]['I3GENIEResultDict.y'])*0.5) #Estimate energy carried away by secondary nutau (~0.5), subtract that from true_energy to get deposited
		else:
			deposited_energy = true_energy
	elif "nc" in particle_name: #All NC
		pdg = particle[particle_name]['MCInIcePrimary.pdg_encoding']
		#calc_r = np.sqrt( (true1['x'] - x1_origin)**2 + (true1['y'] - y1_origin)**2 )
		check_isCC = np.zeros((size))
		check_isTrack = np.zeros((size))
		deposited_energy = np.multiply(true_energy, particle[particle_name]['I3GENIEResultDict.y']) #Inelasticity, has fraction of true energy that remains (subtracted out the daughter lepton energy)
	elif "muon" in particle_name:
		pdg =  np.ones((size))*13
		#calc_r = np.full((size)),None) #None saved for muon and noise
		check_isCC = np.ones((size))*2
		check_isTrack = np.ones((size))
		deposited_energy = true_energy
	elif "noise" in particle_name:
		pdg =  np.ones((size))*88
		#calc_r = np.full((size)),None) #None saved for muon and noise
		check_isCC = np.ones((size))*2
		check_isTrack = np.zeros((size))
		deposited_energy = true_energy
	else:
		print("UNKNOWN FILE/PARTICLE")

	#Save other variables by name
	check_prob_mu = np.ones((size)) - np.array(particle[particle_name]['FLERCNN_BDT_ProbNu'])
	if par_index == 0:
		weights1 = np.ones((size)) #particle[particle_name]['I3MCWeightDict.OneWeight']

		true1['PID'] = pdg
		true1['isTrack'] = check_isTrack
		true1['isCC'] = check_isCC
		true1['run_id'] = np.array(particle[particle_name]['I3EventHeader.run_id'])
		true1['subrun_id'] = np.array(particle[particle_name]['I3EventHeader.sub_run_id'])
		true1['event_id'] =  np.array(particle[particle_name]['I3EventHeader.event_id'])
		true1['deposited_energy'] =  deposited_energy

		reco1['prob_track'] = np.array(particle[particle_name]['FLERCNN_prob_track'])
		reco1['prob_nu'] = np.array(particle[particle_name]['FLERCNN_BDT_ProbNu'])
		reco1['prob_mu'] = check_prob_mu
		reco1['nDOMs'] = np.array(particle[particle_name]['FLERCNN_nDOM'])

		more_info1['coin_muon'] = np.array(particle[particle_name]['L7_CoincidentMuon_bool'])
		more_info1['noise_class'] = np.array(particle[particle_name]['L4_NoiseClassifier_ProbNu'])
		more_info1['nhit_doms'] = np.array(particle[particle_name]['L5_SANTA_DirectPulsesHitMultiplicity.n_hit_doms'])
		more_info1['n_top15'] = np.array(particle[particle_name]['L7_CoincidentMuon_Variables.n_top15'])
		more_info1['n_outer'] = np.array(particle[particle_name]['L7_CoincidentMuon_Variables.n_outer'])
	else:
		weights1 = np.concatenate((weights1, np.ones((size))))#particle[particle_name]['I3MCWeightDict.OneWeight']))

		true1['isCC'] = np.concatenate((true1['isCC'], check_isCC))
		true1['isTrack'] = np.concatenate((true1['isTrack'], check_isTrack))
		true1['PID'] = np.concatenate((true1['PID'], pdg))
		true1['run_id'] = np.concatenate((true1['run_id'],np.array(particle[particle_name]['I3EventHeader.run_id'])))
		true1['subrun_id'] = np.concatenate((true1['subrun_id'], np.array(particle[particle_name]['I3EventHeader.sub_run_id'])))
		true1['event_id'] =  np.concatenate((true1['event_id'], np.array(particle[particle_name]['I3EventHeader.event_id'])))
		true1['deposited_energy'] =  np.concatenate((true1['deposited_energy'], deposited_energy))

		reco1['prob_track'] = np.concatenate((reco1['prob_track'], np.array(particle[particle_name]['FLERCNN_prob_track'])))
		reco1['prob_nu'] = np.concatenate((reco1['prob_nu'], np.array(particle[particle_name]['FLERCNN_BDT_ProbNu'])))
		reco1['prob_mu'] = np.concatenate((reco1['prob_mu'], check_prob_mu))
		reco1['nDOMs'] = np.concatenate((reco1['nDOMs'], np.array(particle[particle_name]['FLERCNN_nDOM'])))

		more_info1['coin_muon'] = np.concatenate((more_info1['coin_muon'], np.array(particle[particle_name]['L7_CoincidentMuon_bool'])))
		more_info1['noise_class'] = np.concatenate((more_info1['noise_class'], np.array(particle[particle_name]['L4_NoiseClassifier_ProbNu'])))
		more_info1['nhit_doms'] = np.concatenate((more_info1['nhit_doms'], np.array(particle[particle_name]['L5_SANTA_DirectPulsesHitMultiplicity.n_hit_doms'])))
		more_info1['n_top15'] = np.concatenate((more_info1['n_top15'], np.array(particle[particle_name]['L7_CoincidentMuon_Variables.n_top15'])))
		more_info1['n_outer'] =  np.concatenate((more_info1['n_outer'], np.array(particle[particle_name]['L7_CoincidentMuon_Variables.n_outer'])))


#PID identification
muon_mask_test1 = abs(true1['PID']) == 13
true1['isMuon'] = np.array(muon_mask_test1,dtype=bool)
numu_mask_test1 = abs(true1['PID']) == 14
true1['isNuMu'] = np.array(numu_mask_test1,dtype=bool)
nue_mask_test1 = abs(true1['PID']) == 12
true1['isNuE'] = np.array(nue_mask_test1,dtype=bool)
nutau_mask_test1 = abs(true1['PID']) == 16
true1['isNuTau'] = np.array(nutau_mask_test1,dtype=bool)
nu_mask1 = np.logical_or(np.logical_or(numu_mask_test1, nue_mask_test1), nutau_mask_test1)
true1['isNu'] = np.array(nu_mask1,dtype=bool)

#Calculated variables
#print(len(true1['r']), sum([true1['isNu']]))
#true1['r'] = np.zeros((len(true1['x']))) 
#true1['r'][true1['isNu']] = np.sqrt( (true1['x'][true1['isNu']] - x1_origin[true1['isNu']])**2 + (true1['y'][true1['isNu']] - y1_origin[true1['isNu']])**2 ) #only calculate for neutrino events, which have non-None inputs
x1_origin = np.ones((len(reco1['x'])))*46.290000915527344
y1_origin = np.ones((len(reco1['y'])))*-34.880001068115234
reco1['r'] = np.sqrt( (reco1['x'] - x1_origin)**2 + (reco1['y'] - y1_origin)**2 )
weights_squared1 = weights1*weights1
together1 = [str(i) + str(j) + str(k) for i, j, k in zip(true1['run_id'], true1['subrun_id'], true1['event_id'])]
true1['full_ID'] = np.array(together1,dtype=int )
true1['isNC'] = np.logical_not(true1['isCC'])
true1['isCascade'] = np.logical_not(true1['isTrack'])

#RECO masks
mask1 = {}
print(type(reco1['energy']))
print(true1['x'][-10:])
for i in range((len(reco1['energy']))):
	if type(reco1['energy'][i]) is not np.float64:
		print(type(reco1['energy'][i]),reco1['energy'][i],i)
 
mask1['Energy'] = np.logical_and(reco1['energy'] >= cut1['emin'], reco1['energy'] <= cut1['emax'])
mask1['Zenith'] = reco1['coszenith'] <= cut1['coszen']
mask1['R'] = reco1['r'] < cut1['r']
mask1['Z'] = np.logical_and(reco1['z'] > cut1['zmin'], reco1['z'] < cut1['zmax'])
mask1['Vertex'] = np.logical_and(mask1['R'], mask1['Z'])
mask1['ProbMu'] = reco1['prob_mu'] <= cut1['mu']
mask1['Reco'] = np.logical_and(mask1['ProbMu'], np.logical_and(mask1['Zenith'], np.logical_and(mask1['Energy'], mask1['Vertex'])))
mask1['DOM'] = reco1['nDOMs'] >= cut1['nDOM']
mask1['RecoNoEn'] = np.logical_and(mask1['ProbMu'], np.logical_and(mask1['Zenith'], mask1['Vertex']))
mask1['RecoNoZenith'] = np.logical_and(mask1['ProbMu'], np.logical_and(mask1['Energy'], mask1['Vertex']))
mask1['RecoNoZ'] = np.logical_and(mask1['ProbMu'], np.logical_and(mask1['Zenith'], np.logical_and(mask1['Energy'], mask1['R'])))
mask1['RecoNoR'] = np.logical_and(mask1['ProbMu'], np.logical_and(mask1['Zenith'], np.logical_and(mask1['Energy'], mask1['Z'])))
mask1['All'] = true1['energy'] > 0
true1['All'] = true1['energy'] > 0

mask1['Noise'] = mask1['All'] #more_info1['noise_class'] > 0.95
mask1['nhit'] = more_info1['nhit_doms'] > 2.5
mask1['ntop']= more_info1['n_top15'] < 0.5
mask1['nouter'] = more_info1['n_outer'] < 7.5
mask1['CoinHits'] = np.logical_and(np.logical_and(mask1['nhit'], mask1['ntop']), mask1['nouter'])
mask1['MC'] = np.logical_and(np.logical_and(mask1['CoinHits'],mask1['Noise']),mask1['DOM'])

#Combined Masks
mask1['Analysis'] = np.logical_and(mask1['MC'], mask1['Reco'])
mask1['AnalysisNoDOM'] = np.logical_and(np.logical_and(mask1['CoinHits'],mask1['Noise']),mask1['Reco'])

#print("Events file 1: %i, NuMu Rate: %.2e"%(len(true1['energy']),sum(weights1[true1['isNuMu']])))

print("Events file 1: %i, Events after analysis cut: %i"%(len(true1['energy']),sum(mask1['Analysis'])))


"""
if input_file2 is not None:
    #IMPORT FILE 2
    f2 = h5py.File(input_file2, "r")
    truth2 = f2["Y_test_use"][:]
    predict2 = f2["Y_predicted"][:]
    raw_weights2 = f2["weights_test"][:]
    try:
        old_reco2 = f2["reco_test"][:]
    except:
        old_reco2 = None
    try:
        info2 = f2["additional_info"][:]
    except: 
        info2 = None
    f2.close()
    del f2

    #Truth
    true2 = {}
    true2['energy'] = np.array(truth2[:,0])
    true2['em_equiv_energy'] = np.array(truth2[:,14])
    true2['total_daughter_energy'] = np.array(truth2[:,13])
    true2['x'] = np.array(truth2[:,4])
    true2['y'] = np.array(truth2[:,5])
    true2['z'] = np.array(truth2[:,6])
    x2_origin = np.ones((len(true2['x'])))*46.290000915527344
    y2_origin = np.ones((len(true2['y'])))*-34.880001068115234
    true2['r'] = np.sqrt( (true2['x'] - x2_origin)**2 + (true2['y'] - y2_origin)**2 )
    true2['isCC'] = np.array(truth2[:,11],dtype=bool)
    true2['isTrack'] = np.array(truth2[:,8]) == 1
    true2['isCascade'] = np.array(truth2[:,8]) == 0
    true2['PID'] = truth2[:,9]
    true2['zenith'] = np.array(truth2[:,12])
    true2['coszenith'] = np.cos(np.array(truth2[:,12]))
    try:
        true2['daughter_energy'] =  np.array(truth2[:,15])
    except:
        true2['daughter_energy'] = None

    reco2 = {}
    mask2 = {}
    if compare_cnn:
        #Reconstructed values (CNN)
        reco2['energy'] = np.array(predict2[:,0])
        reco2['prob_track'] = np.array(predict2[:,1])
        reco2['zenith'] = np.array(predict2[:,2])
        reco2['coszenith'] = np.cos(reco2['zenith'])
        reco2['x'] = np.array(predict2[:,3])
        reco2['y'] = np.array(predict2[:,4])
        reco2['z'] = np.array(predict2[:,5])
        reco2['r'] = np.sqrt( (reco2['x'] - x2_origin)**2 + (reco2['y'] - y2_origin)**2 )
        reco2['prob_mu'] = np.array(predict2[:,6])
        reco2['nDOMs'] = np.array(predict2[:,7])

        #RECO masks
        mask2['ProbMu'] = reco2['prob_mu'] <= cut2['mu']
        mask2['DOM'] = reco2['nDOMs'] >= cut2['nDOM']

    else:
        reco2['energy'] = np.array(old_reco2[:,0])
        reco2['zenith'] = np.array(old_reco2[:,1])
        reco2['coszenith'] = np.cos(reco2['zenith'])
        reco2['time'] = np.array(old_reco2[:,3])
        reco2['prob_track'] = np.array(old_reco2[:,13])
        reco2['prob_track_full'] = np.array(old_reco2[:,12])
        reco2['x'] = np.array(old_reco2[:,4])
        reco2['y'] = np.array(old_reco2[:,5])
        reco2['z'] = np.array(old_reco2[:,6])
        reco2['r'] = np.sqrt( (reco2['x'] - x2_origin)**2 + (reco2['y'] - y2_origin)**2 )
        reco2['iterations'] = np.array(old_reco2[:,14])
        reco2['nan'] = np.isnan(reco2['energy'])
        
    mask1['Reco'] = np.logical_and(mask1['ProbMu'], np.logical_and(mask1['Zenith'], np.logical_and(mask1['Energy'], mask1['Vertex'])))
    mask1['RecoNoEn'] = np.logical_and(mask1['ProbMu'], np.logical_and(mask1['Zenith'], mask1['Vertex']))

    #PID identification
    muon_mask_test2 = (true2['PID']) == 13
    true2['isMuon'] = np.array(muon_mask_test2,dtype=bool)
    numu_mask_test2 = (true2['PID']) == 14
    true2['isNuMu'] = np.array(numu_mask_test2,dtype=bool)
    nue_mask_test2 = (true2['PID']) == 12
    true2['isNuE'] = np.array(nue_mask_test2,dtype=bool)
    nutau_mask_test2 = (true2['PID']) == 16
    true2['isNuTau'] = np.array(nutau_mask_test2,dtype=bool)
    nu_mask2 = np.logical_or(np.logical_or(numu_mask_test2, nue_mask_test2), nutau_mask_test2)
    true2['isNu'] = np.array(nu_mask2,dtype=bool)

    #Weight adjustments
    weights2 = raw_weights2[:,8]
    if weights2 is not None:
        if sum(true2['isNuMu']) > 1:
            weights2[true2['isNuMu']] = weights2[true2['isNuMu']]/numu_files2
        if sum(true2['isNuE']) > 1:
            weights2[true2['isNuE']] = weights2[true2['isNuE']]/nue_files2
        if sum(true2['isMuon']) > 1:
            weights2[true2['isMuon']] = weights2[true2['isMuon']]/muon_files2
        if sum(nutau_mask_test1) > 1:
            weights2[true2['isNuTau']] = weights2[true2['isNuTau']]/nutau_files2
        true2['run_id'] = np.array(raw_weights2[:,0],dtype=int)
        true2['subrun_id'] = np.array(raw_weights2[:,1],dtype=int)
        true2['event_id'] = np.array(raw_weights2[:,2],dtype=int)
        together2 = [str(i) + str(j) + str(k) for i, j, k in zip(true2['run_id'], true2['subrun_id'], true2['event_id'])]
        true2['full_ID'] = np.array(together2,dtype=int )

    weights_squared2 = weights2*weights2
   
   #Deposited Energy
    if true2['daughter_energy'] is not None:
        true2['deposited_energy'] = np.zeros(len(true2['energy']))
        
        CC_not_tau = np.logical_and(np.logical_or(true2['isNuMu'], true2['isNuE']), true2['isCC'])
        CC_tau = np.logical_and(true2['isCC'], true2['isNuTau'])
        true2['isNC'] = np.logical_not(true2['isCC'])

        #NC
        true2['deposited_energy'][true2['isNC']] = true2['energy'][true2['isNC']] - true2['daughter_energy'][true2['isNC']]
        #Tau CC
        true2['deposited_energy'][CC_tau] = true2['energy'][CC_tau] - (true2['daughter_energy'][CC_tau]*0.5)
        #NuMu CC and NuE CC
        true2['deposited_energy'][CC_not_tau] = true2['total_daughter_energy'][CC_not_tau]
    else:
        true2['deposited_energy'] = None

#Additional info
    more_info2 = {}
    if info2 is not None:
        more_info2['prob_nu'] = info2[:,1]
        more_info2['coin_muon'] = info2[:,0]
        more_info2['true_ndoms'] = info2[:,2]
        more_info2['fit_success'] = info2[:,3]
        more_info2['noise_class'] = info2[:,4]
        more_info2['nhit_doms'] = info2[:,5]
        more_info2['n_top15'] = info2[:,6]
        more_info2['n_outer'] = info2[:,7]
        more_info2['prob_nu2'] = info2[:,8]
        more_info2['total_hits'] = info2[:,9]

    #INFO masks
    if info2 is not None:
        mask2['Hits8'] = more_info2['total_hits'] >= 8
        print("Hits8 failure cuts:", sum(mask2['Hits8']/len(mask2['Hits8'])))
        mask2['oscNext_Nu'] = more_info2['prob_nu'] > 0.4
        mask2['Noise'] = more_info2['noise_class'] > 0.95
        mask2['nhit'] = more_info2['nhit_doms'] > 2.5
        mask2['ntop'] = more_info2['n_top15'] < 2.5
        mask2['nouter'] = more_info2['n_outer'] < 7.5
        mask2['CoinHits'] = np.logical_and(np.logical_and(mask2['nhit'], mask2['ntop']), mask2['nouter'])
        if not compare_cnn:
            mask2['ProbMu'] = mask2['oscNext_Nu']
            mask2['Time'] = reco2['time'] < 14500
            mask2['NotNAN'] = np.logical_not(reco2['nan'])
            mask2['RetroIterations'] = reco2['iterations'] < 10000
            mask2['RetroPass'] = np.logical_and(np.logical_and(mask2['Hits8'],mask2['RetroIterations']), mask2['NotNAN'])
            print("Retro failure cuts:", sum(mask2['RetroPass']/len(mask2['RetroPass'])))
            mask2['Class'] = np.logical_and(mask2['oscNext_Nu'],mask2['Noise'])
            mask2['MC'] = np.logical_and(np.logical_and(mask2['CoinHits'],mask2['Class']), mask2['RetroPass'])
        else:
            mask2['Class'] = mask2['Noise']
            mask2['MC'] = np.logical_and(np.logical_and(mask2['CoinHits'],mask2['Class']),mask2['DOM'])

    #RECO2 MASKS
    mask2['Energy'] = np.logical_and(reco2['energy'] > cut2['emin'], reco2['energy'] < cut2['emax'])
    mask2['Zenith'] = reco2['coszenith'] <= cut2['coszen']
    mask2['R'] = reco2['r'] < cut2['r']
    mask2['Z'] = np.logical_and(reco2['z'] > cut2['zmin'], reco2['z'] < cut2['zmax'])
    mask2['Vertex'] = np.logical_and(mask2['R'], mask2['Z'])
    mask2['Reco'] = np.logical_and(mask2['ProbMu'], np.logical_and(mask2['Zenith'], np.logical_and(mask2['Energy'], mask2['Vertex'])))
    mask2['RecoNoEn'] = np.logical_and(mask2['ProbMu'], np.logical_and(mask2['Zenith'], mask2['Vertex']))
    mask2['RecoNoZenith'] = np.logical_and(mask2['ProbMu'], np.logical_and(mask2['Energy'], mask2['Vertex']))
    mask2['RecoNoZ'] = np.logical_and(mask2['ProbMu'], np.logical_and(mask2['Zenith'], np.logical_and(mask2['Energy'], mask2['R'])))
    mask2['RecoNoR'] = np.logical_and(mask2['ProbMu'], np.logical_and(mask2['Zenith'], np.logical_and(mask2['Energy'], mask2['Z'])))
    mask2['All'] = true2['energy'] > 0
    true2['All'] = true2['energy'] > 0

    if info2 is not None:
        mask2['Analysis'] = np.logical_and(mask2['MC'], mask2['Reco'])

    print("Events file 2: %i, NuMu Rate: %.2e"%(len(true2['energy']),sum(weights2[true2['isNuMu']])))
"""
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

check_depo = true1['energy'] - true1['deposited_energy']
near_zero = check_depo < 1e-3
print("Check deposited different than true 1: %i"%sum(near_zero))
if input_file2 is not None:
	check_depo2 = true2['energy'] - true2['deposited_energy']
	near_zero2 = check_depo2 < 1e-3
	print("Check deposited different than true 2: %i"%sum(near_zero2))

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
if input_file2 is not None:
    if compare_cnn:
        all_remaining2 = np.logical_and(mask2['Analysis'])
    else:
        all_remaining2 = np.logical_and(mask2['Analysis'], mask2['RetroPass'])


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


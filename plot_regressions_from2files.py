import h5py
import argparse
import os, sys
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import matplotlib.colors as colors

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input",type=str,default=None,
                    dest="input_file", help="path and name of the input file")
parser.add_argument("--input2",type=str,default=None,
                    dest="input_file2", help="path and name of the input file")
parser.add_argument("-o", "--outdir",type=str,default='/mnt/home/micall12/LowEnergyNeuralNetwork/output_plots/',
                    dest="output_dir", help="path of ouput file")
parser.add_argument("-n", "--savename",default=None,
                    dest="savename", help="additional directory to save in")
parser.add_argument("--compare_cnn", default=False,action='store_true',
                        dest='compare_cnn',help="compare to CNN")
parser.add_argument("--i3",default=False,action='store_true',
                    dest="i3",help="flag if inputting i3 files (not hdf5)")
parser.add_argument("-f", "--flavor",type=str,default="All",
                    dest="flavor", help="flavor particle to cut on (NuMu, NuE, NuTau, Muon, All)")
parser.add_argument("-s", "--sample",type=str,default="All",
                    dest="sample", help="neutrino interaction to look at (CC, NC, All)")
args = parser.parse_args()

input_file = args.input_file
input_file2 = args.input_file2
compare_cnn = args.compare_cnn
save_folder_name = args.output_dir + "/"
if args.savename is not None:
    save_folder_name += args.savename + "/"
    if os.path.isdir(save_folder_name) != True:
            os.mkdir(save_folder_name)
print("Saving to %s"%save_folder_name)
i3 = args.i3
flavor=args.flavor
sample = args.sample

#CUT values
cut1 = {}
cut1['r'] = 165
cut1['zmin'] = -495
cut1['zmax'] = -225
cut1['coszen'] = 0.3
cut1['emin'] = 5
cut1['emax'] = 100
cut1['mu'] = 0.01
cut1['nDOM'] = 7
numu_files1 = 391
nue_files1 = 91
muon_files1 = 1
nutau_files1 = 187

cut2 = {}
cut2['r'] = 165 #300
cut2['zmin'] = -495 #-500
cut2['zmax'] = -225 #-200
cut2['coszen'] = 0.3
cut2['emin'] = 5
cut2['emax'] = 100 #300
cut2['mu'] = 0.01
cut2['nDOM'] = 7
numu_files2 = 391
nue_files2 = 91
muon_files2 = 1
nutau_files2 = 187

#IMPORT FILE 1
if i3:
    #Find (and edit) number of files
    numu_file_list = list(filter(lambda x: "pass2.14" in x, full_path))
    nue_file_list = list(filter(lambda x: "pass2.12" in x, full_path))
    muon_file_list = list(filter(lambda x: "pass2.13" in x, full_path))
    nutau_file_list = list(filter(lambda x: "pass2.16" in x, full_path))
    if numu_files is None:
        numu_files = len(numu_file_list)
        print("Including all %i NuMu files"%numu_files)
    else:
        numu_files = int(numu_files)
        print("Cutting NuMu files to include first %i files, from %s to %s"%(numu_files,numu_file_list[0],numu_file_list[numu_files-1]))
    if nue_files is None:
        nue_files = len(nue_file_list)
        print("Including all %i NuE files"%nue_files)
    else:
        nue_files = int(nue_files)
        print("Cutting NuE files to include LAST %i files, from %s to %s"%(nue_files,nue_file_list[-nue_files],nue_file_list[-1]))
    if nutau_files is None:
        nutau_files = len(nutau_file_list)
        print("Including all %i NuTau files"%nutau_files)
    else:
        nutau_files = int(nutau_files)
        print("Cutting NuTau files to include first %i files, from %s to %s"%(nutau_files,nutau_file_list[0],nutau_file_list[nutau_files-1]))
    if muon_files is None:
        muon_files = len(muon_file_list)
        print("Including all %i Muon files"%muon_files)
    else:
        muon_files = int(muon_files)
        print("Cutting Muon files to include LAST %i files, from %s to %s"%(muon_files,muon_file_list[-muon_files],muon_file_list[-1]))

    print("Using %i numu files, %i nue files, %i nutau files, %i muon files"%(numu_files, nue_files, nutau_files, muon_files))
    numu_file_list = numu_file_list[:numu_files]
    nue_file_list = nue_file_list[-nue_files:]
    nutau_file_list = nutau_file_list[:nutau_files]
    muon_file_list = muon_file_list[-muon_files:]
    full_path = np.concatenate((numu_file_list,nue_file_list,nutau_file_list,muon_file_list))


    from read_cnn_i3_files import read_i3_files
    variable_list = ["energy", "prob_track", "zenith", "vertex_x", "vertex_y", "vertex_z", "prob_muon", "nDOM", "end_x", "end_y", "end_z"]
    predict1, truth1, old_reco, info1, raw_weights1, input_features_DC, input_features_IC = read_i3_files(full_path,variable_list)

else:
    f = h5py.File(input_file, "r")
    truth1 = f["Y_test_use"][:]
    predict1 = f["Y_predicted"][:]
#reco1 = f["reco_test"][:]
    raw_weights1 = f["weights_test"][:]
    try:
        info1 = f["additional_info"][:]
    except: 
        info1 = None
    f.close()
    del f

#Truth
true1 = {}
true1['energy'] = np.array(truth1[:,0])
true1['em_equiv_energy'] = np.array(truth1[:,14])
true1['x'] = np.array(truth1[:,4])
true1['y'] = np.array(truth1[:,5])
true1['z'] = np.array(truth1[:,6])
x1_origin = np.ones((len(true1['x'])))*46.290000915527344
y1_origin = np.ones((len(true1['y'])))*-34.880001068115234
true1['r'] = np.sqrt( (true1['x'] - x1_origin)**2 + (true1['y'] - y1_origin)**2 )
true1['isCC'] = np.array(truth1[:,11],dtype=bool)
true1['isTrack'] = np.array(truth1[:,8],dtype=bool)
true1['PID'] = truth1[:,9]
true1['zenith'] = np.array(truth1[:,12])
true1['coszenith'] = np.cos(np.array(truth1[:,12]))
#ending calculation
true1['azimuth'] = truth1[:,2]
true1['track_length'] = truth1[:,7]
n_x = np.sin(true1['zenith'])*np.cos(true1['azimuth'])
n_y = np.sin(true1['zenith'])*np.sin(true1['azimuth'])
n_z = np.cos(true1['zenith'])
true1['x_end'] = true1['x'] + true1['track_length']*n_x
true1['y_end'] = true1['y'] + true1['track_length']*n_y
true1['z_end'] = true1['z'] + true1['track_length']*n_z
true1['r_end'] = np.sqrt( (true1['x_end'] - x1_origin)**2 + (true1['y_end'] - y1_origin)**2 )

#Reconstructed values (CNN)
reco1 = {}
reco1['energy'] = np.array(predict1[:,0])
reco1['prob_track'] = np.array(predict1[:,1])
reco1['zenith'] = np.array(predict1[:,2])
reco1['coszenith'] = np.cos(reco1['zenith'])
reco1['x'] = np.array(predict1[:,3])
reco1['y'] = np.array(predict1[:,4])
reco1['z'] = np.array(predict1[:,5])
reco1['r'] = np.sqrt( (reco1['x'] - x1_origin)**2 + (reco1['y'] - y1_origin)**2 )
reco1['prob_mu'] = np.array(predict1[:,6])
reco1['nDOMs'] = np.array(predict1[:,7])
try:
    reco1['x_end'] = np.array(predict1[:,8])
    reco1['y_end'] = np.array(predict1[:,9])
    reco1['z_end'] = np.array(predict1[:,10])
    reco1['r_end'] = np.sqrt( (reco1['x_end'] - x1_origin)**2 + (reco1['y_end'] - y1_origin)**2 )
except:
    reco1['x_end'] = None
    reco1['y_end'] = None
    reco1['z_end'] = None
    reco1['r_end'] = None

#RECO masks
mask1 = {}
mask1['Energy'] = np.logical_and(reco1['energy'] > cut1['emin'], reco1['energy'] < cut1['emax'])
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


#PID identification
muon_mask_test1 = (true1['PID']) == 13
true1['isMuon'] = np.array(muon_mask_test1,dtype=bool)
numu_mask_test1 = (true1['PID']) == 14
true1['isNuMu'] = np.array(numu_mask_test1,dtype=bool)
nue_mask_test1 = (true1['PID']) == 12
true1['isNuE'] = np.array(nue_mask_test1,dtype=bool)
nutau_mask_test1 = (true1['PID']) == 16
true1['isNuTau'] = np.array(nutau_mask_test1,dtype=bool)
nu_mask1 = np.logical_or(np.logical_or(numu_mask_test1, nue_mask_test1), nutau_mask_test1)
true1['isNu'] = np.array(nu_mask1,dtype=bool)

#Weight adjustments
weights1 = raw_weights1[:,8]
if weights1 is not None:
    if sum(true1['isNuMu']) > 1:
        weights1[true1['isNuMu']] = weights1[true1['isNuMu']]/numu_files1
    if sum(true1['isNuE']) > 1:
        weights1[true1['isNuE']] = weights1[true1['isNuE']]/nue_files1
    if sum(true1['isMuon']) > 1:
        weights1[true1['isMuon']] = weights1[true1['isMuon']]/muon_files1
    if sum(nutau_mask_test1) > 1:
        weights1[true1['isNuTau']] = weights1[true1['isNuTau']]/nutau_files1
weights_squared1 = weights1*weights1

#Additional info
more_info1 = {}
if info1 is not None:
    more_info1['prob_nu'] = info1[:,1]
    more_info1['coin_muon'] = info1[:,0]
    more_info1['true_ndoms'] = info1[:,2]
    more_info1['fit_success'] = info1[:,3]
    more_info1['noise_class'] = info1[:,4]
    more_info1['nhit_doms'] = info1[:,5]
    more_info1['n_top15'] = info1[:,6]
    more_info1['n_outer'] = info1[:,7]
    more_info1['prob_nu2'] = info1[:,8]
    more_info1['hits8'] = info1[:,9]

#INFO masks
if info1 is not None:
    mask1['Hits8'] = more_info1['hits8'] == 1
    mask1['oscNext_Nu'] = more_info1['prob_nu'] > 0.4
    mask1['Noise'] = more_info1['noise_class'] > 0.95
    mask1['nhit'] = more_info1['nhit_doms'] > 2.5
    mask1['ntop']= more_info1['n_top15'] < 2.5
    mask1['nouter'] = more_info1['n_outer'] < 7.5
    mask1['Hits'] = np.logical_and(np.logical_and(mask1['nhit'], mask1['ntop']), mask1['nouter'])
    mask1['Class'] = np.logical_and(mask1['oscNext_Nu'],mask1['Noise'])
    mask1['MC'] = np.logical_and(mask1['Hits'],mask1['Class'])

    #Combined Masks
    mask1['Analysis'] = np.logical_and(mask1['MC'], mask1['Reco'])

print("Events file 1: %i, NuMu Rate: %.2e"%(len(true1['energy']),sum(weights1[true1['isNuMu']])))

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
    true2['x'] = np.array(truth2[:,4])
    true2['y'] = np.array(truth2[:,5])
    true2['z'] = np.array(truth2[:,6])
    x2_origin = np.ones((len(true2['x'])))*46.290000915527344
    y2_origin = np.ones((len(true2['y'])))*-34.880001068115234
    true2['r'] = np.sqrt( (true2['x'] - x2_origin)**2 + (true2['y'] - y2_origin)**2 )
    true2['isCC'] = np.array(truth2[:,11],dtype=bool)
    true2['isTrack'] = np.array(truth2[:,8])
    true2['PID'] = truth2[:,9]
    true2['zenith'] = np.array(truth2[:,12])
    true2['coszenith'] = np.cos(np.array(truth2[:,12]))

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
            weights2[true2['isMuon']] = weights2[true['isMuon']]/muon_files2
        if sum(nutau_mask_test1) > 1:
            weights2[true2['isNuTau']] = weights2[true2['isNuTau']]/nutau_files2
    weights_squared2 = weights2*weights2

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
        more_info2['hits8'] = info2[:,9]

    #INFO masks
    if info2 is not None:
        mask2['Hits8'] = more_info2['hits8'] == 1
        mask2['oscNext_Nu'] = more_info2['prob_nu'] > 0.4
        mask2['Noise'] = more_info2['noise_class'] > 0.95
        mask2['nhit'] = more_info2['nhit_doms'] > 2.5
        mask2['ntop'] = more_info2['n_top15'] < 2.5
        mask2['nouter'] = more_info2['n_outer'] < 7.5
        mask2['Hits'] = np.logical_and(np.logical_and(mask2['nhit'], mask2['ntop']), mask2['nouter'])
        mask2['Class'] = np.logical_and(mask2['oscNext_Nu'],mask2['Noise'])
        mask2['MC'] = np.logical_and(mask2['Hits'],mask2['Class'])
        if not compare_cnn:
            mask2['ProbMu'] = mask2['oscNext_Nu']
            mask2['Time'] = reco2['time'] < 14500

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

#Plot
from PlottingFunctions import plot_distributions
from PlottingFunctions import plot_2D_prediction
from PlottingFunctions import plot_single_resolution
from PlottingFunctions import plot_bin_slices
from PlottingFunctions import plot_rms_slices

save=True
if save ==True:
    print("Saving to %s"%save_folder_name)
save_base_name = save_folder_name
path=save_folder_name

variable_names = ['energy', 'coszenith', 'z', 'r', 'x_end', 'y_end', 'z_end', 'r_end', 'x', 'y']
############## CHANGE THESE LINES ##############
variable_index = 0 #chose variable from list above
plot_confusion = False
cut_or = False
energy_type = "True" #"EM Equiv" #or True
##################################################

if flavor == "All":
    flavor_key = flavor
else:
    flavor_key = "is%s"%flavor
if sample =="CC" or sample == "NC":
    select = "is%s"%sample
else:
    select = sample

var_type="True" #labeling purposes
if energy_type is "EM Equiv":
    use_em = 'em_equiv_'
    if variable_index == 0:
        var_type="EM Equiv"
else:
    use_em = ''
print("using flavor %s and sample %s and energy type %s"%(flavor, sample,var_type))

check1 = np.logical_and(np.logical_and(true1[flavor_key],true1[select]), mask1['Reco'])
check1_add = np.logical_and(check1, mask1['DOM'])
print(cut1['nDOM'],sum(weights1[check1_add])/sum(weights1[check1]))

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

a_mask1 = np.logical_and(np.logical_and(np.logical_and(true1[flavor_key],true1[select]), mask1[maskname]), mask1['DOM'])
variable_mask1 = np.logical_and(a_mask1,mask1[keyname])
a_mask1 = variable_mask1
if input_file2 is not None:
    if compare_cnn:
        a_mask2 = np.logical_and(np.logical_and(np.logical_and(true2[flavor_key],true2[select]), mask2[maskname]), mask2['DOM'])
    else:
        a_mask2 = np.logical_and(np.logical_and(true2[flavor_key],true2[select]), mask2[maskname])
    variable_mask2 = np.logical_and(a_mask2,mask2[keyname])
    a_mask2 = variable_mask2

name1 = "CNN"
name2 = "Retro"
logmax = 10**1.5
bins_log = 10**np.linspace(0,1.5,100)

true1_value = true1[use_em + variable_name][a_mask1]
reco1_value = reco1[variable_name][a_mask1]
true1_value_res = true1[use_em + variable_name][variable_mask1]
reco1_value_res = reco1[variable_name][variable_mask1]
weights1_value = weights1[a_mask1]
weights1_value_res = weights1[variable_mask1]
true1_energy_res = true1[use_em + 'energy'][variable_mask1]
if cut_min is not None:
    if cut_max is not None:
        if cut_or:
            true1_binary = np.logical_or(true1[variable_name][a_mask1] > cut_min, true1[variable_name][a_mask1] < cut_max)
            reco1_binary = np.logical_or(reco1[variable_name][a_mask1] > cut_min, reco1[variable_name][a_mask1] < cut_max)
            print(variable_name, "Checking > ", cut_min, " OR < ", cut_max)
        else:
            true1_binary = np.logical_and(true1[variable_name][a_mask1] > cut_min, true1[variable_name][a_mask1] < cut_max)
            reco1_binary = np.logical_and(reco1[variable_name][a_mask1] > cut_min, reco1[variable_name][a_mask1] < cut_max)
            print(variable_name, "Checking > ", cut_min, " AND < ", cut_max)
    else:
        true1_binary = true1[variable_name][a_mask1] > cut_min
        reco1_binary = reco1[variable_name][a_mask1] > cut_min
        print(variable_name, "Checking > ", cut_min)
else:
    true1_binary = true1[variable_name][a_mask1] < cut_max
    reco1_binary = reco1[variable_name][a_mask1] < cut_max
    print(variable_name, "Checking < ", cut_max)

print(true1_binary[:10],reco1_binary[:10])
print(sum(weights1_value_res)/sum(weights1[true1['isCC']]))
print(true1_value[:10], reco1_value[:10])

if input_file2 is not None:
    true2_value = true2[use_em + variable_name][a_mask2]
    reco2_value = reco2[variable_name][a_mask2]
    true2_value_res = true2[use_em + variable_name][variable_mask2]
    reco2_value_res = reco2[variable_name][variable_mask2]
    weights2_value = weights2[a_mask2]
    weights2_value_res = weights2[variable_mask2]
    true2_energy_res = true2[use_em + 'energy'][variable_mask2]
    if cut_min is not None:
        if cut_max is not None:
            true2_binary = np.logical_and(true2[variable_name][a_mask2] > cut_min, true2[variable_name][a_mask2] < cut_max)
            reco2_binary = np.logical_and(reco2[variable_name][a_mask2] > cut_min, reco2[variable_name][a_mask2] < cut_max)
            print(variable_name, "Checking > ", cut_min, " AND < ", cut_max)
        else:
            true2_binary = true2[variable_name][a_mask2] > cut_min
            reco2_binary = reco2[variable_name][a_mask2] > cut_min
            print(variable_name, "Checking > ", cut_min)
    else:
        true2_binary = true2[variable_name][a_mask2] < cut_max
        reco2_binary = reco2[variable_name][a_mask2] < cut_max
        print(variable_name, "Checking < ", cut_max)
    print(sum(weights2_value_res)/sum(weights2[true2['isCC']]))
    print(true2_value[:10], reco2_value[:10])
else:
    true2_value = None 
    reco2_value = None
    true2_value_res = None
    reco2_value_res = None
    weights2_value = None
    weights2_value_res = None
    true2_energy = None

"""
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
plt.savefig("%s/%sLogEnergyDist_ZoomInLE.png"%(path,name1.replace(" ","")))

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
plt.savefig("%s/%sLogEnergyDist_ZoomInLE.png"%(path,name2.replace(" ","")))
"""
plot_distributions(true1_value, reco1_value,
                    weights=weights1_value,
                    save=save, savefolder=save_folder_name,
                    cnn_name = name1, variable=plot_name, units= plot_units,
                    minval=minval,maxval=maxval,bins=bins,true_name=energy_type)

if input_file2 is not None:
    plot_distributions(true2_value, old_reco=reco2_value,
                    weights=weights2_value,
                    save=save, savefolder=save_folder_name,
                    reco_name = name2, variable=plot_name, units= plot_units,
                    minval=minval,maxval=maxval,bins=bins,true_name=energy_type)

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
"""
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
"""
#Resolution
if input_file2 is None:
    use_old_reco = False
else:
    use_old_reco = True
plot_single_resolution(true1_value_res, reco1_value_res, 
                   weights=weights1_value_res,old_reco_weights=weights2_value_res,
                   use_old_reco = use_old_reco, old_reco = reco2_value_res,
                   old_reco_truth=true2_value_res,\
                   minaxis=-res_range, maxaxis=res_range, bins=bins,\
                   save=save, savefolder=save_folder_name,\
                   variable=plot_name, units=plot_units, reco_name=name1,
                   flavor=flavor,sample=sample)

plot_single_resolution(true1_value_res, reco1_value_res,
                    weights=weights1_value_res,
                    old_reco_weights=weights2_value_res,\
                    use_old_reco = use_old_reco, old_reco = reco2_value_res,
                    old_reco_truth=true2_value_res,\
                    minaxis=-frac_res_range, maxaxis=frac_res_range, 
                    bins=bins, use_fraction=True,\
                    save=save, savefolder=save_folder_name,\
                    variable=plot_name, units=plot_units, reco_name=name1,
                    flavor=flavor,sample=sample)

#Bin Slices
plot_bin_slices(true1_value_res, reco1_value_res, 
                old_reco = reco2_value_res,old_reco_truth=true2_value_res,
                weights=weights1_value_res, old_reco_weights=weights2_value_res,\
                use_fraction = binned_frac, bins=syst_bin, 
                min_val=minval, max_val=maxval,\
                save=save, savefolder=save_folder_name,
                variable=plot_name, units=plot_units, 
                cnn_name=name1, reco_name=name2,variable_type=var_type,
                add_contour=True, flavor=flavor,sample=sample)

if variable_name is not 'energy':
    plot_bin_slices(true1_value_res, reco1_value_res, 
                energy_truth=true1_energy_res,
                old_reco = reco2_value_res, old_reco_truth=true2_value_res,
                reco_energy_truth = true2_energy_res,
                weights=weights1_value_res, old_reco_weights=weights2_value_res,\
                use_fraction = binned_frac, bins=syst_bin,
                min_val=minvals[0], max_val=maxvals[0],\
                save=save, savefolder=save_folder_name,
                variable=plot_name, units=plot_units, 
                cnn_name=name1, reco_name=name2,
                energy_type=energy_type,
                flavor=flavor,sample=sample)

plot_bin_slices(true1_value, reco1_value, 
                old_reco = reco2_value,old_reco_truth=true2_value,
                weights=weights1_value, old_reco_weights=weights2_value,\
                use_fraction = binned_frac, bins=syst_bin, 
                min_val=minval, max_val=maxval,\
                save=save, savefolder=save_folder_name,
                variable=plot_name, units=plot_units, 
                cnn_name=name1, reco_name=name2,variable_type=var_type,
                vs_predict=True,flavor=flavor,sample=sample)

if plot_confusion:
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

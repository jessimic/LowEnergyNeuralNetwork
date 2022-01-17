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

#CUT values
r_cut1 = 165
zmin_cut1 = -495
zmax_cut1 = -225
coszen_cut1 = 0.3
emin_cut1 = 5
emax_cut1 = 100
mu_cut1 = 0.01
nDOM_cut1 = 9
numu_files1 = 391
nue_files1 = 1
muon_files1 = 1
nutau_files1 = 1

r_cut2 = 165
zmin_cut2 = -495
zmax_cut2 = -225
coszen_cut2 = 0.3
emin_cut2 = 5
emax_cut2 = 100
mu_cut2 = 0.01
nDOM_cut2 = 9
numu_files2 = 391
nue_files2 = 1
muon_files2 = 1
nutau_files2 = 1


#IMPORT FILE 1
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

#RECO masks
mask1 = {}
mask1['Energy'] = np.logical_and(reco1['energy'] > emin_cut1, reco1['energy'] < emax_cut1)
mask1['Zenith'] = reco1['coszenith'] <= coszen_cut1
mask1['R'] = reco1['r'] < r_cut1
mask1['Z'] = np.logical_and(reco1['z'] > zmin_cut1, reco1['z'] < zmax_cut1)
mask1['Vertex'] = np.logical_and(mask1['R'], mask1['Z'])
mask1['ProbMu'] = reco1['prob_mu'] <= mu_cut1
mask1['Reco'] = np.logical_and(mask1['ProbMu'], np.logical_and(mask1['Zenith'], np.logical_and(mask1['Energy'], mask1['Vertex'])))
mask1['RecoNoEn'] = np.logical_and(mask1['ProbMu'], np.logical_and(mask1['Zenith'], mask1['Vertex']))
mask1['DOM'] = reco1['nDOMs'] >= nDOM_cut1

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
        weights1[true1['isMuon']] = weights1[true['isMuon']]/muon_files1
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

#IMPORT FILE 2
f2 = h5py.File(input_file2, "r")
truth2 = f2["Y_test_use"][:]
predict2 = f2["Y_predicted"][:]
raw_weights2 = f2["weights_test"][:]
try:
    reco2 = f2["reco_test"][:]
except:
    reco2 = None
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
    mask2['ProbMu'] = reco2['prob_mu'] <= mu_cut2
    mask2['DOM'] = reco2['nDOMs'] >= nDOM_cut2

else:
    reco2['energy'] = np.array(reco2[:,0])
    reco2['zenith'] = np.array(reco2[:,1])
    reco2['coszenith'] = np.cos(reco2['zenith'])
    reco2['time'] = np.array(reco2[:,3])
    reco2['prob_track'] = np.array(reco2[:,13])
    reco2['prob_track_full'] = np.array(reco2[:,12])
    reco2['x'] = np.array(reco2[:,4])
    reco2['y'] = np.array(reco2[:,5])
    reco2['z'] = np.array(reco2[:,6])
    reco2['r'] = np.sqrt( (reco2['x'] - x2_origin)**2 + (reco2['y'] - y2_origin)**2 )
    reco2['iterations'] = np.array(reco2[:,14])
    reco2['nan'] = np.isnan(reco2['energy'])
    
#RECO2 MASKS
mask2['Energy'] = np.logical_and(reco2['energy'] > emin_cut2, reco2['energy'] < emax_cut2)
mask2['Zenith'] = reco2['coszenith'] <= coszen_cut2
mask2['R'] = reco2['r'] < r_cut2
mask2['Z'] = np.logical_and(reco2['z'] > zmin_cut2, reco2['z'] < zmax_cut2)
mask2['Vertex'] = np.logical_and(mask2['R'], mask2['Z'])
mask2['Reco'] = np.logical_and(mask2['ProbMu'], np.logical_and(mask2['Zenith'], np.logical_and(mask2['Energy'], mask2['Vertex'])))
mask2['RecoNoEn'] = np.logical_and(mask2['ProbMu'], np.logical_and(mask2['Zenith'], mask2['Vertex']))
    

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
    mask2['ProbMu'] = reco2['prob_mu'] <= mu_cut2

    #Combined Masks
    mask2['Analysis'] = np.logical_and(mask2['MC'], mask2['Reco'])

#Print Summary of the two files
print("Events file 1: %i, NuMu Rate: %.2e"%(len(true1['energy']),sum(weights1[true1['isNuMu']])))
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

plot_name = "Energy"
plot_units = "(GeV)"
maxabs_factors = 100.

ana_mask1 = np.logical_and(true1['isCC'], mask1['Reco'])
ana_mask2 = np.logical_and(true2['isCC'], mask2['Reco'])
mask1 = np.logical_and(np.logical_and(true1['isCC'], mask1['RecoNoEn']),mask1['DOM'])
mask2 = np.logical_and(np.logical_and(true2['isCC'], mask2['RecoNoEn']),mask2['DOM'])
save_base_name = save_folder_name
minval = 5
maxval = 100
logmax = 10**1.5
bins_log = 10**np.linspace(0,1.5,100)
bins = 95
syst_bin = 95
name1 = "Old Energy"
name2 = "New Energy"
units = "(GeV)"

true1_value = true1['energy'][mask1]
reco1_value = reco1['energy'][mask1]
true2_value = true2['energy'][mask2]
reco2_value = reco2['energy'][mask2]
weights1_value = weights1[mask1]
weights2_value = weights2[mask2]

print(sum(weights1_value)/sum(weights1[true1['isCC']]), sum(weights1_value)/sum(weights1[ana_mask1]))
print(true1_value[:10], reco1_value[:10])
print(sum(weights2_value)/sum(weights2[true2['isCC']]), sum(weights2_value)/sum(weights2[ana_mask2]))
print(true2_value[:10], reco2_value[:10])

path=save_folder_name
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
plt.hist(true1_value, label="true",bins=100,
        range=[minval,maxval],weights=weights1_value,alpha=0.5)
plt.hist(reco1_value, label=name1,bins=100,
        range=[minval,maxval],weights=weights1_value,alpha=0.5)
plt.title("Energy Distribution Weighted for %s events"%len(true1_value),fontsize=25)
plt.xlabel("Energy (GeV)",fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.axvline(5,linewidth=3,linestyle="--",color='k',label="Cut at 5 GeV")
plt.legend(fontsize=15)
plt.savefig("%s/%sEnergyDist.png"%(path,name1.replace(" ","")))

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

plt.figure(figsize=(10,7))
plt.hist(true2_value, label="true",bins=100,
        range=[minval,maxval],weights=weights2_value,alpha=0.5)
plt.hist(reco2_value, label=name2,bins=100,
        range=[minval,maxval],weights=weights2_value,alpha=0.5)
plt.title("Energy Distribution Weighted for %s events"%len(true2_value),fontsize=25)
plt.xlabel("Energy (GeV)",fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.axvline(5,linewidth=3,linestyle="--",color='k',label="Cut at 5 GeV")
plt.legend(fontsize=15)
plt.savefig("%s/%sEnergyDist.png"%(path,name2.replace(" ","")))

switch = False
plot_2D_prediction(true1_value, reco1_value,
                    weights=weights1_value,\
                    save=save, savefolder=save_folder_name,
                    bins=bins, switch_axis=switch,
                    variable=plot_name, units=plot_units, reco_name=name1)

plot_2D_prediction(true2_value, reco2_value,
                    weights=weights2_value,
                    save=save, savefolder=save_folder_name,
                    bins=bins,switch_axis=switch,\
                    variable=plot_name, units=plot_units, reco_name=name2)

plot_2D_prediction(true1_value, reco1_value,
                    weights=weights1_value,\
                    save=save, savefolder=save_folder_name,
                    bins=bins,switch_axis=switch,\
                    minval=minval, maxval=maxval, axis_square=True,\
                    variable=plot_name, units=plot_units, reco_name=name1)

plot_2D_prediction(true2_value, reco2_value,
                    weights=weights2_value,
                    save=save, savefolder=save_folder_name,
                    bins=bins,switch_axis=switch,\
                    minval=minval, maxval=maxval, axis_square=True,\
                    variable=plot_name, units=plot_units, reco_name=name2)

switch = True
plot_2D_prediction(true1_value, reco1_value,
                    weights=weights1_value,\
                    save=save, savefolder=save_folder_name,
                    bins=bins, switch_axis=switch,
                    variable=plot_name, units=plot_units, reco_name=name1)

plot_2D_prediction(true2_value, reco2_value,
                    weights=weights2_value,
                    save=save, savefolder=save_folder_name,
                    bins=bins,switch_axis=switch,\
                    variable=plot_name, units=plot_units, reco_name=name2)

plot_2D_prediction(true1_value, reco1_value,
                    weights=weights1_value,\
                    save=save, savefolder=save_folder_name,
                    bins=bins,switch_axis=switch,\
                    minval=minval, maxval=maxval,
                    cut_truth=True, axis_square=True,\
                    variable=plot_name, units=plot_units, reco_name=name1)

plot_2D_prediction(true2_value, reco2_value,
                    weights=weights2_value,
                    save=save, savefolder=save_folder_name,
                    bins=bins,switch_axis=switch,\
                    minval=minval, maxval=maxval,
                    cut_truth=True, axis_square=True,\
                    variable=plot_name, units=plot_units, reco_name=name2)

#Resolution
plot_single_resolution(true1_value, reco1_value, 
                   weights=weights1_value,old_reco_weights=weights2_value,
                   use_old_reco = True, old_reco = reco2_value,
                   old_reco_truth=true2_value,\
                   minaxis=-maxval, maxaxis=maxval, bins=bins,\
                   save=save, savefolder=save_folder_name,\
                   variable=plot_name, units=plot_units, reco_name=name2)

plot_single_resolution(true1_value, reco1_value,
                    weights=weights1_value,old_reco_weights=weights2_value,\
                    use_old_reco = True, old_reco = reco2_value,
                    old_reco_truth=true2_value,\
                    minaxis=-2., maxaxis=2, bins=bins, use_fraction=True,\
                    save=save, savefolder=save_folder_name,\
                    variable=plot_name, units=plot_units, reco_name=name2)

#Bin Slices
plot_bin_slices(true1_value, reco1_value, 
                old_reco = reco2_value,old_reco_truth=true2_value,
                weights=weights1_value, old_reco_weights=weights2_value,\
                use_fraction = True, bins=syst_bin, 
                min_val=minval, max_val=maxval,\
                save=save, savefolder=save_folder_name,
                variable=plot_name, units=plot_units, 
                cnn_name=name1, reco_name=name2,add_contour=True)


"""
plot_bin_slices(true1_energy[mask1], reco1_energy[mask1], 
                energy_truth=true1_energy[mask1],
                old_reco = reco2_energy[mask2],old_reco_truth=true2_energy[mask2],
                reco_energy_truth = true2_energy[mask2],
                weights=weights1[mask1], old_reco_weights=weights2[mask2],\
                use_fraction = True, bins=syst_bin, 
                min_val=minval, max_val=maxval,\
                save=save, savefolder=save_folder_name,
                variable=plot_name, units=plot_units, 
                cnn_name=name1, reco_name=name2)

reco_nan = np.isnan(retro_energy)
not_nan = np.logical_not(reco_nan)
assert sum(not_nan) > 0, "Retro is all nans"
cuts = np.logical_and(cuts,not_nan)
reco_nan2 = np.isnan(retro_energy2)
not_nan2 = np.logical_not(reco_nan2)
assert sum(not_nan2) > 0, "Retro is all nans"
cuts2 = np.logical_and(cuts2,not_nan2)
plot_bin_slices(true_energy[cuts], retro_energy[cuts], weights=weights[cuts], old_reco_weights=weights2[cuts2],\

                    old_reco = retro_energy2[cuts2],old_reco_truth=true_energy2[cuts2],\
                    use_fraction = True, bins=syst_bin, min_val=minval, max_val=maxval,\
                    save=save, savefolder=save_folder_name,\
                    variable=plot_name, units=plot_units, cnn_name=name1, reco_name=name2)
"""

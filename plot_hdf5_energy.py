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
parser.add_argument("-o", "--outdir",type=str,default='/mnt/home/micall12/LowEnergyNeuralNetwork/output_plots/',
                    dest="output_dir", help="path of ouput file")
args = parser.parse_args()

input_file = args.input_file
save_folder_name = args.output_dir

f = h5py.File(input_file, "r")
cnn_energy = f["cnn_energy"][:]
true_energy = f["true_energy"][:]
retro_energy = f["retro_energy"][:]
retro_zenith = f["retro_zenith"][:]
true_x = f["true_x"][:]
true_y = f["true_y"][:]
true_z = f["true_z"][:]
true_CC = f["true_CC"][:]
true_ndoms = f["true_ndoms"][:]
fit_success = f["fit_success"][:]
weights = f["weight"][:]
coin_muon = f["coin_muon"][:]
prob_nu = f["prob_nu"][:]
reco_x = f["reco_x"][:]
reco_y = f["reco_y"][:]
reco_z = f["reco_z"][:]
reco_r = f["reco_r"][:]
f.close()
del f
        
cnn_energy = np.array(cnn_energy)
true_energy = np.array(true_energy)
retro_energy = np.array(retro_energy)
retro_zenith = np.array(retro_zenith)
true_x = np.array(true_x)
true_y = np.array(true_y)
true_z = np.array(true_z)
true_CC = np.array(true_CC)
true_ndoms = np.array(true_ndoms)
fit_success = np.array(fit_success)
weights = np.array(weights)
coin_muon = np.array(coin_muon)
prob_nu = np.array(prob_nu)
reco_x = np.array(reco_x)
reco_y = np.array(reco_y)
reco_z = np.array(reco_z)
reco_r = np.array(reco_r)
retro_coszen = np.cos(retro_zenith)

#Vertex Position
x_origin = np.ones((len(true_x)))*46.290000915527344
y_origin = np.ones((len(true_y)))*-34.880001068115234
true_r = np.sqrt( (true_x - x_origin)**2 + (true_y - y_origin)**2 )

def plot_vertex(radius,z):
    fig = plt.figure(figsize=(14,12))
    ax=fig.add_subplot(111)
    ax.plot(radius,z,'g.',label="starting vertex",zorder=1)
    DC = patches.Rectangle((0,-500),90,300,zorder=10,fill=False)
    DC_extended = patches.Rectangle((0,-500),150,300,zorder=10,Fill=False)
    boxes = [DC,DC_extended]
    acolor=["black","brown"]
    pc = PatchCollection(boxes,edgecolor=acolor,linewidth=3,zorder=10)
    ax.add_collection(pc)
    ax.set_title("Starting Track Position")
    ax.set_ylabel("Z position (m)")
    ax.set_xlabel("Radial position (m)")
    plt.savefig("%sTrueVertexPosition.png"%(save_folder_name))

def plot_NDOMS(ndoms,radius,cut=250,bins=100):
    mask = radius > cut
    plt.figure(figsize=(14,12))
    plt.hist(ndoms[mask],bins=bins)
    plt.title("Number of Cleaned DOMs Hit (r > %i m)"%cut)
    plt.xlabel("Number Cleaned DOMs Hit")
    plt.savefig("%sNDOMsRgt%i.png"%(save_folder_name,cut))
    plt.close()

def plot_ZR(z,radius,energy=None,weights=None,cut_min=0,cut_max=200,bins=100,rmax=200,save_folder_name=None):
    if energy is not None:
        mask = np.logical_and(energy > cut_min, energy < cut_max)
    else:
        mask = radius > 0
    plt.figure(figsize=(14,12))
    plt.hist(z[mask],bins=bins,range=[-650,-100],weights=weights[mask])
    plt.title("Starting Vertex Z Position (%i > e > %i GeV)"%(cut_min,cut_max))
    plt.xlabel("Z Position (m)")
    plt.savefig("%sStartingZVertexE%ito%i.png"%(save_folder_name,cut_min,cut_max))
    plt.close()

    plt.figure(figsize=(14,12))
    plt.hist(radius[mask]*radius[mask],bins=bins,range=[0,900],weights=weights[mask])
    plt.title("Starting Vertex R Position (%i > e > %i GeV)"%(cut_min,cut_max))
    plt.xlabel("Radial Position (m)")
    plt.savefig("%sStartingRVertexE%ito%i.png"%(save_folder_name,cut_min,cut_max))
    plt.close()
    rmask = np.logical_and(mask, radius > rmax)
    if energy is not None:
        print("Energy range: [%i, %i]"%(cut_min, cut_max))
    print("Percent r > %i: %f"%(rmax,100*sum(weights[rmask])/sum(weights[mask])))

def plot_vs_Energy(energy,variable2,weights=None,variable_name="ndoms",\
                        xmin=None,ymin=None,ymax=None,xmax=None,\
                        log=True,zmax=None,bins=200,units="",savefolder=None):
        
    if xmin is None:
        xmin = min(energy)
    if ymin is None:
        ymin = min(variable2)
    if xmax is None:
        xmax = max(energy)
    if ymax is None:
        ymax = max(variable2)
    if weights is None:
        cmin = 1
    else:
        cmin = 1e-12

    print("XRange: %f - %f, YRange: %f - %f"%(xmin,xmax,ymin,ymax))

    plt.figure(figsize=(10,7))
    if log:
        cts,xbin,ybin,img = plt.hist2d(energy, variable2, bins=bins,range=[[xmin,xmax],[ymin,ymax]],\
                        cmap='viridis_r', norm=colors.LogNorm(), weights=weights, cmax=zmax, cmin=cmin)
    else:
        cts,xbin,ybin,img = plt.hist2d(energy, variable2, bins=bins,range=[[xmin,xmax],[ymin,ymax]],\
                        cmap='viridis_r', weights=weights, cmax=zmax, cmin=cmin)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('counts', rotation=90)
    plt.xlabel("True Neutrino Energy (GeV)",fontsize=20)
    plt.ylabel("%s %s"%(variable_name,units),fontsize=20)
    title = "%s vs True Energy"%(variable_name)
    if weights is not None:
        title += " Weighted"
    plt.plot([xmin,xmax],[ymin,ymax],'k:',label="1:1")
    plt.savefig("%sTrueEnergyVs%s_2DHist.png"%(savefolder,variable_name),bbox_inches='tight')
    plt.close()

#plot_ZR(true_z,true_r,true_energy,weights=weights,cut_min=0,cut_max=5,bins=100,save_folder_name=save_folder_name)
#plot_ZR(true_z,true_r,true_energy,weights=weights,cut_min=10,cut_max=20,bins=100,save_folder_name=save_folder_name)
#plot_ZR(true_z,true_r,true_energy,weights=weights,cut_min=min(true_energy),cut_max=max(true_energy),bins=100,save_folder_name=save_folder_name)

print("Energy range: [%i, %i]"%(min(true_energy), max(true_energy)))
big = [100, 200, 300, 400, 500]
for an_energy in big:
    energy_big = true_energy>an_energy
    print("Percent energy > %i: %f"%(an_energy,100*sum(weights[energy_big])/sum(weights)))
energy_small = true_energy<5
print("Percent energy < 5: %f"%(100*sum(weights[energy_small])/sum(weights)))

plt.figure(figsize=(14,12))
a_max = 500
e_counts = []
w_counts = []
for i in range(0,a_max):
    e_mask = true_energy > i
    e_counts.append(sum(e_mask))
    w_counts.append(sum(w_counts[e_mask]))

plt.hist(e_counts,bins=a_max,range=[0,a_max],weights=weights)
plt.title("Cumulative Energy Plot",fontsize=20)
plt.xlabel("Energy (GeV)",fontsize=20)
plt.hlines(0.98, 0, a_max,label="0.98") 
plt.legend(fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig("%sCumulativeEnergy.png"%(save_folder_name))
plt.close()

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

maskNONE = true_energy > 0.
assert sum(maskNONE)==len(true_energy), "Some true energy < 0? Check!" 
maskCC = true_CC == 1
maskZ = np.logical_and(true_z > -505, true_z < 192)
maskR = true_r < 90.
maskDC = np.logical_and(maskZ,maskR)
maskE = np.logical_and(true_energy > 5., true_energy < 100.)
maskNu = prob_nu > 0.3
maskMu = coin_muon > 0
maskReco = np.logical_and(prob_nu,coin_muon)
maskRecoZ = np.logical_and(reco_z > -500., reco_z < -200.)
maskRecoR = reco_r < 300.
maskRecoDC = np.logical_and(maskRecoZ, maskRecoR)
maskRecoBins = np.logical_and(retro_energy < 300, retro_coszen < 0.3)
maskANA = np.logical_and(np.logical_and(maskRecoDC, maskReco), maskRecoBins)

#plot_vs_Energy(true_energy[maskCC],true_ndoms[maskCC],weights=weights[maskCC],bins=100,savefolder=save_folder_name)
#plot_vs_Energy(true_energy[maskCC],true_ndoms[maskCC],weights=weights[maskCC],xmax=300,bins=100,savefolder=save_folder_name)


cut_list = [maskCC, np.logical_and(maskE, maskCC), np.logical_and(maskANA, maskCC)]
cut_names = ["WeightedNoCuts_CC", "WeightedNoCutsE5100_CC", "WeightedRetroCuts_CC"]
minvals = [1, 5, 1]
maxvals = [200, 100, 200]
binss = [199, 95, 199]
syst_bins = [20, 10, 20]
save_base_name = save_folder_name
"""
for cut_index in range(0,len(cut_list)):
    cuts = cut_list[cut_index]
    folder_name = cut_names[cut_index]
    minval = minvals[cut_index]
    maxval = maxvals[cut_index]
    bins = binss[cut_index]
    syst_bin = syst_bins[cut_index]

    print("Working on %s"%folder_name)

    save_folder_name = save_base_name + "/%s/"%folder_name
    if os.path.isdir(save_folder_name) != True:
        os.mkdir(save_folder_name)


    plot_NDOMS(true_ndoms[cuts],true_r[cuts])
    plot_NDOMS(true_ndoms[cuts],true_r[cuts],cut=0)
    
    plot_distributions(true_energy[cuts], cnn_energy[cuts], old_reco=retro_energy[cuts],\
                                save=save, savefolder=save_folder_name, weights=weights[cuts],\
                                reco_name = "Retro", variable=plot_name, units= plot_units,
                                minval=minval,maxval=maxval,bins=bins)
    plot_distributions(true_energy[cuts], cnn_energy[cuts], old_reco=retro_energy[cuts],\
                                save=save, savefolder=save_folder_name, weights=weights[cuts],\
                                reco_name = "Retro", variable=plot_name, units= plot_units)
    plot_distributions(true_r[cuts], reco_r[cuts],\
                                save=save, savefolder=save_folder_name, weights=weights[cuts],\
                                cnn_name = "Retro", variable="Radial Vertex", units= "(m)",log=True)
    plot_distributions(true_z[cuts], reco_z[cuts],\
                                save=save, savefolder=save_folder_name, weights=weights[cuts],\
                                cnn_name = "Retro", variable="Z Vertex", units= "(m)",log=True)
    
    plot_2D_prediction(true_energy[cuts], cnn_energy[cuts],weights=weights[cuts],\
                            save=save, savefolder=save_folder_name,bins=bins,
                            minval=minval, maxval=maxval, cut_truth=True, axis_square=True,\
                            variable=plot_name, units=plot_units, reco_name="CNN")
    plot_2D_prediction(true_energy[cuts], retro_energy[cuts], weights=weights[cuts],
                            save=save, savefolder=save_folder_name,bins=bins,\
                            minval=minval, maxval=maxval, cut_truth=True, axis_square=True,\
                            variable=plot_name, units=plot_units, reco_name="Retro")
    
    plot_single_resolution(true_energy[cuts], cnn_energy[cuts], weights=weights[cuts],\
                       use_old_reco = True, old_reco = retro_energy[cuts],\
                       minaxis=-maxval, maxaxis=maxval, bins=bins,\
                       save=save, savefolder=save_folder_name,\
                       variable=plot_name, units=plot_units, reco_name="Retro")
    
    plot_bin_slices(true_energy[cuts], cnn_energy[cuts], weights=weights[cuts],  
                        old_reco = retro_energy[cuts],\
                        use_fraction = True, bins=syst_bin, min_val=minval, max_val=maxval,\
                        save=save, savefolder=save_folder_name,\
                        variable=plot_name, units=plot_units, reco_name="Retro")

    plot_rms_slices(true_energy[cuts], cnn_energy[cuts], weights=weights[cuts],  
                        old_reco = retro_energy[cuts],\
                        use_fraction = True, bins=syst_bin, min_val=minval, max_val=maxval,\
                        save=save, savefolder=save_folder_name,\
                        variable=plot_name, units=plot_units, reco_name="Retro")

    plot_vertex(true_r[cuts],true_z[cuts])

    plot_bin_slices(true_energy[cuts], cnn_energy[cuts], old_reco = retro_energy[cuts],\
                    weights=weights[cuts],energy_truth=true_r[cuts],\
                    xvariable="Starting Vertex R Position",xunits="(m)",
                    use_fraction = True, bins=syst_bin, min_val=0, max_val=maxval*3,\
                    save=save, savefolder=save_folder_name,\
                    variable=plot_name, units=plot_units, reco_name="Retro")
   
"""
 
"""
plot_distributions(true_energy,cnn_energy,save=save,savefolder=save_folder_name,old_reco=retro_energy)
plot_distributions(true_energy,cnn_energy,save=save,savefolder=save_folder_name,old_reco=retro_energy,minval=1,maxval=150,bins=150)
plot_2D_prediction(true_energy, cnn_energy, save, save_folder_name,bins=bins,\
                        minval=minval, maxval=maxval, cut_truth=True, axis_square=True,\
                        variable=plot_name, units=plot_units, reco_name="CNN")
plot_2D_prediction(true_energy, retro_energy, save, save_folder_name,bins=bins,\
                        minval=minval, maxval=maxval, cut_truth=True, axis_square=True,\
                        variable=plot_name, units=plot_units, reco_name="Retro")
plot_single_resolution(true_energy, cnn_energy, use_old_reco = True, old_reco = retro_energy,\
                   minaxis=-maxval, maxaxis=maxval, bins=bins,\
                   save=save, savefolder=save_folder_name,\
                   variable=plot_name, units=plot_units, reco_name="Retro")
plot_bin_slices(true_energy, cnn_energy, old_reco = retro_energy,\
                    use_fraction = True, bins=15, min_val=minval, max_val=maxval,\
                    save=save, savefolder=save_folder_name,\
                    variable=plot_name, units=plot_units, reco_name="Retro")
"""
#Compare CC vs. NC
"""
save_folder_name = save_folder_base + "/CCNC/"
if os.path.isdir(save_folder_name) != True:
    os.mkdir(save_folder_name)
#Plot stuff
maskNC = true_CC == 0
maskCC = true_CC == 1
print(sum(maskCC),sum(maskNC),sum(maskCC)/len(cnn_energy), sum(maskNC)/len(cnn_energy))
plot_2D_prediction(true_energy[maskCC], cnn_energy[maskCC], save, save_folder_name,bins=bins,\
                        minval=minval, maxval=maxval, cut_truth=True, axis_square=True,\
                        variable=plot_name, units=plot_units, reco_name="CNN CC")
plot_2D_prediction(true_energy[maskNC], cnn_energy[maskNC], save, save_folder_name,bins=bins,\
                        minval=minval, maxval=maxval, cut_truth=True, axis_square=True,\
                        variable=plot_name, units=plot_units, reco_name="CNN NC")
plot_2D_prediction(true_energy[maskCC], retro_energy[maskCC], save, save_folder_name,bins=bins,\
                        minval=minval, maxval=maxval, cut_truth=True, axis_square=True,\
                        variable=plot_name, units=plot_units, reco_name="Retro CC")
plot_2D_prediction(true_energy[maskNC], retro_energy[maskNC], save, save_folder_name,bins=bins,\
                        minval=minval, maxval=maxval, cut_truth=True, axis_square=True,\
                        variable=plot_name, units=plot_units, reco_name="Retro NC")
plot_distributions(cnn_energy[maskCC],cnn_energy[maskNC],save=save,savefolder=save_folder_name,reco_name="CC")
plot_distributions(cnn_energy[maskCC],cnn_energy[maskNC],save=save,savefolder=save_folder_name,minval=1,maxval=150,bins=150)
plot_single_resolution(true_energy[maskCC], cnn_energy[maskCC], use_old_reco = True,\
                   old_reco = cnn_energy[maskNC], old_reco_truth=true_energy[maskNC],\
                   minaxis=-maxval, maxaxis=maxval, bins=bins,\
                   save=save, savefolder=save_folder_name,\
                   variable=plot_name, units=plot_units, reco_name="NC")
plot_bin_slices(true_energy[maskCC], cnn_energy[maskCC],
                    old_reco = cnn_energy[maskNC], old_reco_truth=true_energy[maskNC],\
                    use_fraction = True, bins=15, min_val=minval, max_val=maxval,\
                    save=save, savefolder=save_folder_name,\
                    variable=plot_name, units=plot_units, reco_name="NC")
"""
"""
print(max(true_energy),min(true_energy))
maskbig = true_energy > 200
print(sum(maskbig),len(true_energy))
masksmall = true_energy < 5
print(sum(masksmall),sum(masksmall)/len(true_energy))
plt.figure()
plt.hist(true_energy,range=[1,200],bins=100)
plt.title("True Energy Distribution")
plt.xlabel("Energy (GeV)")
plt.savefig(save_folder_name + "TrueEnergyDistribution.png",bbox_inches='tight')
"""

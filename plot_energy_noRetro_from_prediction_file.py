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
truth = f["Y_test_use"][:]
predict = f["Y_predicted"][:]
try: 
    weights = f["weights_test"][:]
except:
    weights = None
try:
    reco = f["reco_test"][:]
except:
    reco = None
try:
    info = f["additional_info"][:]
except: 
    info = None
f.close()
del f

cnn_energy = np.array(predict[:,0])
true_energy = np.array(truth[:,0])
true_x = np.array(truth[:,4])
true_y = np.array(truth[:,5])
true_z = np.array(truth[:,6])
true_CC = np.array(truth[:,11])
if weights is not None:
    weights = np.array(weights[:,8])
if reco is not None:
    retro_energy = np.array(reco[:,0])
    retro_zenith = np.array(reco[:,1])
    retro_time = np.array(reco[:,3])
    reco_x = np.array(reco[:,4])
    reco_y = np.array(reco[:,5])
    reco_z = np.array(reco[:,6])
    retro_coszen = np.cos(retro_zenith)
    prob_nu = info[:,1]


#Vertex Position
x_origin = np.ones((len(true_x)))*46.290000915527344
y_origin = np.ones((len(true_y)))*-34.880001068115234
true_r = np.sqrt( (true_x - x_origin)**2 + (true_y - y_origin)**2 )
if reco is not None:
    reco_r = np.sqrt( (reco_x - x_origin)**2 + (reco_y - y_origin)**2 )

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

print(min(true_energy),max(true_energy))

#plot_ZR(true_z,true_r,true_energy,weights=weights,cut_min=0,cut_max=5,bins=100,save_folder_name=save_folder_name)
#plot_ZR(true_z,true_r,true_energy,weights=weights,cut_min=10,cut_max=20,bins=100,save_folder_name=save_folder_name)
#plot_ZR(true_z,true_r,true_energy,weights=weights,cut_min=min(true_energy),cut_max=max(true_energy),bins=100,save_folder_name=save_folder_name)

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
maxabs_factor = 100.

maskNONE = true_energy > 0.
assert sum(maskNONE)==len(true_energy), "Some true energy < 0? Check!" 
maskCC = true_CC == 1
maskZ = np.logical_and(true_z > -505, true_z < 192)
maskR = true_r < 90.
maskDC = np.logical_and(maskZ,maskR)
maskE = np.logical_and(true_energy*maxabs_factor > 5., true_energy*maxabs_factor < 100.)
maskE2 = np.logical_and(true_energy > 1., true_energy < 200.)
maskCNNE = np.logical_and(cnn_energy > 5., cnn_energy < 100.)
#maskNu = prob_nu > 0.4
#maskMu = coin_muon > 0
#maskReco = np.logical_and(prob_nu,coin_muon)
if reco is not None:
    maskRecoZ = np.logical_and(reco_z > -500., reco_z < -200.)
    maskRecoR = reco_r < 300.
    maskRecoDC = np.logical_and(maskRecoZ, maskRecoR)
    maskRetroZenith = retro_zenith <= 0.3
    maskRetroEnergy = np.logical_and(retro_energy >= 5., retro_energy <= 300.)
    maskRetroTime = retro_time < 14500.
    maskRetro = np.logical_and(np.logical_and(maskRetroZenith, maskRetroEnergy), maskRetroTime)
    maskANA = np.logical_and(maskRecoDC,  maskRetro)
    assert sum(maskANA)!=len(maskANA), "No events after ANA mask"

#plot_vs_Energy(true_energy[maskCC],true_ndoms[maskCC],weights=weights[maskCC],bins=100,savefolder=save_folder_name)
#plot_vs_Energy(true_energy[maskCC],true_ndoms[maskCC],weights=weights[maskCC],xmax=300,bins=100,savefolder=save_folder_name)

cut_list = [maskNONE, maskCC, np.logical_and(maskE, maskCC), np.logical_and(maskCNNE, maskCC), np.logical_and(maskE2, maskCC)]
cut_names = ["NoCuts", "NoCuts_CC", "E5100_CC", "CNNE5100_CC", "E1200_CC"]
minvals = [1, 1, 5, 5, 1]
maxvals = [200, 200, 100, 100, 200]
binss = [199, 199, 95, 95, 199]
syst_bins = [20, 20, 10, 10, 20]
save_base_name = save_folder_name

for cut_index in range(1,len(cut_list)):
    cuts = cut_list[cut_index]
    folder_name = cut_names[cut_index]
    minval = minvals[cut_index]
    maxval = maxvals[cut_index]
    bins = binss[cut_index]
    syst_bin = syst_bins[cut_index]

    truth = true_energy[cuts]*maxabs_factor
    cnn = cnn_energy[cuts]*maxabs_factor
    if reco is None:
        old_reco = None
        use_old_reco_bool = False
    else:
        old_reco = retro_energy[cuts]
        use_old_reco_bool = True
    if weights is None:
        weights_plot = None
    else:
        weights_plot =  weights[cuts]

    print("Working on %s"%folder_name)

    save_folder_name = save_base_name + "/%s/"%folder_name
    if os.path.isdir(save_folder_name) != True:
        os.mkdir(save_folder_name)


    print(truth[:10], cnn[:10])
    #plot_NDOMS(true_ndoms[cuts],true_r[cuts])
    #plot_NDOMS(true_ndoms[cuts],true_r[cuts],cut=0)
    
    plot_distributions(truth, cnn, old_reco=old_reco,\
                                save=save, savefolder=save_folder_name, weights=weights_plot,\
                                reco_name = "Retro", variable=plot_name, units= plot_units,
                                minval=minval,maxval=maxval,bins=bins)
    plot_distributions(truth, cnn, old_reco=old_reco,\
                                save=save, savefolder=save_folder_name, weights=weights_plot,\
                                reco_name = "Retro", variable=plot_name, units= plot_units)
    if reco is not None:
        plot_distributions(true_r[cuts], reco_r[cuts],\
                                    save=save, savefolder=save_folder_name, weights=weights_plot,\
                                    cnn_name = "Retro", variable="Radial Vertex", units= "(m)",log=True)
        plot_distributions(true_z[cuts], reco_z[cuts],\
                                save=save, savefolder=save_folder_name, weights=weights_plot,\
                                cnn_name = "Retro", variable="Z Vertex", units= "(m)",log=True)
    
    plot_2D_prediction(truth, cnn,weights=weights_plot,\
                            save=save, savefolder=save_folder_name,bins=bins,
                            variable=plot_name, units=plot_units, reco_name="CNN")
    plot_2D_prediction(truth, cnn,weights=weights_plot,\
                            save=save, savefolder=save_folder_name,bins=bins,
                            minval=minval, maxval=maxval, cut_truth=True, axis_square=True,\
                            variable=plot_name, units=plot_units, reco_name="CNN")
    if reco is not None:
        plot_2D_prediction(truth, retro_energy[cuts], weights=weights_plot,
                                save=save, savefolder=save_folder_name,bins=bins,\
                                variable=plot_name, units=plot_units, reco_name="Retro")
        plot_2D_prediction(truth, retro_energy[cuts], weights=weights_plot,
                                save=save, savefolder=save_folder_name,bins=bins,\
                                minval=minval, maxval=maxval, cut_truth=True, axis_square=True,\
                                variable=plot_name, units=plot_units, reco_name="Retro")
    
    plot_single_resolution(truth, cnn, weights=weights_plot,\
                       use_old_reco = use_old_reco_bool, old_reco = old_reco,\
                       minaxis=-maxval, maxaxis=maxval, bins=bins,\
                       save=save, savefolder=save_folder_name,\
                       variable=plot_name, units=plot_units, reco_name="Retro")
    
    plot_bin_slices(truth, cnn, weights=weights_plot,  
                        old_reco = old_reco,\
                        use_fraction = True, bins=syst_bin, min_val=minval, max_val=maxval,\
                        save=save, savefolder=save_folder_name,\
                        variable=plot_name, units=plot_units, reco_name="Retro")

    plot_rms_slices(truth, cnn, weights=weights_plot,  
                        old_reco = old_reco,\
                        use_fraction = True, bins=syst_bin, min_val=minval, max_val=maxval,\
                        save=save, savefolder=save_folder_name,\
                        variable=plot_name, units=plot_units, reco_name="Retro")

    plot_vertex(true_r[cuts],true_z[cuts])

    plot_bin_slices(truth, cnn, old_reco = old_reco,\
                    weights=weights_plot,energy_truth=true_r[cuts],\
                    xvariable="Starting Vertex R Position",xunits="(m)",
                    use_fraction = True, bins=syst_bin, min_val=0, max_val=maxval*3,\
                    save=save, savefolder=save_folder_name,\
                    variable=plot_name, units=plot_units, reco_name="Retro")
   

 
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

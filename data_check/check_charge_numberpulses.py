import numpy as np
import h5py
import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import argparse
import glob
from handle_data import CutMask

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_file",default=None,
                    type=str,dest="input_files", help="names for input file")
parser.add_argument("-d", "--path",type=str,default='/mnt/scratch/micall12/training_files/',
                    dest="path", help="path to input files")
parser.add_argument("-o", "--outdir",type=str,default='/mnt/home/micall12/LowEnergyNeuralNetwork/output_plots/',
                    dest="outdir", help="out directory for plots")
parser.add_argument("-n", "--name",default=None,
                    dest="name", help="name for output folder")
parser.add_argument("--large_charge",type=float,default=40.,
                    dest="large_charge", help="Max charge to distinguish for statistics")
parser.add_argument("--large_numpulses",type=int,default=20,
                    dest="large_numpulses", help="Max number of pulses to distinguish for statistics")
parser.add_argument("--emax",type=float,default=100.0,
                    dest="emax",help="Cut anything greater than this energy (in GeV)")
parser.add_argument("--emin",type=float,default=5.0,
                    dest="emin",help="Cut anything less than this energy (in GeV)")
parser.add_argument("-c", "--cuts",type=str, default="CC",
                    dest="cuts", help="Type of events to keep (all, cascade, track, CC, NC, etc.)")
args = parser.parse_args()


input_file = args.path + args.input_files

output_path = args.outdir
name = args.name
outdir = output_path + name
if os.path.isdir(outdir) != True:
    os.mkdir(outdir)
print("Saving plots to %s"%outdir)
        

large_number_pulses = args.large_numpulses
large_charge = args.large_charge
energy_min = args.emin
energy_max = args.emax
cut_name = args.cuts
bins = int(energy_max - energy_min)

check_charge = True
check_numpulses = True
true_energy_plots = True

### Import Files ###
f = h5py.File(input_file, 'r')
labels = f['labels'][:]
features_DC = f['features_DC'][:]
features_IC = f['features_IC'][:]
stats = f['initial_stats'][:]
num_pulses = f['num_pulses_per_dom'][:]
try:
    trig_time = f['trigger_times'][:]
except:
    trig_time = None
f.close()
del f

# Apply Cuts
mask = CutMask(labels)
cut_energy = np.logical_and(labels[:,0] > energy_min, labels[:,0] < energy_max)
all_cuts = np.logical_and(mask[cut_name], cut_energy)
labels = labels[all_cuts]
stats = stats[all_cuts]
num_pulses = num_pulses[all_cuts]
trig_time = trig_time[all_cuts]
features_DC = features_DC[all_cuts]
features_IC = features_IC[all_cuts]

## WHAT EACH ARRAY CONTAINS! ##
# reco: (energy, zenith, azimuth, time, x, y, z) 
# stats: (count_outside, charge_outside, count_inside, charge_inside) 
# num_pulses: [ string num, dom index, num pulses]
# trig_time: [DC_trigger_time]

num_events = stats.shape[0]
print("Checking %i events"%num_events)

# Charge outside vs inside
if check_charge:
    count_outside = stats[:,0]
    charge_outside = stats[:,1]
    count_inside = stats[:,2]
    charge_inside = stats[:,3]
    fraction_count_inside = count_inside/(count_outside + count_inside)
    fraction_charge_inside = charge_inside/(charge_outside + charge_inside)
    mask_large_charge = charge_inside > large_charge
    fraction_large_charge = sum(charge_inside[mask_large_charge])/sum(charge_inside)
    print("Median of counts inside is %f with median total charge inside is %f, in the subset of chosen strings over all events"%(np.median(fraction_count_inside),np.median(fraction_charge_inside)))
    print("PERCENTAGE of charge that is greater than %i inside subset of strings over all events: %f percent"%(large_charge,fraction_large_charge*100))

    plt.figure()
    plt.title("Fraction of # pulses inside subset strings")
    plt.hist(fraction_count_inside,bins=50,alpha=0.5);
    plt.xlabel("# pulses inside subset strings / total # pulses in event")
    plt.savefig("%s/FractionPulsesInside.png"%outdir)
    plt.close()

    plt.figure()
    plt.title("Fraction of charge inside subset strings")
    plt.hist(fraction_charge_inside,bins=50,alpha=0.5);
    plt.xlabel("charge recorded inside subset strings / total charge recorded in event")
    plt.savefig("%s/FractionChargeInside.png"%outdir)
    plt.close()

# Number of pulses on all DOMS
if check_numpulses:
    num_pulses_all = num_pulses[:,:,:,0].flatten()
    large_mask = num_pulses_all > large_number_pulses
    large_num = sum(num_pulses_all[large_mask])
    fraction_large = large_num/len(num_pulses_all)
    print("PERCENTAGE of DOMS that see more pulses than %i over all events: %f percent"%(large_number_pulses,fraction_large*100))

    gt0 = num_pulses_all > 0
    plt.figure()
    plt.title("Number of pulses > 0 on ALL DOMS for ALL events")
    plt.hist(num_pulses_all[gt0],bins=50,alpha=0.5);
    plt.xlabel("# pulses per dom")
    plt.yscale('log')
    plt.savefig("%s/NumberPulsesAllDOMS.png"%outdir)
    plt.close()

    plt.figure()
    for i in range(0,10):
        num_pulses_one_evt = num_pulses[i,:,:,0].flatten()
        gt0 = num_pulses_one_evt > 0
        plt.hist(num_pulses_one_evt[gt0],bins=5,alpha=0.5);
    plt.title("Number of pulses > 0 on ALL DOMS per 10 events")
    plt.xlabel("# pulses per dom")
    plt.yscale('log')
    plt.savefig("%s/NumberPulsesAllDOMS_10Events.png"%outdir)
    plt.close()


# Stats vs True Energy
if true_energy_plots == True:
    charge_max = 380
    hits_max = 195
    energy = labels[:,0]
    num_pulses_sum = np.sum(np.sum(num_pulses[:,:,:,0],axis=1),axis=1)
    count_outside = stats[:,0]
    charge_outside = stats[:,1]
    count_inside = np.array(stats[:,2])
    charge_inside = np.array(stats[:,3])
    charge_total = np.array(charge_inside + charge_outside)
    count_total = np.array(count_inside + count_outside)

    #Charge 2
    charge2_DC = np.sum(np.sum(features_DC[:,:,:,0],axis=1),axis=1)
    charge2_IC = np.sum(np.sum(features_IC[:,:,:,0],axis=1),axis=1)
    charge2_inside = charge2_DC + charge2_IC

    def plot_hist2d(x_array,y_array,title,bins,x_name,y_name,plot_name,ymax=None):
        plt.figure()
        plt.title(title,fontsize=20)
        if ymax:
             plt.hist2d(x_array,y_array,bins=bins,cmap='viridis_r', range=[[energy_min,energy_max],[0,ymax]],cmin=1);
        else:
            plt.hist2d(x_array,y_array,bins=bins,cmap='viridis_r', cmin=1);
        plt.xlabel(x_name,fontsize=20)
        plt.ylabel(y_name,fontsize=20)
        cbar = plt.colorbar()
        plt.savefig("%s/%s.png"%(outdir,plot_name))
        plt.close()
    
    #Hits Stats
    plot_hist2d(energy,count_total,"Total Hits Per Event",bins,"True Energy (GeV)","Total Hits","TotalHits")
    plot_hist2d(energy,count_total,"Total Hits Per Event (Cut)",bins,"True Energy (GeV)","Total Hits","TotalHitsCut",ymax = hits_max)
    plot_hist2d(energy,count_inside,"Total Hits Per Event Inside Used Strings",bins,"True Energy (GeV)","Total Hits Inside Used Strings","TotalHitsInside") 
    plot_hist2d(energy,count_inside,"Total Hits Per Event Inside Used Strings (Cut)",bins,"True Energy (GeV)","Total Hits Inside Used Strings","TotalHitsInsideCut",ymax=hits_max) 
    
    # Charge Stats
    plot_hist2d(energy,charge_total,"Total Charge Per Event",bins,"True Energy (GeV)","Total Charge","TotalCharge")
    plot_hist2d(energy,charge_total,"Total Charge Per Event Cut",bins,"True Energy (GeV)","Total Charge","TotalChargeCut",ymax=charge_max)
    plot_hist2d(energy,charge_inside,"Total Charge Inside Used Strings Per Event",bins,"True Energy (GeV)","Total Charge Inside Used Strings","TotalChargeInside")
    plot_hist2d(energy,charge_inside,"Total Charge Inside Used Strings Per Event (Cut)",bins,"True Energy (GeV)","Total Charge Inside Used Strings","TotalChargeInsideCut",ymax=charge_max)

     # Charge2 Stats
    plot_hist2d(energy,charge2_inside,"Total Charge Input Per Event",bins,"True Energy (GeV)","Total Charge","TotalChargeInput")
    plot_hist2d(energy,charge2_inside,"Total Charge Input Per Event (Cut)",bins,"True Energy (GeV)","Total Charge","TotalChargeInputCut",ymax=charge_max)
    plot_hist2d(energy,charge2_DC,"Total Charge Input DC Per Event",bins,"True Energy (GeV)","Total Charge","TotalChargeInputDC")
    plot_hist2d(energy,charge2_DC,"Total Charge Input DC Per Event (Cut)",bins,"True Energy (GeV)","Total Charge","TotalChargeInputDCCut",ymax=charge_max)
    plot_hist2d(energy,charge2_IC,"Total Charge Input IC Per Event",bins,"True Energy (GeV)","Total Charge","TotalChargeInputIC")
    plot_hist2d(energy,charge2_IC,"Total Charge Input IC Per Event (Cut)",bins,"True Energy (GeV)","Total Charge","TotalChargeInputICCut",ymax=charge_max)

    # Find Average Per GeV Energy Bin
    count_per_bin = np.zeros((bins))
    charge_per_bin = np.zeros((bins))
    count_in_per_bin = np.zeros((bins))
    charge_in_per_bin = np.zeros((bins))
    charge2_DC_per_bin = np.zeros((bins))
    charge2_IC_per_bin = np.zeros((bins))
    charge2_per_bin = np.zeros((bins))
    energy_array = np.arange(energy_min,energy_max)
    amask = energy < 6.0
    for e in energy_array:
        emask = np.logical_and(energy >= e,energy < e+1)
        events_in_bin = sum(emask)
        if events_in_bin == 0:
            print("No events at %i GeV, skipping..."%e)
            continue
        index = int(e-energy_min)
        count_per_bin[index] = sum(count_total[emask])/events_in_bin
        count_in_per_bin[index] = sum(count_inside[emask])/events_in_bin
        charge_per_bin[index] = sum(charge_total[emask])/events_in_bin
        charge_in_per_bin[index] = sum(charge_inside[emask])/events_in_bin
        charge2_DC_per_bin[index] = sum(charge2_DC[emask])/events_in_bin
        charge2_IC_per_bin[index] = sum(charge2_IC[emask])/events_in_bin
        charge2_per_bin[index] = sum(charge2_inside[emask])/events_in_bin

    plt.figure()
    plt.title("Avg Hits Per GeV",fontsize=25)
    plt.bar(energy_array,count_per_bin,alpha=0.5,width=1,align='edge',label="All")
    plt.bar(energy_array,count_in_per_bin,alpha=0.5,width=1,align='edge',label="Used Strings")
    plt.xlabel("True Energy (GeV)",fontsize=20)
    plt.ylabel("Avg Hits",fontsize=20)
    plt.legend(fontsize=15)
    plt.savefig("%s/AvgHits.png"%outdir)
    plt.close()
    
    plt.figure()
    plt.title("Avg Charge Per GeV",fontsize=25)
    plt.bar(energy_array,charge_per_bin,alpha=0.5,width=1,align='edge',label="All")
    plt.bar(energy_array,charge_in_per_bin,alpha=0.5,width=1,align='edge',label="Used Strings")
    plt.xlabel("True Energy (GeV)",fontsize=20)
    plt.ylabel("Avg Charge",fontsize=20)
    plt.legend(fontsize=15)
    plt.savefig("%s/AvgCharge.png"%outdir)
    plt.close()

    plt.figure()
    plt.title("Avg Charge From Input Per GeV",fontsize=25)
    plt.bar(energy_array,charge2_per_bin,alpha=0.5,width=1,align='edge',label="DC + IC19")
    plt.bar(energy_array,charge2_DC_per_bin,alpha=0.5,width=1,align='edge',label="DC")
    plt.bar(energy_array,charge2_IC_per_bin,alpha=0.5,width=1,align='edge',label="IC19")
    plt.xlabel("True Energy (GeV)",fontsize=20)
    plt.ylabel("Avg Charge",fontsize=20)
    plt.legend(fontsize=15)
    plt.savefig("%s/AvgInputCharge.png"%outdir)
    plt.close()

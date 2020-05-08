import numpy
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

check_charge = True
check_numpulses = True

### Import Files ###
f = h5py.File(input_file, 'r')
labels = f['labels'][:]
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
cut_energy = numpy.logical_and(labels[:,0] > energy_min, labels[:,0] < energy_max)
all_cuts = numpy.logical_and(mask[cut_name], cut_energy)
labels = labels[all_cuts]
stats = stats[all_cuts]
num_pulses = num_pulses[all_cuts]
trig_time = trig_time[all_cuts]

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
    print("Median of counts inside is %f with median total charge inside is %f, in the subset of chosen strings over all events"%(numpy.median(fraction_count_inside),numpy.median(fraction_charge_inside)))
    print("PERCENTAGE of charge that is greater than %i inside subset of strings over all events: %f percent"%(large_charge,fraction_large_charge*100))

    plt.figure()
    plt.title("Fraction of # pulses inside subset strings")
    plt.hist(fraction_count_inside,bins=50,alpha=0.5);
    plt.xlabel("# pulses inside subset strings / total # pulses in event")
    plt.savefig("%s/FractionPulsesInside.png"%outdir)

    plt.figure()
    plt.title("Fraction of charge inside subset strings")
    plt.hist(fraction_charge_inside,bins=50,alpha=0.5);
    plt.xlabel("charge recorded inside subset strings / total charge recorded in event")
    plt.savefig("%s/FractionChargeInside.png"%outdir)

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

    plt.figure()
    for i in range(0,10):
        num_pulses_one_evt = num_pulses[i,:,:,0].flatten()
        gt0 = num_pulses_one_evt > 0
        plt.hist(num_pulses_one_evt[gt0],bins=5,alpha=0.5);
    plt.title("Number of pulses > 0 on ALL DOMS per 10 events")
    plt.xlabel("# pulses per dom")
    plt.yscale('log')
    plt.savefig("%s/NumberPulsesAllDOMS_10Events.png"%outdir)


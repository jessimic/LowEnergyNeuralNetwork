import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import glob
import os
from handle_data import CutMask

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_files",default=None,
                    type=str,dest="input_files", help="names for input files")
parser.add_argument("-d", "--path",type=str,default='/mnt/scratch/micall12/training_files/',
                    dest="path", help="path to input files")
parser.add_argument("-o", "--outdir",type=str,default='/mnt/home/micall12/LowEnergyNeuralNetwork/output_plots/',
                    dest="outdir", help="out directory for plots")
parser.add_argument("-n", "--name",type=str,default="None",
                    dest="name", help="name for output folder")
parser.add_argument("-b", "--bin_size",type=float,default=1.,
                    dest="bin_size", help="Size of energy bins in GeV (default = 1GeV)")
parser.add_argument("--emax",type=float,default=100.0,
                    dest="emax",help="Cut anything greater than this energy (in GeV)")
parser.add_argument("--emin",type=float,default=5.0,
                    dest="emin",help="Cut anything less than this energy (in GeV)")
parser.add_argument("-c", "--cuts",type=str, default="CC",
                    dest="cuts", help="Type of events to keep (all, cascade, track, CC, NC, etc.)")
args = parser.parse_args()

input_files = args.input_files
files_with_paths = args.path + input_files
event_file_names = sorted(glob.glob(files_with_paths))

emax = args.emax
emin = args.emin
bin_size = args.bin_size
cut_name = args.cuts
energy_bin_array = np.arange(emin,emax,bin_size)
    
if args.name is "None":
    file_name = event_file_names[0].split("/")
    output = file_name[-1][:-5]
else:
    output = args.name
outdir = args.outdir + output
if os.path.isdir(outdir) != True:
    os.mkdir(outdir)

# Find number of bins
bins = int((emax-emin)/float(bin_size))
if emax%bin_size !=0:
    bins +=1 #Put remainder into additional bin
count_energy = np.zeros((bins))

print("Evts in File\t After Type Cut\t After Energy Cut")
for a_file in event_file_names:
    
    ### Import Files ###
    f = h5py.File(a_file, 'r')
    file_labels = f['labels'][:]
    #file_labels = f['Y_test'][:]
    f.close()
    del f
    #file_labels[:,0] = file_labels[:,0]*emax

    if file_labels.shape[0] == 0:
        print("Empty file...skipping...")
        continue
    
    mask = CutMask(file_labels)

    # Make cuts for event type and energy
    energy = np.array(file_labels[:,0])
    total_events = len(energy)
    energy = energy[mask[cut_name]]
    events_after_type_cut = len(energy)
    energy_mask = np.logical_and(energy > emin, energy < emax)
    energy = energy[energy_mask]
    events_after_energy_cut = len(energy)
    print("%i\t %i\t %i\t"%(total_events,events_after_type_cut,events_after_energy_cut))

    #Sort into bins and count
    emin_array = np.ones((events_after_energy_cut))*emin
    energy_bins = np.floor((energy-emin_array)/float(bin_size))
    count_energy += np.bincount(energy_bins.astype(int),minlength=bins)
    
min_number_events = min(count_energy)
min_bin = np.where(count_energy==min_number_events)
if type(min_bin) is tuple:
    min_bin = min_bin[0][0]
print("Minimum bin value %i events at %i GeV"%(min_number_events,energy_bin_array[min_bin]))
print("Cutting there gives total events: %i"%(min_number_events*bins))
print(count_energy)

afile = open("%s/final_distribution_emin%.0femax%.0f_%s.txt"%(outdir,emin,emax,cut_name),"w")
afile.write("Minimum bin value %i events at %i GeV"%(min_number_events,energy_bin_array[min_bin]))
afile.write('\n')
afile.write("Cutting there gives total events: %i"%(min_number_events*bins))
afile.write('\n')
afile.write("Bin\t Energy\t Number Events\n")
for index in range(0,len(count_energy)):
    afile.write(str(index) + '\t' + str(int(energy_bin_array[index])) + '\t' + str(int(count_energy[index])) + '\n')
afile.close()

plt.figure(figsize=(10,8))
plt.title("Events Binned by %.1f GeV"%bin_size)
plt.bar(energy_bin_array,count_energy,alpha=0.5,width=1,align='edge')
plt.xlabel("energy (GeV)")
plt.ylabel("number of events")
plt.savefig("%s/EnergyDistribution_emin%.0femax%.0f_%s.png"%(outdir,emin,emax,cut_name))
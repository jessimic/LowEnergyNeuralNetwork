############################
# Counts number of events in i3 file
#   Takes only 1 file, first arg
#   Returns the number of events in file
###########################

import numpy
import sys
import argparse
import glob

from icecube import icetray, dataio, dataclasses
from I3Tray import I3Units

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
parser.add_argument("--emax",type=float,default=500.0,
                    dest="emax",help="Cut anything greater than this energy (in GeV)")
parser.add_argument("--emin",type=float,default=1.0,
                    dest="emin",help="Cut anything less than this energy (in GeV)")
args = parser.parse_args()

input_files = args.input_files
files_with_paths = args.path + input_files
event_file_names = sorted(glob.glob(files_with_paths))

emax = args.emax
emin = args.emin
efactor = args.efactor
tfactor = args.tfactor
bin_size = args.bin_size
cut_name = args.cuts

energy_bins = int((emax-emin)/float(bin_size))
count_energy = np.zeros((bins))

def sort_energy_per_file(event_file,current_count):
    event_file = dataio.I3File(event_file)
    energy = []
    for frame in event_file:
        if frame.Stop == icetray.I3Frame.Physics:
            header = frame["I3EventHeader"]

            if frame["I3EventHeader"].sub_event_stream != "InIceSplit":
                continue


            nu_energy = frame["I3MCTree"][0].energy
    
            energy.append(nu_energy)
        
            count_events +=1

    emin_array = np.ones((events_after_energy_cut))*emin
    energy_bins = np.floor((energy-emin_array)/float(bin_size))
    current_count += np.bincount(energy_bins.astype(int),minlength=bins)
        
    return count_energy

for a_file in event_file_names:
   
    count_energy = sort_energy_per_file(a_file,count_energy)

print(count_energy)

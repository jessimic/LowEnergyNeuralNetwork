############################
# Counts number of events in i3 file
#   Takes only 1 file, first arg
#   Returns the number of events in file
###########################

import numpy
import sys

from icecube import icetray, dataio, dataclasses
from I3Tray import I3Units

event_file_name = sys.argv[1]
emax = 200
emin = 5
energy_bins = 195
print("reading file: {}".format(event_file_name))
event_file = dataio.I3File(event_file_name)
count_events = 0
for event_file_name in filename_list:
    energy = []
    while event_file.more():
        try:
            frame = event_file.pop_physics()
        except:
            continue
        if frame["I3EventHeader"].sub_event_stream != "InIceSplit":
            continue

        try:
            cleaned = frame["SRTTWOfflinePulsesDC"]
        except:
            continue

        nu_energy = frame["I3MCTree"][0].energy
        if nu_energy > emax:
            continue
    
        energy.append(nu_energy)

    #emin_array = np.ones((events_after_energy_cut))*emin
    energy_bins = #np.floor((energy-emin_array)/float(bin_size))
    count_energy += np.bincount(energy_bins.astype(int),minlength=bins)
        
        count_events +=1

print("Total number of events: %i"%count_events)

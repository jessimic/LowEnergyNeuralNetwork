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

print("reading file: {}".format(event_file_name))
event_file = dataio.I3File(event_file_name)
count_events = 0

while event_file.more():
    try:
        frame = event_file.pop_physics()
    except:
        continue
    if frame["I3EventHeader"].sub_event_stream != "InIceSplit":
        continue

    count_events +=1

print("Total number of events: %i"%count_events)

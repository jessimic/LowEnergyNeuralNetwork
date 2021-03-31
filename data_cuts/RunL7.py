#!/usr/bin/python
##Test oscNext_L6_atm_muon_variables STV and TH##
 
from __future__ import division
import math
import os

import icecube
from I3Tray import *
from operator import itemgetter

from icecube import lilliput
import icecube.lilliput.segments
import oscNext_L7_nocuts

from icecube import icetray, phys_services, dataclasses, dataio, photonics_service, TrackHits, StartingTrackVetoLE

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i","--index",type=int,
                    dest="index",help="Index for file")
parser.add_argument("-s","--syst",type=int,default=149999,
                    dest="syst",help="systematic set including PID")
parser.add_argument("-d","--dir",default=None,
                    dest="directory",help="directory for folder in scratch")
args = parser.parse_args()

index = args.index
syst = args.syst
if args.directory is None:
    inpath='/mnt/research/IceCube/jmicallef/official_oscnext/level6/%s/'%syst
else:
    inpath=args.directory + "/"

full_index = "%06d" % (index,)
infilename='oscNext_genie_level6_v02.00_pass2.%s.%s.i3.zst'%(syst,full_index)
outpath=inpath
outfilename='oscNext_genie_level6.5_v02.00_pass2.%s.%s'%(syst,full_index)

file_list = []
name_f = outpath+outfilename
data_file = inpath+infilename
if os.path.isfile(data_file):
    print("Working on %s"%data_file)
else:
    print("Cannot find %s"%data_file)
gcd_file = "/mnt/research/IceCube/gcd_file/GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz"
file_list.append(gcd_file)
# for filename in glob.glob(data_file):
file_list.append(data_file)
    

#@icetray.traysegment
tray = I3Tray()
tray.AddModule("I3Reader","reader", FilenameList = file_list)
tray.AddSegment(oscNext_L7_nocuts.oscNext_L7,"L7nocuts",
                uncleaned_pulses="SplitInIcePulses",
                cleaned_pulses="SRTTWOfflinePulsesDC")
tray.AddModule('I3Writer', 'writer', Filename= name_f+'.i3.bz2')
tray.AddModule('TrashCan','thecan')
tray.Execute()
tray.Finish()


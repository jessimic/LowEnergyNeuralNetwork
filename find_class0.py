import numpy as np
from icecube import icetray, dataio, dataclasses, simclasses, recclasses
from I3Tray import I3Units
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--inputs",type=str,default=None,
                    dest="input_files", help="paths + names of the input files")
parser.add_argument("-n", "--outname",default=None,
                    dest="outname", help="name of output file")
parser.add_argument("-o", "--outdir",type=str,default='/mnt/home/micall12/LowEnergyNeuralNetwork/',
                    dest="output_dir", help="path of ouput file")
parser.add_argument("--variable",type=str,default="class",
                    dest="variable", help="name of variable that was predicted")
args = parser.parse_args()

input_files = args.input_files
output_dir=args.output_dir
variable = args.variable
if args.outname is None:
    output_name = "class0_events_%s"%variable #input_file.split("/")[-1]
else:
    output_name = args.outname
    
event_file_names = sorted(glob.glob(input_files))
assert event_file_names,"No files loaded, please check path."

outfile = dataio.I3File(output_dir+output_name+".i3",'w')

for count, event_file_name in enumerate(event_file_names):
    event_file = dataio.I3File(event_file_name)

    for frame in event_file:
        if frame.Stop == icetray.I3Frame.Physics:

            cnn_prediction = frame['FLERCNN_%s'%variable].value
            if cnn_prediction < 0.1:
                outfile.push(frame)

        else:
            outfile.push(frame)


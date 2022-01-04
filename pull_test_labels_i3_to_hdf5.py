import numpy as np
import h5py
from icecube import icetray, dataio, dataclasses, simclasses, recclasses, sim_services
from I3Tray import I3Units
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input",type=str,default=None,
                    dest="input_file", help="path and name of the input file")
parser.add_argument("-n", "--outname",default=None,
                    dest="outname", help="name of output file")
parser.add_argument("-o", "--outdir",type=str,default='/mnt/home/micall12/LowEnergyNeuralNetwork/output_plots/',
                    dest="output_dir", help="path of ouput file")
parser.add_argument("--variable_list",nargs='+',default=[],
                    dest="variable_list", help="names of variables that were predicted")
parser.add_argument("--simtype",type=str,default="genie",
                    dest="simtype", help="name of sim type, only muongun is set to do anything special at the moment")
parser.add_argument("--save_inputs", default=False,action='store_true',
                    dest="save_inputs", help="saving input features of the cnn")
args = parser.parse_args()

input_file = args.input_file
save_folder_name=args.output_dir
sim_type = args.simtype
save_cnn_input = args.save_inputs

variable_list = args.variable_list

if args.outname is None:
    output_name = "prediction_values_%ivar"%len(variable_list) #input_file.split("/")[-1]
else:
    output_name = args.outname
outdir = args.output_dir

from read_cnn_i3_files import read_i3_files

event_file_names = sorted(glob.glob(input_file))
assert event_file_names,"No files loaded, please check path."
output_cnn, output_labels, output_reco_labels, output_info, output_weights, input_features_DC, input_features_IC= read_i3_files(event_file_names,variable_list,save_cnn_input=save_cnn_input,sim_type=sim_type)

print(output_info.shape)

f = h5py.File("%s/%s.hdf5"%(outdir,output_name), "w")
f.create_dataset("Y_predicted", data=output_cnn)
f.create_dataset("Y_test_use", data=output_labels)
f.create_dataset("reco_test", data=output_reco_labels)
f.create_dataset("additional_info", data=output_info)
f.create_dataset("weights_test", data=output_weights)
if save_cnn_input:
    f.create_dataset("features_DC", data=input_features_DC)
    f.create_dataset("features_IC", data=input_features_IC)
f.close()

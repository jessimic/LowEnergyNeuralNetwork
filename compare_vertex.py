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

counter = 0
SANTA_zero = 0
SANTA_fit = []
Finite_vertex = []
LEERA_vertex = []
True_vertex = []
for count, event_file_name in enumerate(event_file_names):
    event_file = dataio.I3File(event_file_name)

    for frame in event_file:
        if frame.Stop == icetray.I3Frame.Physics:
            #True
            tree = frame["I3MCTree"][0]
            nu_x = nu.pos.x
            nu_y = nu.pos.y
            nu_z = nu.pos.z

            #Finite REco
            Finite_x = frame['FiniteRecoFit'].pos.x
            Finite_y = frame['FiniteRecoFit'].pos.y
            Finite_z = frame['FiniteRecoFit'].pos.z

            # Santa/LEERA fit
            fit_type = frame['L6_SANTA_FitType'].value
            Cascade_EM_x = np.nan 
            Cascade_EM_y = np.nan
            Cascade_EM_z = np.nan
            Cascade_Had_x =np.nan
            Cascade_Had_y =np.nan
            Cascade_Had_z =np.nan
            Muon_x = np.nan
            Muon_y = np.nan
            Muon_z = np.nan
            SANTA_x = np.nan
            SANTA_y = np.nan
            SANTA_z = np.nan

            if fit_type == 0:
                SANTA_zero += 1
            elif fit_type == 1 or fit_type == 2:
                SANTA_x = frame['L6_SANTA_sel_Particle'].pos.x
                SANTA_y = frame['L6_SANTA_sel_Particle'].pos.y
                SANTA_z = frame['L6_SANTA_sel_Particle'].pos.z
                Cascade_EM_x = frame['L7_SANTA_sel_LEERAFit_CascadeEM'].pos.x
                Cascade_EM_y = frame['L7_SANTA_sel_LEERAFit_CascadeEM'].y
                Cascade_EM_z = frame['L7_SANTA_sel_LEERAFit_CascadeEM'].z
                Cascade_Had_x = frame['L7_SANTA_sel_LEERAFit_CascadeHad'].x
                Cascade_Had_y = frame['L7_SANTA_sel_LEERAFit_CascadeHad'].y
                Cascade_Had_z = frame['L7_SANTA_sel_LEERAFit_CascadeHad'].z
                Muon_x = frame['L7_SANTA_sel_LEERAFit_Muon'].x
                Muon_y = frame['L7_SANTA_sel_LEERAFit_Muon'].y
                Muon_z = frame['L7_SANTA_sel_LEERAFit_Muon'].z
                assert(Cascade_EM_x == Cascade_Had_x, "Casc EM != Casc Had x") 
                assert(Cascade_EM_y == Cascade_Had_y, "Casc EM != Casc Had y") 
                assert(Cascade_EM_z == Cascade_Had_z, "Casc EM != Casc Had z") 
                assert(Cascade_EM_x == Muon_x, "Casc EM != Muon x") 
                assert(Cascade_EM_y == Muon_y, "Casc EM != Muon y") 
                assert(Cascade_EM_z == Muon_z, "Casc EM != Muon z") 
           else:
                print("SANTA FIT TYPE IS NOT 0, 1, or 2?????")

            SANTA_fit.append( int(fit_type)  )
            FINITE_vertex.append( np.array( [float(SANTA_x), float(SANTA_y), float(SANTA_z) ]) )
            LEERA_vertex.append( np.array( [float(Cascade_EM_x), float(Cascade_EM_y), float(Cascade_EM_z) ]) )

            counter +=1

print("Saved %i total events, %i which have SANTA fittype 0"%(counter, SANTA_zero))

f = h5py.File("%s/%s.hdf5"%(outdir,output_name), "w")
f.create_dataset("SANTA_fit", data=SANTA_fit)
f.create_dataset("Finite_vertex", data=Finite_vertex)
f.create_dataset("LEERA_vertex", data=LEERA_vertex)
f.create_dataset("True_vertex", data=True_vertex)
f.close()

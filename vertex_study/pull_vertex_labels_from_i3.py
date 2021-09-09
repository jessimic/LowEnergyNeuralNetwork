import numpy as np
from icecube import icetray, dataio, dataclasses, simclasses, recclasses
from I3Tray import I3Units
import argparse
import glob
import h5py

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--inputs",type=str,default=None,
                    dest="input_files", help="paths + names of the input files")
parser.add_argument("-n", "--outname",default=None,
                    dest="outname", help="name of output file")
parser.add_argument("-o", "--outdir",type=str,default='/mnt/home/micall12/LowEnergyNeuralNetwork/',
                    dest="output_dir", help="path of ouput file")
args = parser.parse_args()

input_files = args.input_files
output_dir=args.output_dir
if args.outname is None:
    output_name = "vertex_recos"
else:
    output_name = args.outname
    
event_file_names = sorted(glob.glob(input_files))
assert event_file_names,"No files loaded, please check path."
do_LEERA = False
do_retro = True

counter = 0
SANTA_zero = 0
SANTA_fit = []
Finite_vertex = []
SANTA_vertex = []
if do_LEERA:
    LEERA_EM_vertex = []
    LEERA_Had_vertex = []
    LEERA_Mu_vertex = []
L3 = []
HLC = []
CorridorWide = []
Corridor = []
Monopod = []
MPE = []
if do_retro:
    Retro_fit = []
    Retro = []
True_vertex = []
weights = []
print("Using %i files"%len(event_file_names))
for count, event_file_name in enumerate(event_file_names):
    event_file = dataio.I3File(event_file_name)

    for frame in event_file:
        if frame.Stop == icetray.I3Frame.Physics:
            #True
            nu = frame["I3MCTree"][0]
            nu_x = nu.pos.x
            nu_y = nu.pos.y
            nu_z = nu.pos.z

            weight = frame['I3MCWeightDict']["weight"]/len(event_file_names)

            L3_vertex = frame['IC2018_LE_L3_Vars']
            HLC_vertex = frame['L4_first_hlc'].pos
            corridor_wide_vertex = frame['L5_WideCorridorCutTrack'].pos
            #corridor_vertex = frame['L6_CorridorCutTrack'].pos
            monopod_vertex = frame['MonopodFit'].pos
            #MPE_vertex = frame['MPEFitMuEX'].pos

            if do_retro:
                retro_fit = ( "retro_crs_prefit__fit_status" in frame ) and frame["retro_crs_prefit__fit_status"] == 0
                if retro_fit:
                    retro_x = frame['L7_reconstructed_vertex_x'].value
                    retro_y = frame['L7_reconstructed_vertex_y'].value
                    retro_z = frame['L7_reconstructed_vertex_z'].value
                else:
                    retro_x = np.nan
                    retro_y = np.nan
                    retro_z = np.nan

            #Finite REco
            Finite_x = frame['FiniteRecoFit'].pos.x
            Finite_y = frame['FiniteRecoFit'].pos.y
            Finite_z = frame['FiniteRecoFit'].pos.z

            # Santa/LEERA fit
            fit_type = frame['L6_SANTA_FitType'].value
            if do_LEERA:
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
            elif (fit_type == 1 or fit_type == 2):
                SANTA_x = frame['L6_SANTA_sel_Particle'].pos.x
                SANTA_y = frame['L6_SANTA_sel_Particle'].pos.y
                SANTA_z = frame['L6_SANTA_sel_Particle'].pos.z
                if do_LEERA:
                    Cascade_EM_x = frame['L7_SANTA_sel_LEERAFit_CascadeEM'].pos.x
                    Cascade_EM_y = frame['L7_SANTA_sel_LEERAFit_CascadeEM'].pos.y
                    Cascade_EM_z = frame['L7_SANTA_sel_LEERAFit_CascadeEM'].pos.z
                    Cascade_Had_x = frame['L7_SANTA_sel_LEERAFit_CascadeHad'].pos.x
                    Cascade_Had_y = frame['L7_SANTA_sel_LEERAFit_CascadeHad'].pos.y
                    Cascade_Had_z = frame['L7_SANTA_sel_LEERAFit_CascadeHad'].pos.z
                    Muon_x = frame['L7_SANTA_sel_LEERAFit_Muon'].pos.x
                    Muon_y = frame['L7_SANTA_sel_LEERAFit_Muon'].pos.y
                    Muon_z = frame['L7_SANTA_sel_LEERAFit_Muon'].pos.z
            else:
                print("SANTA FIT TYPE IS NOT 0, 1, or 2?????")

            L3.append( np.array( [float(L3_vertex["VertexGuessX"]), float(L3_vertex["VertexGuessY"]), float(L3_vertex["VertexGuessZ"]) ]))
            #MPE.append( np.array( [float(MPE_vertex.x), float(MPE_vertex.y), float(MPE_vertex.z) ]))
            HLC.append( np.array( [float(HLC_vertex.x), float(HLC_vertex.y), float(HLC_vertex.z) ]))
            CorridorWide.append( np.array( [float(corridor_wide_vertex.x), float(corridor_wide_vertex.y), float(corridor_wide_vertex.z) ]))
            #Corridor.append( np.array( [float(corridor_vertex.x), float(corridor_vertex.y), float(corridor_vertex.z) ]))
            Monopod.append( np.array( [float(monopod_vertex.x), float(monopod_vertex.y), float(monopod_vertex.z) ]))
            weights.append( float(weight) ) 
            SANTA_fit.append( int(fit_type)  )
            SANTA_vertex.append( np.array( [float(SANTA_x), float(SANTA_y), float(SANTA_z) ]) )
            Finite_vertex.append( np.array( [float(Finite_x), float(Finite_y), float(Finite_z) ]) )
            if do_LEERA:
                LEERA_EM_vertex.append( np.array( [float(Cascade_EM_x), float(Cascade_EM_y), float(Cascade_EM_z) ]) )
                LEERA_Had_vertex.append( np.array( [float(Cascade_Had_x), float(Cascade_Had_y), float(Cascade_Had_z) ]) )
                LEERA_Mu_vertex.append( np.array( [float(Muon_x), float(Muon_y), float(Muon_z) ]) )
            if do_retro:
                Retro_fit.append(retro_fit)
                Retro.append( np.array( [float(retro_x), float(retro_y), float(retro_z)]) )
            True_vertex.append( np.array( [float(nu_x), float(nu_y), float(nu_z) ]) )

            counter +=1

print("Saved %i total events, %i which have SANTA fittype 0"%(counter, SANTA_zero))

f = h5py.File("%s/%s.hdf5"%(output_dir,output_name), "w")
f.create_dataset("SANTA_fit", data=SANTA_fit)
f.create_dataset("SANTA_vertex", data=SANTA_vertex)
f.create_dataset("Finite_vertex", data=Finite_vertex)
if do_LEERA:
    f.create_dataset("LEERA_EM_vertex", data=LEERA_EM_vertex)
    f.create_dataset("LEERA_Had_vertex", data=LEERA_Had_vertex)
    f.create_dataset("LEERA_Mu_vertex", data=LEERA_Mu_vertex)
if do_retro:
    f.create_dataset("Retro_vertex", data=Retro)
    f.create_dataset("Retro_fit", data=Retro_fit)
f.create_dataset("True_vertex", data=True_vertex)
f.create_dataset("L3_vertex", data=L3)
#f.create_dataset("MPE_vertex", data=MPE)
f.create_dataset("HLC_vertex", data=HLC)
#f.create_dataset("Corridor_vertex", data=Corridor)
f.create_dataset("CorridorWide_vertex", data=CorridorWide)
f.create_dataset("Monopod_vertex", data=Monopod)
f.create_dataset("weights", data=weights)
f.close()

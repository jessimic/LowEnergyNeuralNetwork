import numpy as np
import h5py
from icecube import icetray, dataio, dataclasses
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
args = parser.parse_args()

input_file = args.input_file
save_folder_name=args.output_dir
if args.outname is None:
    output_name = input_file.split("/")[-1]
else:
    output_name = args.outname
outdir = args.output_dir

def read_i3_files(filenames_list):
    cnn_energy = []
    true_energy = []
    retro_energy = []
    retro_zenith = []
    true_CC = []
    true_x = []
    true_y = []
    true_z = []
    fit_success = []
    weight = []
    coin_muon = []
    prob_nu = []
    reco_z = []
    reco_x = []
    reco_y = []
    reco_r = []
    true_ndoms = []

    max_files = len(filenames_list)
    if max_files > 10:
        ten_percent = max_files/10

    for count, event_file_name in enumerate(filenames_list):
        event_file = dataio.I3File(event_file_name)

        for frame in event_file:
            if frame.Stop == icetray.I3Frame.Physics:

                cnn_energy.append(frame['FLERCNN'].value)
                nu = frame["I3MCTree"][0]
                true_energy.append(nu.energy)
                true_x.append(nu.pos.x)
                true_y.append(nu.pos.y)
                true_z.append(nu.pos.z)
                if frame['I3MCWeightDict']['InteractionType']==1:
                    isCC = 1
                else:
                    isCC = 0
                true_CC.append(isCC)
                weight.append(frame['I3MCWeightDict']["weight"])
                fit_success.append(( "retro_crs_prefit__fit_status" in frame ) and frame["retro_crs_prefit__fit_status"] == 0)
                try:
                    retro_energy.append(frame['L7_reconstructed_total_energy'].value)
                    retro_zenith.append(frame['L7_reconstructed_zenith'].value)
                except:
                    retro_energy.append(np.nan)
                    rretro_zenith.append(np.nan)
                coin_muon.append(frame['L7_CoincidentMuon_bool'].value > 0)
                prob_nu.append(frame['L7_MuonClassifier_ProbNu'].value)
                reco_z.append(frame['L7_reconstructed_vertex_z'].value)
                reco_x.append(frame['L7_reconstructed_vertex_x'].value)
                reco_y.append(frame['L7_reconstructed_vertex_y'].value)
                reco_r.append(frame['L7_reconstructed_vertex_rho36'].value)

                #ice_pulses = dataclasses.I3RecoPulseSeriesMap.from_frame(frame,'SplitInIcePulses')
                #save_om = []
                #count = 0
                #for omkey,pulselist in ice_pulses:
                #    if omkey.om not in save_om:
                #        save_om.append(omkey.om)
                #        count +=1
                true_ndoms.append(frame['IC2018_LE_L3_Vars']['NchCleaned'])


        count +=1
        if (max_files > 10) and (count%ten_percent == 0):
            print("Progress Percent: %i"%(count/max_files*100))

    return cnn_energy, true_energy, retro_energy, true_x, true_y, true_z, true_CC, fit_success, weight, coin_muon, prob_nu, reco_x, reco_y, reco_z, reco_r, true_ndoms, retro_zenith

event_file_names = sorted(glob.glob(input_file))
assert event_file_names,"No files loaded, please check path."
cnn_energy, true_energy, retro_energy, true_x, true_y, true_z, true_CC, fit_success, weight, coin_muon, prob_nu, reco_x, reco_y, reco_z, reco_r, true_ndoms, retro_zenith = read_i3_files(event_file_names)
cnn_energy = np.array(cnn_energy)
true_energy = np.array(true_energy)
retro_energy = np.array(retro_energy)
retro_zenith = np.array(retro_zenith)
true_x = np.array(true_x)
true_y = np.array(true_y)
true_z = np.array(true_z)
true_CC = np.array(true_CC)
fit_success = np.array(fit_success)
weight = np.array(weight)
coin_muon = np.array(coin_muon)
prob_nu = np.array(prob_nu)
reco_x = np.array(reco_x)
reco_y = np.array(reco_y)
reco_z = np.array(reco_z)
reco_r = np.array(reco_r)
true_ndoms = np.array(true_ndoms)

output_path = outdir + output_name + ".hdf5"
f = h5py.File(output_path, "w")
f.create_dataset("cnn_energy", data=cnn_energy)
f.create_dataset("true_energy", data=true_energy)
f.create_dataset("retro_energy", data=retro_energy)
f.create_dataset("retro_zenith", data=retro_zenith)
f.create_dataset("true_x", data=true_x)
f.create_dataset("true_y", data=true_y)
f.create_dataset("true_z", data=true_z)
f.create_dataset("true_CC", data=true_CC)
f.create_dataset("fit_success", data=fit_success)
f.create_dataset("weight",data=weight)
f.create_dataset("coin_muon",data=coin_muon)
f.create_dataset("prob_nu",data=prob_nu)
f.create_dataset("reco_x",data=reco_x)
f.create_dataset("reco_y",data=reco_y)
f.create_dataset("reco_z",data=reco_z)
f.create_dataset("reco_r",data=reco_r)
f.create_dataset("true_ndoms",data=true_ndoms)
f.close()

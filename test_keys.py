#############################
# Read IceCube files and create training file with labels only (hdf5)
#   read_files = read in files and add truth labels
#   Can take 1 or multiple files
#   Input:
#       -i input: name of input file, include path
#       -n name: name for output file, automatically puts in my scratch
#       --emax: maximum energy saved (60 is default, so keep all < 60 GeV)
#       --true_name: need name of key to check if I3MCTree primary particle matches truth
##############################

import numpy
import h5py
import argparse

from icecube import icetray, dataio, dataclasses
from I3Tray import I3Units

from collections import OrderedDict
import itertools
import random

## Create ability to change settings from terminal ##
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input",type=str,default='Level5_IC86.2013_genie_numu.014640.00000?.i3.bz2',
                    dest="input_file", help="path and name of the input file")
parser.add_argument("-n", "--name",type=str,default='Level5_IC86.2013_genie_numu.014640.00000X',
                    dest="output_name",help="name for output file (no path)")
parser.add_argument("--emax",type=float,default=60.0,
                    dest="emax",help="Max energy to keep, cut anything above")
parser.add_argument("--vertex",type=str, default="DC",
                    dest="vertex_name",help="Name of vertex cut to put on file")
parser.add_argument("--true_name",type=str,default="MCInIcePrimary",
                    dest="true_name", help="Name of key for true particle information")
args = parser.parse_args()
input_file = args.input_file
output_name = args.output_name
emax = args.emax
vertex_name = args.vertex_name
true_name = args.true_name

def read_files(filename_list):
    """
    Read list of files, make sure they pass L5 cuts, create truth labels
    Receives:
        filename_list = list of strings, filenames to read data from
    Returns:
        output_labels = dict with output labels  (energy, zenith, azimith, time, x, y, z, 
                        tracklength, isTrack, flavor ID, isAntiNeutrino, isCC)
    Throws errors if I3MCTree[0] is not the same as the trueNeutrino
    """
    output_features_DC = []
    output_features_IC = []
    output_labels = []
    output_reco_labels = []
    output_initial_stats = []
    output_num_pulses_per_dom = []
    output_trigger_times = []
    not_in_DC = 0
    isOther_count = 0

    for event_file_name in filename_list:
        print("reading file: {}".format(event_file_name))
        event_file = dataio.I3File(event_file_name)

        while event_file.more():
            try:
                frame = event_file.pop_physics()
            except:
                continue
            
            if frame["I3EventHeader"].sub_event_stream != "InIceSplit":
                continue

            # ALWAYS USE EVENTS THAT PASSES CLEANING!
            try:
                cleaned = frame["SRTTWOfflinePulsesDC"]
            except:
                continue

            # some truth labels (we do *not* have these in real data and would like to figure out what they are)
            nu_tree = frame["I3MCTree"][0]
            nu_true = frame[true_name]

            if (nu_tree.type != dataclasses.I3Particle.NuMu and nu_tree.type != dataclasses.I3Particle.NuMuBar\
                and nu_tree.type != dataclasses.I3Particle.NuE and nu_tree.type != dataclasses.I3Particle.NuEBar\
                and nu_tree.type != dataclasses.I3Particle.NuTau and nu_tree.type != dataclasses.I3Particle.NuTauBar):
                print("PARTICLE IS NOT NUMU!!!!!!!!!!!!!!!!!!")

            nu_x_tree = nu_tree.pos.x
            nu_y_tree = nu_tree.pos.y
            nu_z_tree = nu_tree.pos.z
            nu_zenith_tree = nu_tree.dir.zenith
            nu_azimuth_tree = nu_tree.dir.azimuth
            nu_energy_tree = nu_tree.energy
            nu_time_tree = nu_tree.time


            nu_x_true = nu_true.pos.x
            nu_y_true = nu_true.pos.y
            nu_z_true = nu_true.pos.z
            nu_zenith_true = nu_true.dir.zenith
            nu_azimuth_true = nu_true.dir.azimuth
            nu_energy_true = nu_true.energy
            nu_time_true = nu_true.time

             
            assert nu_x_true==nu_x_tree,"Check nu_x"
            assert nu_y_true==nu_y_tree,"Check nu_y"
            assert nu_z_true==nu_z_tree,"Check nu_z"
            assert nu_time_true==nu_time_tree,"Check nu_time"
            assert nu_zenith_true==nu_zenith_tree,"Check nu_zenith"
            assert nu_azimuth_true==nu_azimuth_tree,"Check nu_azimuth"
            assert nu_energy_true==nu_energy_tree,"Check nu_energy"
            assert nu_tree==nu_true,"CHECK I3PARTICLE"
            

            #track_length = frame["trueMuon"].length
            isTrack = frame['I3MCWeightDict']['InteractionType']==1.   # it is a cascade with a trac
            isCascade = frame['I3MCWeightDict']['InteractionType']==2. # it is just a cascade
            isCC = frame['I3MCWeightDict']['InteractionType']==1.
            isNC = frame['I3MCWeightDict']['InteractionType']==2.
            isOther = not isCC and not isNC
            

            # input file sanity check: this should not print anything since "isOther" should always be false
            if isOther:
                print("isOTHER - not Track or Cascade...skipping")
                isOther_count += 1
                continue

                #print(frame['I3MCWeightDict'])
            
            # set track classification for numu CC only
            if ((nu_tree.type == dataclasses.I3Particle.NuMu or nu_tree.type == dataclasses.I3Particle.NuMuBar) and isCC):
                isTrack = True
                isCascade = False
                if frame["I3MCTree"][1].type == dataclasses.I3Particle.MuMinus or frame["I3MCTree"][1].type == dataclasses.I3Particle.MuPlus:
                    track_length = frame["I3MCTree"][1].length
                else:
                    print("Second particle not Muon, continuing")
                    continue
            else:
                isTrack = False
                isCascade = True
                track_length = 0
        
            #Save flavor and particle type (anti or not)
            if (nu_tree.type == dataclasses.I3Particle.NuMu):
                neutrino_type = 14
                particle_type = 0 #particle
            elif (nu_tree.type == dataclasses.I3Particle.NuMuBar):
                neutrino_type = 14
                particle_type = 1 #antiparticle
            elif (nu_tree.type == dataclasses.I3Particle.NuE):
                neutrino_type = 12
                particle_type = 0 #particle
            elif (nu_tree.type == dataclasses.I3Particle.NuEBar):
                neutrino_type = 12
                particle_type = 1 #antiparticle
            elif (nu_tree.type == dataclasses.I3Particle.NuTau):
                neutrino_type = 16
                particle_type = 0 #particle
            elif (nu_tree.type == dataclasses.I3Particle.NuTauBar):
                neutrino_type = 16
                particle_type = 1 #antiparticle
            else:
                print("Do not know first particle type in MCTree, should be neutrino, skipping this event")
                continue

            # Only look at "low energy" events for now
            if nu_energy_tree > emax:
                continue
            
            # Cut to only use events with true vertex in DeepCore
            if vertex_name == "IC19":
                #print("Using IC19 radius for cuts")
                radius = 300
            if vertex_name == "DC":
                #print("Using DC only for cuts")
                radius = 90
            x_origin = 54
            y_origin = -36
            shift_x = nu_x_tree - x_origin
            shift_y = nu_y_tree - y_origin
            z_val = nu_z_tree
            radius_calculation = numpy.sqrt(shift_x**2+shift_y**2)
            if( radius_calculation > radius or z_val > 192 or z_val < -505 ):
                not_in_DC += 1
                continue

            # regression variables
            # OUTPUT: [ nu energy, nu zenith, nu azimuth, nu time, nu x, nu y, nu z, track length (0 for cascade), isTrack, flavor, type (anti = 1), isCC]

            output_labels.append( numpy.array([ float(nu_energy_tree), float(nu_zenith_tree), float(nu_azimuth_tree), float(nu_time_tree), float(nu_x_tree), float(nu_y_tree), float(nu_z_tree), float(track_length), float(isTrack), float(neutrino_type), float(particle_type), float(isCC) ]) )

        print("Got rid of %i events not in DC so far"%not_in_DC)
        print("Got rid of %i events classified as other so far"%isOther_count)

        # close the input file once we are done
        del event_file

    output_labels=numpy.asarray(output_labels)

    print("Got rid of %i events not in DC in total"%not_in_DC)

    return output_labels

#Construct list of filenames
import glob

file_name = input_file

event_file_names = sorted(glob.glob(file_name))
assert event_file_names,"No files loaded, please check path."

#Call function to read and label files
#Currently set to ONLY get track events, no cascades!!! #
labels = read_files(event_file_names)

print(labels.shape)

#Save output to hdf5 file
output_path = "/mnt/scratch/micall12/training_files/" + output_name + "_lt" + str(int(emax)) + "_vertex" + vertex_name + ".hdf5"
f = h5py.File(output_path, "w")
f.create_dataset("labels", data=labels)
f.close()

##########################
# Plot starting and ending positions
#
#
#
##########################

import numpy as np
import h5py
import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import argparse
import glob
from handle_data import CutMask
from handle_data import VertexMask

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_files",default=None,
                    type=str,dest="input_files", help="names for input files")
parser.add_argument("-d", "--path",type=str,default='/mnt/scratch/micall12/training_files/',
                    dest="path", help="path to input files")
parser.add_argument("-o", "--outdir",type=str,default='/mnt/home/micall12/LowEnergyNeuralNetwork/output_plots/',
                    dest="outdir", help="out directory for plots")
parser.add_argument("-n", "--name",default=None,
                    dest="name", help="name for output folder")
parser.add_argument("--emax",type=float,default=100.0,
                    dest="emax",help="Cut anything greater than this energy (in GeV)")
parser.add_argument("--emin",type=float,default=5.0,
                    dest="emin",help="Cut anything less than this energy (in GeV)")
parser.add_argument("-c", "--cuts",type=str, default="CC",
                    dest="cuts", help="Type of events to keep (all, cascade, track, CC, NC, etc.)")
parser.add_argument("--start",type=str, default="start_all",
                    dest="start_cut", help="Starting vertex cut (start_all, old_DC_start, DC_start, IC7_start)")
parser.add_argument("--end",type=str, default="end_all",
                    dest="end_cut", help="Ending vertex cut (end_all, IC19_end, IC7_end)")
parser.add_argument("--labels",type=str,default="labels",
                    dest="labels", help="name of truth array to read in from input files")
parser.add_argument("--transformed",default=False,action='store_true',
                    dest="transformed", help="add flag if labels truth input is already transformed")
parser.add_argument("--tmax",type=float,default=200.0,
                    dest="tmax",help="Multiplication factor for track, only used if transformed")
parser.add_argument("--apply_cuts",default=False,action='store_true',
                    dest="apply_cuts", help="perform energy and zenith cuts")
args = parser.parse_args()
input_files = args.path + args.input_files

output_path = args.outdir
name = args.name
outdir = output_path + name
if os.path.isdir(outdir) != True:
    os.mkdir(outdir)
print("Saving plots to %s"%outdir)

energy_min = args.emin
energy_max = args.emax
cut_name = args.cuts
start_cut = args.start_cut
end_cut = args.end_cut
apply_cuts = args.apply_cuts

transformed = args.transformed
truth_name = args.labels
track_max = args.tmax
azimuth_index = 2
track_index = 7

### Import Files ###
f = h5py.File(input_files, 'r')
labels = f[truth_name][:]
f.close()
del f
if transformed:
    labels[:,0] = labels[:,0]*energy_max
    labels[:,1] = np.arccos(labels[:,1])
    azimuth_index = 7
    track_index = 2
    labels[:,track_index] = labels[:,track_index]*track_max

# Apply Cuts
if apply_cuts:
    mask = CutMask(labels)
    cut_energy = np.logical_and(labels[:,0] > energy_min, labels[:,0] < energy_max)
    all_cuts = np.logical_and(mask[cut_name], cut_energy)
    labels = labels[all_cuts]

## WHAT EACH ARRAY CONTAINS! ##
# reco: (energy, zenith, azimuth, time, x, y, z, track length) 
# stats: (count_outside, charge_outside, count_inside, charge_inside) 
# num_pulses: [ string num, dom index, num pulses]
# trig_time: [DC_trigger_time]

num_events = labels.shape[0]
print(num_events)

# Set up DOM and strings
data_DC = np.genfromtxt("detector_information/icecube_string86.txt", delimiter=' ', names=['string_86','x_DC','y_DC','z_DC'])
data_IC = np.genfromtxt("detector_information/icecube_string36.txt", delimiter=' ', names=['string_36','x_IC','y_IC','z_IC'])
data_stringsXY = np.genfromtxt("detector_information/icecube_stringsXY.txt", delimiter=' ', names=['string_num','x_XY','y_XY'])

z_DC = data_DC['z_DC']
x_DC = data_DC['x_DC']
y_DC = data_DC['y_DC']

z_IC = data_IC['z_IC'][:-4]
x_IC = data_IC['x_IC'][:-4]
y_IC = data_IC['y_IC'][:-4]

strings = np.arange(1,61,1)

x_XY = data_stringsXY['x_XY']
y_XY = data_stringsXY['y_XY']

abs_x_DC = x_DC - x_IC
abs_y_DC = y_DC - y_IC
rho_DC = np.sqrt(abs_x_DC**2 + abs_y_DC**2)
abs_x_IC =  x_IC -  x_IC
abs_y_IC = y_IC - y_IC
rho_IC = np.sqrt(abs_x_IC**2 + abs_y_IC**2)
abs_x_XY = x_XY - np.ones((86))*x_IC[0]
abs_y_XY = y_XY - np.ones((86))*y_IC[0]
rho_XY = np.sqrt(abs_x_XY**2 + abs_y_XY**2)
DC_strings = [79, 80, 81, 82, 83, 84, 85, 86]
IC_near_DC_strings = [17, 18, 19, 25, 26, 27, 28, 34, 35, 36, 37, 38, 44, 45, 46, 47, 54, 55, 56]

# Boundaries
z_min_start = -505 - 50
z_max_start = -155 + 50
rho_start = 150
z_min_end = -505 - 50
z_max_end = 505 + 50
rho_end =  260

# Find position from labels
x_start = labels[:,4]
y_start = labels[:,5]
z_start = labels[:,6]
track_length = labels[:,track_index]
x_origin = np.ones((len(x_start)))*46.290000915527344
y_origin = np.ones((len(y_start)))*-34.880001068115234

#Starting
r_start = np.sqrt( (x_start - x_origin)**2 + (y_start - y_origin)**2)

#Ending
theta = labels[:,1]
phi = labels[:,azimuth_index]
n_x = np.sin(theta)*np.cos(phi)
n_y = np.sin(theta)*np.sin(phi)
n_z = np.cos(theta)
x_end = x_start + track_length*n_x
y_end = y_start + track_length*n_y
r_end = np.sqrt( (x_end - x_origin)**2 + (y_end - y_origin)**2 )
z_end = z_start + track_length*n_z

# Plotting
fig = plt.figure(figsize=(14,12))
ax=fig.add_subplot(111)
size=5

# Plot patches of start and end regions
CNN_end = patches.Rectangle((0,z_min_end),rho_end,z_max_end-z_min_end)
CNN_start = patches.Rectangle((0,z_min_start),rho_start,z_max_start-z_min_start)
boxes = [CNN_end,CNN_start]
facecolor=["blue","green"]
pc = PatchCollection(boxes,facecolor=facecolor,alpha=0.3)
ax.add_collection(pc)

# Plot starting position only
ax.plot(r_end,z_end,'b.',label="ending vertex")
ax.plot(r_start,z_start,'g.',label="starting vertex")

# plot strings and DOMs in region
for one_string in IC_near_DC_strings:
    rho_squared = np.ones((60))*rho_XY[one_string-1] #*rho_XY[one_string-1]
    if one_string==36:
        ax.plot(rho_IC,z_IC,'o',color='gray',markersize=size,linewidth=1,label='IceCube DOMS')
    else:
        ax.plot(rho_squared, z_IC, 'o', color='gray',markersize=size,linewidth=1)
for one_string in DC_strings:
    rho_squared = np.ones((60))*rho_XY[one_string-1] #*rho_XY[one_string-1]
    if one_string == 86:
        ax.plot(rho_squared, z_DC, 'o', color='red',markersize=size,linewidth=1,label='DeepCore DOMS')
    else:
        ax.plot(rho_squared, z_DC, 'o', color='red',markersize=size,linewidth=1)

ax.legend(fontsize=24, loc='upper center', shadow=True,fancybox=True,bbox_to_anchor=(0.5,1.16),ncol=2)
ax.set_xlabel(r'$\rho$ (m)',size=30)
ax.set_ylabel('z depth (m)',size=30)
ax.tick_params(labelsize=30,pad=10)
fig.savefig("%s/StartingEndingVertexCheck.png"%outdir)


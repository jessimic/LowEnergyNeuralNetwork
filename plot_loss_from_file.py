import matplotlib
matplotlib.use('Agg')
import numpy
import argparse
import os
import glob

from PlottingFunctions import plot_history_from_list
from PlottingFunctions import plot_history_from_list_split

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_folder",type=str,default=None,
                    dest="input_folder", help="name of folder in output_plots")
parser.add_argument("-e", "--epoch",default=None,
                    dest="epoch", help="number of epoch folder to grab from")
parser.add_argument("--ymin",default=None,
                    dest="ymin", help="min y value for loss plot")
parser.add_argument("--ymax",default=None,
                    dest="ymax", help="max y value for loss plot")
args = parser.parse_args()

directory = args.input_folder
epoch = args.epoch
ymin = args.ymin
ymax = args.ymax
if ymin:
    ymin = float(ymin)
if ymax:
    ymax = float(ymax)

print("Epoch: %s, ymin: %s, ymax: %s"%(epoch,ymin,ymax))

folder_name = "/mnt/home/micall12/DNN_LE/output_plots/%s/"%directory

if epoch:
    file_name = "%ssaveloss_%iepochs.txt"%(folder_name,epoch)
else:
    all_files = sorted(glob.glob("%ssaveloss_*epochs.txt"%folder_name), key=os.path.getsize)
    file_name = all_files[-1]

print("Using loss from %s"%file_name)

f = open(file_name)

loss = {}
number_lines = len(f.readlines())
if number_lines == 4:
    loss_names = ['loss', 'energy_loss', 'val_loss', 'val_energy_loss']
if number_lines == 6:
    loss_names = ['loss', 'energy_loss', 'zenith_loss', 'val_loss', 'val_energy_loss', 'val_zenith_loss']
f.close()

f = open(file_name)
index = 0
for line in f.readlines():
    delete_start = len(loss_names[index]) + 4
    delete_end = -4
    pretty_line = line[delete_start:delete_end]
    loss[loss_names[index]] = pretty_line.split(',')
    for a_loss in range(0,len(loss[loss_names[index]])):
        loss[loss_names[index]][a_loss] = float(loss[loss_names[index]][a_loss])
    index +=1
f.close()

plot_history_from_list(loss['loss'],loss['val_loss'],save=True,savefolder=folder_name,logscale=True,ymin=ymin,ymax=ymax)
if number_lines == 6:
    plot_history_from_list_split(loss['energy_loss'],loss['val_energy_loss'],loss['zenith_loss'],loss['val_zenith_loss'],save=True,savefolder=folder_name,logscale=True,ymin=ymin,ymax=ymax)


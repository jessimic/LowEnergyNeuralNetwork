import matplotlib
matplotlib.use('Agg')
import numpy
import argparse
import os
import glob
import matplotlib.pyplot as plt

from PlottingFunctions import plot_history_from_list
from PlottingFunctions import plot_history_from_list_split

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_folder",type=str,default=None,
                    dest="input_folder", help="name of folder in output_plots")
parser.add_argument("-d","--dir",type=str,default="/mnt/home/micall12/DNN_LE/output_plots/",
                    dest="outplots_dir", help="path to output plots directory (including it)")
parser.add_argument("-e", "--epoch",default=None,
                    dest="epoch", help="number of epoch folder to grab from")
parser.add_argument("--ymin",default=None,
                    dest="ymin", help="min y value for loss plot")
parser.add_argument("--ymax",default=None,
                    dest="ymax", help="max y value for loss plot")
args = parser.parse_args()

plot_folder = args.input_folder
outplots_dir = args.outplots_dir
epoch = args.epoch
ymin = args.ymin
ymax = args.ymax
if ymin:
    ymin = float(ymin)
if ymax:
    ymax = float(ymax)

print("Epoch: %s, ymin: %s, ymax: %s"%(epoch,ymin,ymax))

full_path = "%s/%s/"%(outplots_dir,plot_folder)
file_name = "%ssaveloss_currentepoch.txt"%(full_path)
print("Using loss from %s"%file_name)

delimiter="\t"
f = open(file_name)
header = f.readline().split(delimiter)
header = header[:-1]
print(header)
    
rawdata = numpy.genfromtxt(file_name, skip_header=1)

data = {}
for variable in range(0,len(header)):
    data[header[variable]] = rawdata[:epoch,variable]

print(data.keys())

# Timing stats
plt.figure()
plt.plot(data[header[0]],data[header[1]],label="load + train")
plt.plot(data[header[0]],data[header[2]],label="train")
plt.xlabel("epoch")
plt.ylabel("times (minutes)")
plt.legend() 
plt.savefig("%sTrainingTimePerEpoch.png"%full_path)

# Loss Plots
plot_history_from_list(data['loss'],data['val_loss'],save=True,savefolder=full_path,logscale=True,ymin=ymin,ymax=ymax)
if len(header)-3 == 6:
    plot_history_from_list_split(data['EnergyLoss'],data['val_EnergyLoss'],data['ZenithLoss'],loss['val_ZenithLoss'],save=True,savefolder=full_path,logscale=True,ymin=ymin,ymax=ymax)

# Average Validation Plot
def average_epochs(loss_list,start=None,end=None,file_num=7):
    
    save = 0
    avg_loss = []
    avg_epoch = []
    if not start:
        start = 1
    if not end:
        end = len(loss_list)+1
    
    for i in range(start,end):
        save += loss_list[i-1]
        if i%file_num == 0:
            avg_loss.append(save/file_num)
            avg_epoch.append(i)
            save = 0
    
    min_full_pass = avg_loss.index(min(avg_loss))+1
    best_model = min_full_pass*file_num
    best_loss = min(avg_loss)
    print("Best loss at %i with value of %f"%(best_model,best_loss))
    return avg_loss, avg_epoch, best_model, best_loss

avg_val_loss, avg_epoch,best_model,best_loss = average_epochs(data['val_loss'])
name = header[-1][:-4]
ymin=min(min(data['loss']),min(data['val_loss']))
ymax=max(max(data['loss']),max(data['val_loss']))
plt.figure(figsize=(10,7))
plt.plot([best_model,best_model],[ymin,ymax],linewidth=4,color='lime')
plt.plot(data["Epoch"],data['loss'],'b',label="%s Training"%name)
plt.plot(data["Epoch"],data['val_loss'],'c',label="%s Validation"%name)
plt.plot(avg_epoch,avg_val_loss,'r',label="Avg %s Validation"%name)
plt.yscale('log')
plt.title("Training and Validation Loss after %s Epochs"%len(data['loss']),fontsize=25)
plt.xlabel('Epochs',fontsize=15)
plt.ylabel('Loss',fontsize=15)
plt.legend(loc="upper right",fontsize=15)
textstr = "Best Avg Model: %i \n Best Avg Loss: %.3f"%(best_model,best_loss)
props = dict(boxstyle='round', facecolor='blue', alpha=0.3)
plt.text(0.7, 0.5, textstr, fontsize=20,
        verticalalignment='top', horizontalalignment='left',bbox=props)
plt.savefig("%sAvgLossVsEpoch.png"%full_path)

# Loss Plots Again
#plot_history_from_list(data['loss'][:best_model],data['val_loss'][:best_model],save=True,savefolder=full_path,logscale=True,ymin=ymin,ymax=ymax)

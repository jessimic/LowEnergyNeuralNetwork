import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import argparse
import math
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename",type=str,default=None,
                    dest="filename", help="name of plot png that you want to grab, including .png")
parser.add_argument("-i", "--input_folder",type=str,default=None,
                    dest="input_folder", help="name of folder in output_plots")
parser.add_argument("-d","--dir",type=str,default="/home/users/jmicallef/LowEnergyNeuralNetwork/output_plots/",
                    dest="outplots_dir", help="path to output plots directory (including it)")
parser.add_argument("--test_type",type=str,default="oscnext",
                    dest="test_type", help="name of subfolder (oscnext or dragon) to pull from")
parser.add_argument("--rows",type=int,default=None,
                    dest="rows", help="number of rows to plot")
parser.add_argument("--cols",type=int,default=None,
                    dest="cols", help="number of cols to plot")
parser.add_argument("-e", "--epochs",nargs='+',default=None,
                    dest="epochs", help="number of epoch folder to grab from")
args = parser.parse_args()

main_path = args.outplots_dir + args.input_folder + "/" 
plot = args.filename
epoch_list = args.epochs
test_type = args.test_type

rows = args.rows
cols = args.cols
if rows is None and cols is None:
    rows = math.ceil(np.sqrt(len(epoch_list)))
    cols = rows
elif (rows is not None and cols is None):
    cols = math.ceil(len(epoch_list)/rows)
elif (rows is None and cols is not None):
    rows = math.ceil(len(epoch_list)/cols)

row_index = 0
col_index = 0
fig, ax = plt.subplots(rows,cols)
if epoch_list is None:
    epoch_name = test_type
else:
    image = {}
    for epoch in epoch_list:
        epoch_name = "%s_%iepochs"%(test_type,int(epoch))
        epoch_path = main_path + epoch_name + "/" + plot
        with cbook.get_sample_data(epoch_path) as image_file:
            image = plt.imread(image_file)        
        if rows > 1 and cols > 1:
            im = ax[row_index,col_index].imshow(image)
            ax[row_index,col_index].axis('off')
            if col_index < (cols-1):
                col_index +=1
            else:
                row_index +=1
                col_index = 0 
        else:
            im = ax[col_index].imshow(image)
            ax[col_index].axis('off')
            col_index +=1
plt.subplots_adjust(wspace=0.05, hspace=0.05)
 
outname = main_path + plot[:-4] + "_compare%iplots.png"%len(epoch_list)
plt.savefig(outname,dpi=800)

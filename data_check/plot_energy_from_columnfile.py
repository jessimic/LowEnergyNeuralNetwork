import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input",default=None,
                    dest="input_file", help="input file and path")
parser.add_argument("-n", "--name",default=None,
                    dest="name", help="name of sample")
parser.add_argument("--input2",default=None,
                    dest="input2_file", help="optional input file 2 (including path)")
parser.add_argument("--name2",default=None,
                    dest="name2", help="name of sample #2")
parser.add_argument("-d","--dir",type=str,default="/mnt/home/micall12/DNN_LE/output_plots/",
                    dest="outplots_dir", help="path to output plots directory (including it)")
parser.add_argument("--emin",type=int,default=1,
                    dest="emin", help="min energy value for histogram to plot")
parser.add_argument("--emax",type=int,default=499,
                    dest="emax",help="max energy value for histogram to plot")
parser.add_argument("--bin_size",type=int,default=1,
                    dest="bin_size", help="bin sizes (step size‚ for histogram")
parser.add_argument("--ymax",default=None,
                    dest="ymax",help="max y value (counts) for histogram distribution")
args = parser.parse_args()

infile1 = args.input_file
infile2 = args.input2_file
outplots_dir = args.outplots_dir
name = args.name
name2 = args.name2
emin = args.emin
emax = args.emax
bin_size = args.bin_size
ymax = args.ymax
plot_range = np.arange(emin,emax,bin_size)

# data = [line/index] [index, energy value, count at energy value]
data = np.genfromtxt(infile1, skip_header=3)
emin_index = np.where(data[:,1]==emin)[0][0]
emax_index = np.where(data[:,1]==emax)[0][0]
energy_count=data[emin_index:emax_index,2]
assert (data[emin_index+1,1]-data[emin_index,1])==bin_size,"Need to resize bins for input file 1"
if infile2:
    data2 = np.genfromtxt(infile2, skip_header=3)
    emin_index2 = np.where(data2[:,1]==emin)[0][0]
    emax_index2 = np.where(data2[:,1]==emax)[0][0]
    energy_count2=data2[emin_index2:emax_index2,2]
    assert (data2[emin_index+1,1]-data2[emin_index,1])==bin_size,"Need to resize bins for input file 2"

plt.figure(figsize=(10,8))
plt.title("Events Binned by 1 GeV",fontsize=20)
plt.bar(plot_range, energy_count, alpha=0.5, width=1, align='edge')
plt.xlabel("energy (GeV)",fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel("number of events",fontsize=20)
plot_name="%sEnergyDistribution_%imin_%imax_%iGeVbins"%(name,emin,emax,bin_size)
if ymax:
    plt.ylim(0,int(ymax))
    plot_name+="_ylim"
plt.savefig("%s/%s.png"%(outplots_dir,plot_name))

if infile2:
    plt.figure(figsize=(10,8))
    plt.title("Events Binned by 1 GeV",fontsize=20)
    plt.bar(plot_range, energy_count2, alpha=0.5, width=1, align='edge')
    plt.xlabel("energy (GeV)",fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel("number of events",fontsize=20)
    if ymax:
        plt.ylim(0,int(ymax))
        plot_name+="_ylim"
    plot_name="%sEnergyDistribution_%imin_%imax_%iGeVbins"%(name2,emin,emax,bin_size)
    plt.savefig("%s/%s.png"%(outplots_dir,plot_name))
        
    ratio = energy_count2/energy_count
    plt.figure(figsize=(10,8))
    plt.title("Ratio of Events Binned by 1 GeV",fontsize=20)
    plt.bar(plot_range, ratio, alpha=0.5, width=1, align='edge')
    plt.xlabel("energy (GeV)",fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel("ratio of events %s/%s"%(name2,name),fontsize=20)
    plot_name="%s%sRatioEnergyDistribution_%imin_%imax_%iGeVbins"%(name,name2,emin,emax,bin_size)
    plt.savefig("%s/%s.png"%(outplots_dir,plot_name))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib as mpl
label_size = 15
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size

def cut_kept_hist(var_array,cut_mask,kept_mask,weights_here=None,
                particle="Muon",var_type="True",variable="Energy",
                units="GeV", bins=50,amin=0,amax=100,notebook=False):
    """
    var_array = 1D array of variable, for example true_energy
    cut_mask = boolean 1D array size of var_array with True == events cut
    kept_mask = boolean 1D array size of var_array with True == events kept
    weights_here = 1D array size of var_array, weights for plotting
    """
    plt.figure(figsize=(10,7))
    
    if weights_here is None:
        plt.ylabel("Normalized Counts",fontsize=20)
        weights_here = np.ones(len(var_array))
    else:
         plt.ylabel("Normalized Weighted Counts",fontsize=20)
    plt.hist(var_array[cut_mask], color="g",
         label="Cut %s"%particle,bins=bins,range=[amin,amax],
         weights=weights_here[cut_mask],alpha=0.5,density=True) 
    plt.hist(var_array[kept_mask], color="b",
         label="Kept %s"%particle,bins=bins,range=[amin,amax],
         weights=weights_here[kept_mask],alpha=0.5,density=True)
    plt.legend(loc='upper right',fontsize=15)
    plt.title("%s %s %s Distribution"%(particle, var_type, variable),fontsize=25)
    plt.xlabel("%s %s"%(variable, units),fontsize=20)
    plt.savefig("%s/%s%s_%sDist.png"%(path,var_type,particle,variable))gy_array[:-1],efficiency_mu_array,'b.-',markersize=10,linewidth=2)

    if notebook == False:
        plt.close()

def cut_kept_ratio(var_array,cut_mask,kept_mask,weights_here=None,
                particle="Muon",var_type="True",variable="Energy",
                units="GeV",bins=50,amin=0,amax=100,notebook=False):
    
    hist_cut, xbin = np.histogram(var_array[cut_mask], bins=bins,range=[amin,amax],
         weights=weights_here[cut_mask]);
    hist_cut_sq, xbin = np.histogram(var_array[cut_mask], bins=bins,range=[amin,amax],
         weights=weights_here[cut_mask]*weights_here[cut_mask]);
    hist_kept, xbin = np.histogram(var_array[kept_mask],bins=bins,range=[amin,amax],
         weights=weights_here[kept_mask]);
    hist_kept_sq, xbin = np.histogram(var_array[kept_mask],bins=bins,range=[amin,amax],
         weights=weights_here[kept_mask]*weights_here[kept_mask]);
    
    xstep = (xbin[1] - xbin[0])
    xplots = np.arange(amin+(xstep/2),amax,xstep)
    
    if sum(kept_mask) > sum(cut_mask):
        ratio = hist_cut/hist_kept
        ylabel = "Cut/Kept"
    else:
        ratio = hist_kept/hist_cut
        ylabel = "Kept/Cut"
    
    efrac_cut = np.sqrt(hist_cut_sq)/hist_cut
    efrac_kept = np.sqrt(hist_kept_sq)/hist_kept
    error_tot = efrac_cut + efrac_kept
    error_abs = ratio*error_tot
    
    plt.figure(figsize=(10,7))
    plt.errorbar(xplots, ratio,yerr=error_abs,xerr=xstep,fmt="b.",markersize=10)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title("%s %s %s Ratio"%(particle, var_type, variable),fontsize=25)
    plt.ylabel("Ratio of %s"%ylabel,fontsize=20)
    plt.xlabel("%s %s"%(variable, units),fontsize=20)
    plt.savefig("%s/%s%s_%sRatio.png"%(path,var_type,particle,variable))

    if notebook == False:
        plt.close()

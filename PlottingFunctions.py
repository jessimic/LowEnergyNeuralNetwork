#!/usr/bin/env python

############################################
# Scripts for plotting functions
# Contains functions:
#     get_RMS - used to calculate RMS for plotting statistics
#     get_FWHM - FWHM calculation method, not currently plugged in
#     plot_history - scatter line plot of loss vs epochs
#     plot_distributions_CCNC - plot energy distribution for truth, and for NN reco
#     plot_resolutions_CCNC - plot energy resoltuion for (NN reco - truth)
#     plot_2D_prediction - 2D plot of True vs Reco
#     plot_single_resolution - Resolution histogram, (NN reco - true) and can compare (old reco - true)
#     plot_compare_resolution - Histograms of resolutions for systematic sets, overlaid
#     plot_systematic_slices - "Scatter plot" with systematic sets on x axis and 68% resolution on y axis
#     plot_energy_slices - Scatter plot energy cut vs resolution
##############################################

import numpy
import h5py
import os, sys
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy import interpolate
import scipy.stats
from scipy.signal import peak_widths

import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)

def get_RMS(resolution):
    mean_array = numpy.ones_like(resolution)*numpy.mean(resolution)
    rms = numpy.sqrt( sum((mean_array - resolution)**2)/len(resolution) )
    return rms

def get_FWHM(resolution,bins):
    x_range = numpy.linspace(min(resolution),max(resolution),bins)
    y_values,bin_edges = numpy.histogram(resolution,bins=bins)
    spline = UnivariateSpline(x_range,y_values - max(y_values)/2.)
    r = spline.roots()
    if len(r) != 2:
        print("Root are weird")
        print(r)
        r1 = 0
        r2 = 0
    else:
        r1, r2 = spline.roots()
    return r1, r2

def plot_history(network_history,save=False,savefolder=None,use_logscale=False):
    """
    Plot history of neural network's loss vs. epoch
    Recieves:
        network_history = array, saved metrics from neural network training
        save = optional, bool to save plot
        savefolder = optional, output folder to save to, if not in current dir
    Returns:
        line scatter plot of epoch vs loss
    """
    plt.figure(figsize=(10,7))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    if use_logscale:
        plt.yscale('log')
    plt.plot(network_history.history['loss'])
    plt.plot(network_history.history['val_loss'])
    plt.legend(['Training', 'Validation'])
    if save == True:
        plt.savefig("%sloss_vs_epochs.png"%savefolder)

    plt.show()

def plot_distributions_CCNC(truth_all_labels,truth,reco,save=False,savefolder=None):
    """
    Plot testing set distribution, with CC and NC distinguished
    Recieves:
        truth_all_labels = array, Y_test truth labels that have ALL values in them (need CC vs NC info)
        truth = array, Y_test truth labels
        reco = array, neural network prediction output
        save = optional, bool to save plot
        savefolder = optional, output folder to save to, if not in current dir
    Returns:
        1D histogram of reco - true with sepearated CC and NC distinction
    """
    CC_mask = truth_all_labels[:,11] ==1
    NC_mask = truth_all_labels[:,11] ==0
    num_CC = sum(CC_mask)
    num_NC = sum(NC_mask)
    print("CC events: %i, NC events: %i, Percent NC: %.2f"%(num_CC,num_NC,float(num_NC/(num_CC+num_NC))*100.))

    plt.figure(figsize=(10,7))
    plt.title("True Energy Distribution")
    plt.hist(truth[CC_mask], bins=100,color='b',alpha=0.5,label="CC");
    plt.hist(truth[NC_mask], bins=100,color='g',alpha=0.5,label="NC");
    plt.xlabel("Energy (GeV)")
    plt.legend()
    if save:
        plt.savefig("%sTrueEnergyDistribution_CCNC.png"%savefolder)

    plt.figure(figsize=(10,7))
    plt.title("NN Energy Distribution")
    plt.hist(reco[CC_mask], bins=100,color='b', alpha=0.5, label="CC");
    plt.hist(reco[NC_mask], bins=100,color='g', alpha=0.5, label="NC");
    plt.xlabel("Energy (GeV)")
    plt.legend()
    if save:
        plt.savefig("%sNNEnergyDistribution_CCNC.png"%savefolder)

def plot_resolution_CCNC(truth_all_labels,truth,reco,save=False,savefolder=None):
    """
    Plot testing set resolution of reconstruction - truth, with CC and NC distinguished
    Recieves:
        truth_all_labels = array, Y_test truth labels that have ALL values in them (need CC vs NC info)
        truth = array, Y_test truth labels
        reco = array, neural network prediction output
        save = optional, bool to save plot
        savefolder = optional, output folder to save to, if not in current dir
    Returns:
        1D histogram of reco - true with sepearated CC and NC distinction
    """
    CC_mask = truth_all_labels[:,11] ==1
    NC_mask = truth_all_labels[:,11] ==0
    num_CC = sum(CC_mask)
    num_NC = sum(NC_mask)
    print("CC events: %i, NC events: %i, Percent NC: %.2f"%(num_CC,num_NC,float(num_NC/(num_CC+num_NC))*100.))

    resolution = reco - truth
    resolution_fraction = (reco - truth)/truth
    resolution = numpy.array(resolution)
    resolution_fraction  = numpy.array(resolution_fraction)

    plt.figure(figsize=(10,7))  
    plt.title("Energy Resolution")
    plt.hist(resolution[CC_mask], bins=50,color='b', alpha=0.5, label="CC");
    plt.hist(resolution[NC_mask], bins=50,color='g', alpha=0.5, label="NC");
    plt.xlabel("NN reconstruction - truth (GeV)")
    plt.legend()
    if save:
        plt.savefig("%sEnergyResolution_CCNC.png"%savefolder)

    plt.figure(figsize=(10,7))  
    plt.title("Fractional Energy Resolution")
    plt.hist(resolution_fraction[CC_mask], bins=50,color='b', alpha=0.5, label="CC");
    plt.hist(resolution_fraction[NC_mask], bins=50,color='g', alpha=0.5, label="NC");
    plt.xlabel("(NN reconstruction - truth) / truth")
    plt.legend()
    if save:
        plt.savefig("%sEnergyResolutionFrac_CCNC.png"%savefolder)

def plot_2D_prediction(truth, nn_reco, \
                        save=False,savefolder=None,syst_set="",\
                        use_fraction=False,bins=60,\
                        minenergy=0.,maxenergy=60.):
    """
    Plot testing set reconstruction vs truth
    Recieves:
        truth = array, Y_test truth
        nn_reco = array, neural network prediction output
        save = optional, bool to save plot
        savefolder = optional, output folder to save to, if not in current dir
    Returns:
        2D plot of True vs Reco
    """
    if not use_fraction:
        plt.figure(figsize=(10,7))
        cts,xbin,ybin,img = plt.hist2d(truth, nn_reco, bins=bins)
        plt.plot([minenergy,maxenergy],[minenergy,maxenergy],'k:')
        plt.xlim(minenergy,maxenergy)
        plt.ylim(minenergy,maxenergy)
        plt.xlabel("True Neutrino Energy (GeV)")
        plt.ylabel("NN Reconstruction Energy (GeV)")
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('counts', rotation=90)
        plt.set_cmap('viridis_r')
        plt.title("Reconstruction (from NN) vs Truth for Energy")
        if save == True:
            plt.savefig("%sTruthReco_2DHist%s.png"%(savefolder,syst_set))
    
    if use_fraction:
        fractional_error = abs(truth - nn_reco)/ truth
        plt.figure(figsize=(10,7))
        plt.title("Fractional Error vs. Energy")
        plt.hist2d(truth, fractional_error,bins=60);
        plt.xlabel("True Energy (GeV)")
        plt.ylabel("Fractional Error")
        #plt.ylim(0,0.5)
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('counts', rotation=90)
        if save == True:
            plt.savefig("%sTruthRecoFrac_2DHist%s.png"%(savefolder,syst_set))

def plot_single_resolution(truth,nn_reco,\
                           bins=100, use_fraction=False,\
                           use_old_reco = False, old_reco=None,\
                           energy_min=None,energy_max=None,\
                           save=False,savefolder=None):
    """Plots resolution for dict of inputs, one of which will be a second reco
    Recieves:
        truth = array of truth or Y_test labels
        nn_reco = array of NN predicted reco or Y_test_predicted results
        bins = int value
        use_fraction = use fractional resolution instead of absolute, where (reco - truth)/truth
        use_reco = True if you want to compare to another reconstruction (like pegleg)
        old_reco = optional, pegleg array of labels
        energy_min = float, min energy if cut desired
        energy_max = float, max energy if cut desired
    Returns:
        1D histogram of Reco - True (or fractional)
        Can have two distributions of NN Reco Resolution vs Pegleg Reco Resolution
    """
    
    fig, ax = plt.subplots(figsize=(10,7)) 
    
    if use_fraction:
        nn_resolution = (nn_reco - truth)/truth
        if use_old_reco:
            old_reco_resolution = (old_reco - truth)/truth
        title = "Fractional Energy Resolution"
        xlabel = "(reconstruction - truth) / truth" 
    else:
        nn_resolution = nn_reco - truth
        if use_old_reco:
            old_reco_resolution = old_reco - truth
        title = "Energy Resolution"
        xlabel = "reconstruction - truth (GeV)"
    
    #Cut if there is an energy cut
    if energy_min or energy_max:
        if energy_min and energy_max:
            energy_mask = numpy.logical_and(truth > energy_min, truth < energy_max)
            title += " (%.2f < energy < %.2f)"%(energy_min,energy_max)
        if energy_min and not energy_max:
            energy_mask = truth > energy_min
            title += " (energy > %.2f)"%(energy_min)
        if energy_max and not energy_min:
            energy_mask = truth < energy_max
            title += " (energy < %.2f)"%(energy_max)
    
        nn_resolution = resolution[energy_mask]
        if use_old_reco:
            old_reco_resolution = old_reco_resolution[energy_mask]
    
    ax.hist(nn_resolution, bins=bins, alpha=0.5, label="neural net");
    
    #Statistics
    rms_nn = get_RMS(nn_resolution)
    r1, r2 = numpy.percentile(nn_resolution, [16,84])
    textstr = '\n'.join((
            r'$\mathrm{events}=%i$' % (numpy.sum(nn_resolution), ),
            r'$\mathrm{median}=%.2f$' % (numpy.median(nn_resolution), ),
            r'$\mathrm{RMS}=%.2f$' % (rms_nn, ),
            r'$\mathrm{1\sigma}=%.2f,%.2f$' % (r1,r2 )))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.6, 0.95, textstr, transform=ax.transAxes, fontsize=20,
        verticalalignment='top', bbox=props)
    
    if use_old_reco:
        rms_old_reco = get_RMS(old_reco_resolution)
        ax.hist(old_reco_resolution, bins=bins, alpha=0.5, label="pegleg");
        ax.legend()
    
        r1_old_reco, r2_old_reco = numpy.percentile(old_reco_resolution, [16,84])
        textstr = '\n'.join((
            r'$\mathrm{events}=%i$' % (numpy.sum(old_reco_resolution), ),
            r'$\mathrm{median}=%.2f$' % (numpy.median(old_reco_resolution), ),
            r'$\mathrm{RMS}=%.2f$' % (rms_old_reco, ),
            r'$\mathrm{1\sigma}=%.2f,%.2f$' % (r1_old_reco,r2_old_reco )))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.2, 0.95, textstr, transform=ax.transAxes, fontsize=20,
            verticalalignment='top', bbox=props)
    
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    
    savename = "EnergyResolution"
    if use_fraction:
        savename += "Frac"
    if use_old_reco:
        savename += "_CompareOldReco"
    if save == True:
        plt.savefig("%s%s.png"%(savefolder,savename))


def plot_compare_resolution(truth,nn_reco,namelist, savefolder=None,\
                            num_namelist = None,save=False,bins=100,use_fraction=False):
    """Plots resolution for dict of inputs
    Receives:
        truth = dict of truth or Y_test labels
                (contents = [key name, energy], shape = [number syst sets, number of events])
        nn_reco = dict of NN predicted or Y_test_predicted results
                    (contents = [key name, energy], shape = [number syst sets, number of events])
        namelist = list of names for the dict, to use as pretty labels
        save_folder_name = string for output file
        num_namelist = shorthand for names of sets (numerical version), for printing
        save = bool where True saves and False does not save plot
        bins = int value
        use_fraction: bool, uses fractional resolution if True
    Returns:
        Histograms of resolutions for systematic sets, overlaid
        Prints statistics for all histograms into table
    """
    
    print("Resolution")
    print('Name\t Mean\t Median\t RMS\t Percentiles\t')
    plt.figure(figsize=(10,7)) 
    if use_fraction:
        title = "Fractional Energy Resolution"
        xlabel = "(NN reconstruction - truth) / truth"
        plt.legend(fontsize=20)
    else:
        title = "Energy Resolution"
        xlabel = "NN reconstruction - truth (GeV)"
        plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=2)
    
    for index in range(0,len(namelist)):
        keyname = "file_%i"%index
        if use_fraction:
            resolution = (nn_reco[keyname] - truth[keyname]) / truth[keyname]
        else:
            resolution = nn_reco[keyname] - truth[keyname]
        plt.hist(resolution, bins=60, alpha=0.5, label="%s"%namelist[index]);
        
        #Statistics
        rms = get_RMS(resolution)
        #r1, r2 = get_FWHM(resolution,bins)
        r1, r2 = numpy.percentile(resolution, [16,84])
        
        if num_namelist:
            names = num_namelist
        else:
            names = namelist
            
        print("%s\t %.2f\t %.2f\t %.2f\t %.2f, %.2f\t"%(names[index], \
                                                        numpy.mean(resolution),\
                                                        numpy.median(resolution),\
                                                        rms,\
                                                        r1, r2))
    plt.title(title)    
    plt.xlabel(xlabel)
    
    if save:
        if use_fraction:
            plt.savefig("%sFractionalEnergyResolution_CompareSets.png"%savefolder)
        else:
            plt.savefig("%sEnergyResolution_CompareSets.png"%savefolder)

def plot_systematic_slices(truth_dict, nn_reco_dict,\
                           namelist, use_fraction=False, \
                           use_old_reco = False, old_reco_dict=None,\
                           save=False,savefolder=None):
    """Plots different arrays vs each other (systematic set arrays)
    Receives:
        truth_dict = dict of arrays with truth labels
                    (contents = [key name, energy], shape = [number syst sets, number of events])
        nn_reco_dict = dict of arrays that has NN predicted reco results
                        (contents = [key name, energy], shape = [number syst sets, number of events])
        namelist = list of names to be used for x_axis ticks
        use_fraction = use fractional resolution instead of absolute, where (reco - truth)/truth
        use_reco = True if you want to compare to another reconstruction (like pegleg)
                    (contents = [key name, energy], shape = [number syst sets, number of events])
        old_reco = optional, dict of pegleg arrays with the labels
    Returns:
        "scatter plot" with systematic sets on x axis,
        y axis has median of resolution with error bars containing 68% of resolution
    """
    
    number_sets = len(namelist)
    percentile_in_peak = 68.27

    left_tail_percentile  = (100.-percentile_in_peak)/2
    right_tail_percentile = 100.-left_tail_percentile
    
    medians  = numpy.zeros(number_sets)
    err_from = numpy.zeros(number_sets)
    err_to   = numpy.zeros(number_sets)
    
    if use_old_reco:
        medians_old_reco  = numpy.zeros(number_sets)
        err_from_old_reco = numpy.zeros(number_sets)
        err_to_old_reco   = numpy.zeros(number_sets)
    
    resolution = {}
    for index in range(0,number_sets):
        keyname = "file_%i"%index
        if use_fraction:
            resolution = (nn_reco_dict[keyname] - truth_dict[keyname])/truth_dict[keyname]
        else:
            resolution = (nn_reco_dict[keyname] - truth_dict[keyname])
    
        lower_lim = numpy.percentile(resolution, left_tail_percentile)
        upper_lim = numpy.percentile(resolution, right_tail_percentile)
        median = numpy.percentile(resolution, 50.)
        
        medians[index] = median
        err_from[index] = lower_lim
        err_to[index] = upper_lim
    
        if use_old_reco:
            if use_fraction:
                resolution_old_reco = ((old_reco_dict[keyname]-truth_dict[keyname])/truth_dict[keyname])
            else:
                resolution_old_reco = (old_reco_dict[keyname]-truth_dict[keyname])
            
            lower_lim_reco = numpy.percentile(resolution_old_reco, left_tail_percentile)
            upper_lim_reco = numpy.percentile(resolution_old_reco, right_tail_percentile)
            median_reco = numpy.percentile(resolution_old_reco, 50.)
            
            medians_reco[index] = median_reco
            err_from_reco[index] = lower_lim_reco
            err_to_reco[index] = upper_lim_reco


    x_range = numpy.linspace(1,number_sets,number_sets)
    
    fig, ax = plt.subplots(figsize=(10,7))
    plt.errorbar(x_range, medians, yerr=[medians-err_from, err_to-medians],  capsize=5.0, fmt='o',label="NN Reco")
    if use_old_reco:
        plt.errorbar(x_range, medians_reco, yerr=[medians_reco-err_from_reco, err_to_reco-medians_reco], capsize=5.0,fmt='o',label="Pegleg Reco")
        plt.legend(loc="upper right")
    ax.plot([0,number_sets+1], [0,0], color='k')
    ax.set_xlim(0,number_sets+1)
    
    #rename axis
    my_xlabels = [item.get_text() for item in ax.get_xticklabels()]
    new_namelist = [" "] + namelist
    for index in range(0,number_sets+1):
        my_xlabels[index] = new_namelist[index]
    ax.set_xticklabels(my_xlabels)
    
    ax.set_xlabel("Systematic Set")
    if use_fraction:
        ax.set_ylabel("Fractional Resolution: \n (reconstruction - truth)/truth")
    else:
        ax.set_ylabel("Resolution: \n reconstruction - truth (GeV)")
    ax.set_title("Resolution Energy Dependence")
    
    savename = "SystematicResolutionCompare"
    if use_fraction:
        savename += "Frac"
    if use_old_reco:
        savename += "_CompareOldReco"
    if save == True:
        plt.savefig("%s%s.png"%(savefolder,savename))

def plot_energy_slices(truth, nn_reco, \
                       use_fraction = False, use_old_reco = False, old_reco=None,\
                       bins=10,minenergy=0.,maxenergy=60.,\
                       save=False,savefolder=None):
    """Plots different energy slices vs each other (systematic set arrays)
    Receives:
        truth= array with truth labels
                (contents = [energy], shape = number of events)
        nn_reco = array that has NN predicted reco results
                    (contents = [energy], shape = number of events)
        use_fraction = bool, use fractional resolution instead of absolute, where (reco - truth)/truth
        use_old_reco = bool, True if you want to compare to another reconstruction (like pegleg)
        old_reco = optional, array of pegleg labels
                (contents = [energy], shape = number of events)
        bins = integer number of data points you want (range/bins = width)
        minenergy = minimum energy value to start cut at (default = 0.)
        maxenergy = maximum energy value to end cut at (default = 60.)
    Returns:
        Scatter plot with energy values on x axis (median of bin width)
        y axis has median of resolution with error bars containing 68% of resolution
    """
    if use_fraction:
        resolution = ((nn_reco-truth)/truth) # in fraction
    else:
        resolution = (nn_reco-truth)
        
    percentile_in_peak = 68.27

    left_tail_percentile  = (100.-percentile_in_peak)/2
    right_tail_percentile = 100.-left_tail_percentile

    energy_ranges  = numpy.linspace(minenergy,maxenergy, num=bins)
    energy_centers = (energy_ranges[1:] + energy_ranges[:-1])/2.

    medians  = numpy.zeros(len(energy_centers))
    err_from = numpy.zeros(len(energy_centers))
    err_to   = numpy.zeros(len(energy_centers))
    
    if use_old_reco:
        if use_fraction:
            resolution_reco = ((old_reco-truth)/truth)
        else:
            resolution_reco = (old_reco-truth)
        medians_reco  = numpy.zeros(len(energy_centers))
        err_from_reco = numpy.zeros(len(energy_centers))
        err_to_reco   = numpy.zeros(len(energy_centers))


    for i in range(len(energy_ranges)-1):
        en_from = energy_ranges[i]
        en_to   = energy_ranges[i+1]

        cut = (truth >= en_from) & (truth < en_to)

        lower_lim = numpy.percentile(resolution[cut], left_tail_percentile)
        upper_lim = numpy.percentile(resolution[cut], right_tail_percentile)
        median = numpy.percentile(resolution[cut], 50.)

        medians[i] = median
        err_from[i] = lower_lim
        err_to[i] = upper_lim
        
        if use_old_reco:
            lower_lim_reco = numpy.percentile(resolution_reco[cut], left_tail_percentile)
            upper_lim_reco = numpy.percentile(resolution_reco[cut], right_tail_percentile)
            median_reco = numpy.percentile(resolution_reco[cut], 50.)
            
            medians_reco[i] = median_reco
            err_from_reco[i] = lower_lim_reco
            err_to_reco[i] = upper_lim_reco

    plt.figure(figsize=(10,7))
    plt.errorbar(energy_centers, medians, yerr=[medians-err_from, err_to-medians], xerr=[ energy_centers-energy_ranges[:-1], energy_ranges[1:]-energy_centers ], capsize=5.0, fmt='o',label="NN Reco")
    if use_old_reco:
        plt.errorbar(energy_centers, medians_reco, yerr=[medians_reco-err_from_reco, err_to_reco-medians_reco], xerr=[ energy_centers-energy_ranges[:-1], energy_ranges[1:]-energy_centers ], capsize=5.0, fmt='o',label="Pegleg Reco")
        plt.legend(loc="upper center")
    plt.plot([minenergy,maxenergy], [0,0], color='k')
    plt.xlim(minenergy,maxenergy)
    plt.xlabel("Energy Range (GeV)")
    if use_fraction:
        plt.ylabel("Fractional Resolution: \n (reconstruction - truth)/truth")
    else:
         plt.ylabel("Resolution: \n reconstruction - truth (GeV)")
    plt.title("Resolution Energy Dependence")

    savename = "EnergyResolutionSlices"
    if use_fraction:
        savename += "Frac"
    if use_old_reco:
        savename += "_CompareOldReco"
    if save == True:
        plt.savefig("%s%s.png"%(savefolder,savename))

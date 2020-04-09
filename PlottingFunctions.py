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
import itertools

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

def find_contours_2D(x_values,y_values,xbins,c1=16,c2=84):   
    """
    Find upper and lower contours and median
    x_values = array, input for hist2d for x axis (typically truth)
    y_values = array, input for hist2d for y axis (typically reconstruction)
    xbins = values for the starting edge of the x bins (output from hist2d)
    c1 = percentage for lower contour bound (16% - 84% means a 68% band, so c1 = 16)
    c2 = percentage for upper contour bound (16% - 84% means a 68% band, so c2=84)
    Returns:
        x = values for xbins, repeated for plotting (i.e. [0,0,1,1,2,2,...]
        y_median = values for y value medians per bin, repeated for plotting (i.e. [40,40,20,20,50,50,...]
        y_lower = values for y value lower limits per bin, repeated for plotting (i.e. [30,30,10,10,20,20,...]
        y_upper = values for y value upper limits per bin, repeated for plotting (i.e. [50,50,40,40,60,60,...]
    """
    y_values = numpy.array(y_values)
    indices = numpy.digitize(x_values,xbins)
    r1_save = []
    r2_save = []
    median_save = []
    for i in range(1,len(xbins)):
        mask = indices==i
        if len(y_values[mask])>0:
            r1, m, r2 = numpy.percentile(y_values[mask],[c1,50,c2])
        else:
            print(i,'empty bin')
            r1 = 0
            m = 0
            r2 = 0
        median_save.append(m)
        r1_save.append(r1)
        r2_save.append(r2)
    median = numpy.array(median_save)
    lower = numpy.array(r1_save)
    upper = numpy.array(r2_save)

    x = list(itertools.chain(*zip(xbins[:-1],xbins[1:])))
    y_median = list(itertools.chain(*zip(median,median)))
    y_lower = list(itertools.chain(*zip(lower,lower)))
    y_upper = list(itertools.chain(*zip(upper,upper)))
    
    return x, y_median, y_lower, y_upper



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

def plot_history_from_list(loss,val,save=False,savefolder=None,logscale=False,ymin=None,ymax=None,title=None):
    
    plt.figure(figsize=(10,7))
    plt.plot(loss,label="Training")
    plt.plot(val,label="Validation")
   
    #Edit Axis
    if logscale:
        plt.yscale('log')
    if ymin and ymax:
        plt.ylim(ymin,ymax)
    elif ymin:
        plt.ylim(ymin,max(max(loss),max(val)))
    elif ymax:
        plt.ylim(min(min(loss),min(val)),ymax)
    
    #Add labels
    if title:
        plt.title(title,fontsize=25)
    else:
        plt.title("Training and Validation Loss after %s Epochs"%len(loss),fontsize=25)
    plt.xlabel('Epochs',fontsize=15)
    plt.ylabel('Loss',fontsize=15)
    plt.legend(fontsize=15)
    
    if save == True:
        plt.savefig("%sloss_vs_epochs.png"%savefolder) 


def plot_history_from_list_split(energy_loss,val_energy_loss,zenith_loss,val_zenith_loss,save=True,savefolder=None,logscale=False,ymin=None,ymax=None,title=None):
    
    plt.figure(figsize=(10,7))
    plt.plot(energy_loss,'b',label="Energy Training")
    plt.plot(val_energy_loss,'c',label="Energy Validation")
    plt.plot(zenith_loss,'r',label="Zenith Training")
    plt.plot(val_zenith_loss,'m',label="Zenith Validation")
    
    #Edit Axis
    if logscale:
        plt.yscale('log')
    if ymin and ymax:
        plt.ylim(ymin,ymax)
    elif ymin:
        plt.ylim(ymin,max(max(loss),max(val)))
    elif ymax:
        plt.ylim(min(min(loss),min(val)),ymax)
    
    #Add labels
    if title:
        plt.title(title,fontsize=25)
    else:
        plt.title("Training and Validation Loss after %s Epochs"%len(energy_loss),fontsize=25)
    plt.xlabel('Epochs',fontsize=15)
    plt.ylabel('Loss',fontsize=15)
    plt.legend(fontsize=15)
    
    if save == True:
        plt.savefig("%sloss_vs_epochs_split.png"%savefolder)

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
    plt.title("True Energy Distribution",fontsize=25)
    plt.hist(truth[CC_mask], bins=100,color='b',alpha=0.5,label="CC");
    plt.hist(truth[NC_mask], bins=100,color='g',alpha=0.5,label="NC");
    plt.xlabel("Energy (GeV)",fontsize=15)
    plt.legend(fontsize=10)
    if save:
        plt.savefig("%sTrueEnergyDistribution_CCNC.png"%savefolder)

    plt.figure(figsize=(10,7))
    plt.title("NN Energy Distribution",fontsize=25)
    plt.hist(reco[CC_mask], bins=100,color='b', alpha=0.5, label="CC");
    plt.hist(reco[NC_mask], bins=100,color='g', alpha=0.5, label="NC");
    plt.xlabel("Energy (GeV)",fontsize=15)
    plt.legend(fontsize=10)
    if save:
        plt.savefig("%sNNEnergyDistribution_CCNC.png"%savefolder)

def plot_distributions(truth,reco,save=False,savefolder=None,variable="Energy",units="GeV"):
    """
    Plot testing set distribution
    Recieves:
        truth = array, Y_test truth labels
        reco = array, neural network prediction output
        save = optional, bool to save plot
        savefolder = optional, output folder to save to, if not in current dir
        variable = string, variable name
        units = string, units for variable
    Returns:
        1D histogram of variable's absolute distribution for truth and for reco overlaid
    """
    max_range = numpy.max([numpy.max(truth),numpy.max(reco)])
    min_range = numpy.min([numpy.min(truth),numpy.min(reco)])
    plt.figure(figsize=(10,7))
    plt.title("%s Distribution"%variable,fontsize=25)
    plt.hist(truth, bins=100,color='b',alpha=0.5,range=[min_range,max_range],label="Truth");
    plt.hist(reco, bins=100,color='g', alpha=0.5,range=[min_range,max_range],label="NN Reco");
    plt.xlabel("%s (%s)"%(variable,units),fontsize=15)
    plt.legend(fontsize=10)
    if save:
        plt.savefig("%s%sDistribution.png"%(savefolder,variable))

def plot_2D_prediction(truth, nn_reco, \
                        save=False,savefolder=None,syst_set="",\
                        bins=60,minval=None,maxval=None,ylim=None,\
                        cut_truth = False, axis_square =False, zmax=None,
                        variable="Energy", units = "GeV", epochs=None):
    """
    Plot testing set reconstruction vs truth
    Recieves:
        truth = array, Y_test truth
        nn_reco = array, neural network prediction output
        save = optional, bool to save plot
        savefolder = optional, output folder to save to, if not in current dir
        syst_set = string, name of the systematic set (for title and saving)
        bins = int, number of bins plot (will use for both the x and y direction)
        minval = float, minimum value to cut nn_reco results
        maxval = float, maximum value to cut nn_reco results
        cut_truth = bool, true if you want to make the value cut on truth rather than nn results
        axis_square = bool, cut axis to be square based on minval and maxval inputs
        variable = string, name of the variable you are plotting
        units = string, units for the variable you are plotting
    Returns:
        2D plot of True vs Reco
    """

    if cut_truth:

        if not minval:
            minval = min(truth)
        if not maxval:
            maxval= max(truth)
        mask = numpy.logical_and(truth >= minval, truth <= maxval)
        name = "True %s [%.2f, %.2f]"%(variable,minval,maxval)

    else:
        if not minval:
            minval = min([min(nn_reco),min(truth)])
        if not maxval:
            maxval= max([max(nn_reco),max(truth)])
        mask = numpy.logical_and(nn_reco >= minval, nn_reco <= maxval)
        name = "NN %s [%.2f, %.2f]"%(variable,minval,maxval)

    #Check if cutting
    cutting = False
    if sum(mask)!= len(truth):
        overflow = len(nn_reco)-sum(mask)
        print("Making a cut for plotting, removing %i events"%(overflow))
        cutting = True
    maxplotline = min([max(nn_reco),max(truth)])
    minplotline = max([min(nn_reco),min(truth)])
    
    truth = truth[mask]
    nn_reco = nn_reco[mask]
    
    plt.figure(figsize=(10,7))
    cts,xbin,ybin,img = plt.hist2d(truth, nn_reco, bins=bins, cmax=zmax)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('counts', rotation=90)
    plt.set_cmap('viridis_r')
    plt.xlabel("True Neutrino %s (%s)"%(variable,units),fontsize=15)
    plt.ylabel("NN Reconstruction %s (%s)"%(variable,units),fontsize=15)
    title = "CNN vs Truth for %s %s"%(variable,syst_set)
    if epochs:
        title += " at %i Epochs"%epochs
    plt.suptitle(title,fontsize=25)
    if cutting:
        plt.title("%s, plotted %i, overflow %i"%(name,len(truth),overflow),fontsize=15)
    
    #Cut axis
    if axis_square:
        plt.xlim(minval,maxval)
        plt.ylim(minval,maxval)
    else:
        plt.xlim(min(truth),max(truth))
        plt.ylim(min(nn_reco),max(nn_reco))

    if type(ylim) is not None:
        plt.ylim(ylim)
        
    #Plot 1:1 line
    if cutting == True:
        plt.plot([minval,maxval],[minval,maxval],'k:',label="1:1")
    else:
        plt.plot([minplotline,maxplotline],[minplotline,maxplotline],'k:',label="1:1")
    
    x, y, y_l, y_u = find_contours_2D(truth,nn_reco,xbin)
    
    plt.plot(x,y,color='r',label='Median')
    plt.plot(x,y_l,color='r',label='68% band',linestyle='dashed')
    plt.plot(x,y_u,color='r',linestyle='dashed')
    plt.legend(fontsize=12)
    
    if cutting:
        nocut_name = ""
    else:
        nocut_name ="_nolim"
    if zmax:
        nocut_name += "_zmax%i"%zmax    
    if save:
        plt.savefig("%sTruthReco%s_2DHist%s%s.png"%(savefolder,variable,syst_set,nocut_name))

def plot_2D_prediction_fraction(truth, nn_reco, \
                            save=False,savefolder=None,syst_set="",\
                            bins=60,minval=None,maxval=None,\
                            cut_truth = False, axis_square =False,
                            variable="Energy", units = "GeV"):
    """
    Plot testing set reconstruction vs truth
    Recieves:
        truth = array, Y_test truth
        nn_reco = array, neural network prediction output
        save = optional, bool to save plot
        savefolder = optional, output folder to save to, if not in current dir
        syst_set = string, name of the systematic set (for title and saving)
        bins = int, number of bins plot (will use for both the x and y direction)
        minval = float, minimum value to cut (truth - nn_reco)/truth fractional results
        maxval = float, maximum value to cut (truth - nn_reco)/truth fractional results
        cut_truth = bool, true if you want to make the value cut on truth rather than nn results
        variable = string, name of the variable you are plotting
        units = string, units for the variable you are plotting
    Returns:
        2D plot of True vs (True - Reco)/True
    """

    fractional_error = abs(truth - nn_reco)/ truth
    
    if cut_truth:
        if not minval:
            minval = min(truth)
        if not maxval:
            maxval= max(truth)
        mask = numpy.logical_and(truth >= minval, truth <= maxval)
        name = "True %s [%.2f, %.2f]"%(variable,minval,maxval)

    else:
        if not minval:
            minval = min(fractional_error)
        if not maxval:
            maxval= max(fractional_error)
        mask = numpy.logical_and(fractional_error >= minval, fractional_error <= maxval)
        name = "NN %s in Fractional Error [%.2f, %.2f]"%(variable,minval,maxval)

    #Check if cutting
    cutting = False
    if sum(mask)!= len(truth):
        overflow = len(nn_reco)-sum(mask)
        print("Making a cut for plotting, removing %i events"%(overflow))
        cutting = True
    maxplotline = min([max(nn_reco),max(truth)])
    minplotline = max([min(nn_reco),min(truth)])
    
    truth = truth[mask]
    fractional_error = fractional_error[mask]
    
    plt.figure(figsize=(10,7))

    cts,xbin,ybin,img = plt.hist2d(truth, fractional_error, bins=bins)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('counts', rotation=90)
    plt.set_cmap('viridis_r')
    #plt.xlim(min(truth),max(truth))
    #plt.ylim(min(fractional_truth),max(fractional_truth))
    
    plt.xlabel("True Neutrino %s (%s)"%(variable,units),fontsize=15)
    plt.ylabel("Fractional Error",fontsize=15)
    plt.suptitle("NN Fractional Error vs. True %s %s"%(variable,syst_set),fontsize=25)
    if cutting:
        plt.title("%s, plotted %i, overflow %i"%(name,len(truth),overflow),fontsize=15)
        
    #Plot 1:1 line
    #if cutting == True:
    #    plt.plot([minval,maxval],[minval,maxval],'k:',label="1:1")
    #else:
    #    plt.plot([minplotline,maxplotline],[minplotline,maxplotline],'k:',label="1:1")
    
    x, y, y_l, y_u = find_contours_2D(truth,fractional_error,xbin)
    plt.plot(x,y,color='r',label='Median')
    plt.plot(x,y_l,color='r',label='68% band',linestyle='dashed')
    plt.plot(x,y_u,color='r',linestyle='dashed')
    plt.legend(fontsize=12)
    
    if cutting:
        nocut_name = ""
    else:
        nocut_name ="_nolim"
    if save:
        plt.savefig("%sTruthRecoFrac%s_2DHist%s%s.png"%(savefolder,variable,syst_set,nocut_name))

def plot_resolution_CCNC(truth_all_labels,truth,reco,save=False,savefolder=None,variable="Energy", units = "GeV"):
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
    plt.title("%s Resolution"%variable)
    plt.hist(resolution[CC_mask], bins=50,color='b', alpha=0.5, label="CC");
    plt.hist(resolution[NC_mask], bins=50,color='g', alpha=0.5, label="NC");
    plt.xlabel("NN reconstruction - truth (%s)"%units)
    plt.legend()
    if save:
        plt.savefig("%s%sResolution_CCNC.png"%(savefolder,variable))

    plt.figure(figsize=(10,7))  
    plt.title("Fractional %s Resolution"%variable)
    plt.hist(resolution_fraction[CC_mask], bins=50,color='b', alpha=0.5, label="CC");
    plt.hist(resolution_fraction[NC_mask], bins=50,color='g', alpha=0.5, label="NC");
    plt.xlabel("(NN reconstruction - truth) / truth")
    plt.legend()
    if save:
        plt.savefig("%s%sResolutionFrac_CCNC.png"%(savefolder,variable))

def plot_single_resolution(truth,nn_reco,\
                           bins=100, use_fraction=False,\
                           use_old_reco = False, old_reco=None,\
                           mintrue=None,maxtrue=None,\
                           minaxis=None,maxaxis=None,\
                           save=False,savefolder=None,
                           variable="Energy", units = "GeV", epochs=None):
    """Plots resolution for dict of inputs, one of which will be a second reco
    Recieves:
        truth = array of truth or Y_test labels
        nn_reco = array of NN predicted reco or Y_test_predicted results
        bins = int value
        use_fraction = use fractional resolution instead of absolute, where (reco - truth)/truth
        use_old_reco = True if you want to compare to another reconstruction (like pegleg)
        old_reco = optional, pegleg array of labels
        mintrue = float, min true value if cut desired
        maxtrue = float, max true value if cut desired
        minaxis = float, min x axis cut
        maxaxis = float, max x axis cut
    Returns:
        1D histogram of Reco - True (or fractional)
        Can have two distributions of NN Reco Resolution vs Pegleg Reco Resolution
    """
    
    fig, ax = plt.subplots(figsize=(10,7)) 
    
    if use_fraction:
        nn_resolution = (nn_reco - truth)/truth
        if use_old_reco:
            old_reco_resolution = (old_reco - truth)/truth
        title = "Fractional %s Resolution"%variable
        xlabel = "(reconstruction - truth) / truth" 
    else:
        nn_resolution = nn_reco - truth
        if use_old_reco:
            old_reco_resolution = old_reco - truth
        title = "%s Resolution"%variable
        xlabel = "reconstruction - truth (%s)"%units
    if epochs:
        title += " at %i Epochs"%epochs
    original_size = len(nn_resolution)
    
    # Cut on true values
    #print(mintrue,maxtrue)
    if mintrue or maxtrue:
         truth_cut=True
    else:
         truth_cut=False
    if not mintrue:
        mintrue = int(min(truth))
    if not maxtrue:
        maxtrue = int(max(truth))+1
    #print(mintrue,maxtrue,truth_cut)
    if truth_cut:
        true_mask = numpy.logical_and(truth >= mintrue, truth <= maxtrue)
        title += "\n(true %s [%.2f, %.2f])"%(variable,mintrue,maxtrue)
        nn_resolution = nn_resolution[true_mask]
        if use_old_reco:
            old_reco_resolution = old_reco_resolution[truth_mask]
    true_cut_size=len(nn_resolution)
    
    # Cut for plot axis
    #print(minaxis,maxaxis)
    if minaxis or maxaxis:
        axis_cut=True
    else:
        axis_cut=False
    if not minaxis:
        axis_cut=True
        minaxis = min(nn_resolution)
    if not maxaxis:
        axis_cut=True
        maxaxis = max(nn_resolution)
    if axis_cut:
        axis_mask = numpy.logical_and(nn_resolution >= minaxis, nn_resolution <= maxaxis)
        nn_resolution = nn_resolution[axis_mask]
        if use_old_reco:
            old_reco_resolution = old_reco_resolution[axis_mask]
    true_axis_cut_size=len(nn_resolution)
    #print(minaxis,maxaxis,axis_cut)
 
    ax.hist(nn_resolution, bins=bins, alpha=0.5, label="neural net");
    
    #Statistics
    rms_nn = get_RMS(nn_resolution)
    r1, r2 = numpy.percentile(nn_resolution, [16,84])
    textstr = '\n'.join((
            r'$\mathrm{events}=%i$' % (len(nn_resolution), ),
            r'$\mathrm{median}=%.2f$' % (numpy.median(nn_resolution), ),
            r'$\mathrm{overflow}=%i$' % (true_cut_size-true_axis_cut_size, ),
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
            r'$\mathrm{events}=%i$' % (len(old_reco_resolution), ),
            r'$\mathrm{median}=%.2f$' % (numpy.median(old_reco_resolution), ),
            r'$\mathrm{RMS}=%.2f$' % (rms_old_reco, ),
            r'$\mathrm{1\sigma}=%.2f,%.2f$' % (r1_old_reco,r2_old_reco )))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.2, 0.95, textstr, transform=ax.transAxes, fontsize=20,
            verticalalignment='top', bbox=props)
   
    #if axis_cut:
    ax.set_xlim(minaxis,maxaxis)
    ax.set_xlabel(xlabel,fontsize=15)
    ax.set_title(title,fontsize=25)
    
    savename = "%sResolution"%variable
    if use_fraction:
        savename += "Frac"
    if truth_cut:
        savename = "_Range%.2f_%.2f"%(mintrue,maxtrue)
    if use_old_reco:
        savename += "_CompareOldReco"
    if axis_cut:
        savename += "_xlim"
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

def plot_bin_slices(truth, nn_reco, energy_truth=None, \
                       use_fraction = False, old_reco=None,\
                       bins=10,min_val=0.,max_val=60., ylim = None,\
                       save=False,savefolder=None,\
                       variable="Energy",units="GeV",epochs=None):
    """Plots different variable slices vs each other (systematic set arrays)
    Receives:
        truth= array with truth labels for this one variable
        nn_reco = array that has NN predicted reco results for one variable (same size of truth)
        energy_truth = optional (will use if given), array that has true energy information (same size of truth)
        use_fraction = bool, use fractional resolution instead of absolute, where (reco - truth)/truth
        old_reco = optional (will use if given), array of pegleg labels for one variable
        bins = integer number of data points you want (range/bins = width)
        min_val = minimum value for variable to start cut at (default = 0.)
        max_val = maximum value for variable to end cut at (default = 60.)
        ylim = List with two entries of ymin and ymax for plot [min, max], leave as None for no ylim applied
    Returns:
        Scatter plot with energy values on x axis (median of bin width)
        y axis has median of resolution with error bars containing 68% of resolution
    """
    nn_reco = numpy.array(nn_reco)
    truth = numpy.array(truth)
    
    if use_fraction:
        resolution = ((nn_reco-truth)/truth) # in fraction
    else:
        resolution = (nn_reco-truth)
    resolution = numpy.array(resolution)
    percentile_in_peak = 68.27

    left_tail_percentile  = (100.-percentile_in_peak)/2
    right_tail_percentile = 100.-left_tail_percentile

    variable_ranges  = numpy.linspace(min_val,max_val, num=bins+1)
    variable_centers = (variable_ranges[1:] + variable_ranges[:-1])/2.

    medians  = numpy.zeros(len(variable_centers))
    err_from = numpy.zeros(len(variable_centers))
    err_to   = numpy.zeros(len(variable_centers))

    if old_reco is not None:
        if use_fraction:
            resolution_reco = ((old_reco-truth)/truth)
        else:
            resolution_reco = (old_reco-truth)
        resolution_reco = numpy.array(resolution_reco)
        medians_reco  = numpy.zeros(len(variable_centers))
        err_from_reco = numpy.zeros(len(variable_centers))
        err_to_reco   = numpy.zeros(len(variable_centers))

    for i in range(len(variable_ranges)-1):
        var_from = variable_ranges[i]
        var_to   = variable_ranges[i+1]
            
        if energy_truth is None:
            cut = (truth >= var_from) & (truth < var_to)
        else:
            "Using energy for x-axis. Make sure your min_val and max_val are in terms of energy!"
            energy_truth = numpy.array(energy_truth)
            cut = (energy_truth >= var_from) & (energy_truth < var_to) 

        lower_lim = numpy.percentile(resolution[cut], left_tail_percentile)
        upper_lim = numpy.percentile(resolution[cut], right_tail_percentile)
        median = numpy.percentile(resolution[cut], 50.)
        
        medians[i] = median
        err_from[i] = lower_lim
        err_to[i] = upper_lim

        if old_reco is not None:
            lower_lim_reco = numpy.percentile(resolution_reco[cut], left_tail_percentile)
            upper_lim_reco = numpy.percentile(resolution_reco[cut], right_tail_percentile)
            median_reco = numpy.percentile(resolution_reco[cut], 50.)

            medians_reco[i] = median_reco
            err_from_reco[i] = lower_lim_reco
            err_to_reco[i] = upper_lim_reco

    plt.figure(figsize=(10,7))
    plt.errorbar(variable_centers, medians, yerr=[medians-err_from, err_to-medians], xerr=[ variable_centers-variable_ranges[:-1], variable_ranges[1:]-variable_centers ], capsize=5.0, fmt='o',label="NN Reco")
    if old_reco is not None:
        plt.errorbar(variable_centers, medians_reco, yerr=[medians_reco-err_from_reco, err_to_reco-medians_reco], xerr=[ variable_centers-variable_ranges[:-1], variable_ranges[1:]-variable_centers ], capsize=5.0, fmt='o',label="Pegleg Reco")
        plt.legend(loc="upper center")
    plt.plot([min_val,max_val], [0,0], color='k')
    plt.xlim(min_val,max_val)
    if type(ylim) is not None:
        plt.ylim(ylim)
    plt.xlabel("True %s (%s)"%(variable,units),fontsize=15)
    if use_fraction:
        plt.ylabel("Fractional Resolution: \n (reconstruction - truth)/truth",fontsize=15)
    else:
         plt.ylabel("Resolution: \n reconstruction - truth (%s)"%units,fontsize=15)
    if energy_truth is None:
        title = "%s Resolution, %s Dependence"%(variable,variable)
    else:
        title = "%s Resolution, Energy Dependence"%(variable)
    if epochs:
        title += " at %i Epochs"%epochs
    plt.title(title,fontsize=25)

    savename = "%sResolutionSlices"%variable
    if use_fraction:
        savename += "Frac"
    if energy_truth is not None:
        savename += "_EnergyBinned"
        plt.xlabel("True Energy (GeV)",fontsize=15)
    if old_reco is not None:
        savename += "_CompareOldReco"
    if type(ylim) is not None:
        savename += "_ylim"
    if save == True:
        plt.savefig("%s%s.png"%(savefolder,savename))

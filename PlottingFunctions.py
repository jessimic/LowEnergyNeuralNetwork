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
import matplotlib.colors as colors
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

def get_RMS(resolution,weights=None):
    if weights is not None:
        import wquantiles as wq

    mean_array = numpy.ones_like(resolution)*numpy.mean(resolution)
    if weights is None:
        rms = numpy.sqrt( sum((mean_array - resolution)**2)/len(resolution) )
    else:
        rms = numpy.zeros_like(resolution)
        rms = numpy.sqrt( sum(weights*(mean_array - resolution)**2)/sum(weights) )
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

def find_contours_2D(x_values,y_values,xbins,weights=None,c1=16,c2=84):   
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
    if weights is not None:
        import wquantiles as wq
    y_values = numpy.array(y_values)
    indices = numpy.digitize(x_values,xbins)
    r1_save = []
    r2_save = []
    median_save = []
    for i in range(1,len(xbins)):
        mask = indices==i
        if len(y_values[mask])>0:
            if weights is None:
                r1, m, r2 = numpy.percentile(y_values[mask],[c1,50,c2])
            else:
                r1 = wq.quantile(y_values[mask],weights[mask],c1/100.)
                r2 = wq.quantile(y_values[mask],weights[mask],c2/100.)
                m = wq.median(y_values[mask],weights[mask])
        else:
            #print(i,'empty bin')
            r1 = numpy.nan
            m = numpy.nan
            r2 = numpy.nan
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

def plot_history_from_list(loss,val,save=False,savefolder=None,logscale=False,ymin=None,ymax=None,title=None,variable="Energy",pick_epoch=None,lr_start=None,lr_drop=None,lr_epoch=None,step=1,notebook=False,xline=None,xline_label="Changed Model"):
    
    fig,ax = plt.subplots(figsize=(10,7))
    start=step
    end=len(loss)*step
    epochs = numpy.arange(start,end+(step/2.),step)
    ax.plot(epochs,loss,'b',label="Training")
    ax.plot(epochs,val,'c',label="Validation")
   
    #Edit Axis
    if logscale:
        ax.set_yscale('log')
    if ymin and ymax:
        pass
    elif ymin:
        ymax = max(max(loss),max(val))
    elif ymax:
        ymin = min(min(loss),min(val))
    else:
        ymax = max(max(loss),max(val))
        ymin = min(min(loss),min(val))
    ax.set_ylim(ymin,ymax)
    
    if xline is not None:
        ax.axvline(int(xline)*step, linewidth=4, color='k', alpha=0.8, label="%s"%xline_label)

    if pick_epoch is not None:
        ax.axvline(pick_epoch*step,linewidth=4, color='g',alpha=0.5,label="Chosen Model")

    if lr_epoch is not None:
        #epoch_drop = numpy.arange(396*step,end,lr_epoch*step)
        epoch_drop = numpy.arange(0,end,lr_epoch*step)
        for lr_print in range(len(epoch_drop)):
            lrate = lr_start*(lr_drop**lr_print)
            ax.axvline(epoch_drop[lr_print],linewidth=1, color='r',linestyle="--")
            ax.annotate(s='lrate='+str("{:.0e}".format(lrate)),xy=(epoch_drop[lr_print]+step,ymax),rotation=90,verticalalignment='top')

    #Add labels
    if title:
        plt.title(title,fontsize=25)
    else:
        plt.title("Loss for %s CNN"%variable,fontsize=25)
    
    plt.xlabel('Epochs',fontsize=20)
    if variable=="Energy":
        plt.ylabel(r'Loss = $\frac{100}{n}\sum_{i=1}^n \vert \frac{T_i - R_i}{T_i} \vert$',fontsize=20)
    elif variable=="Cosine Zenith":
        plt.ylabel(r'Loss = $\frac{1}{n}\sum_{i=1}^n ( T_i - R_i )^2$',fontsize=20)
    else:
        plt.ylabel('Loss',fontsize=20)
    plt.legend(loc="center right",fontsize=20)
    


    if save == True:
        plt.savefig("%sloss_vs_epochs.png"%savefolder,bbox_inches='tight') 
    if not notebook:
        plt.close()


def plot_history_from_list_split(energy_loss,val_energy_loss,zenith_loss,val_zenith_loss,save=True,savefolder=None,logscale=False,ymin=None,ymax=None,title=None,notebook=False):
    
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
    plt.xlabel('Epochs',fontsize=20)
    plt.ylabel('Loss',fontsize=20)
    plt.legend(fontsize=20)
    
    if save == True:
        plt.savefig("%sloss_vs_epochs_split.png"%savefolder)
    if not notebook:
        plt.close()

def plot_distributions_CCNC(truth_all_labels,truth,reco,save=False,savefolder=None,notebook=False):
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
    plt.xlabel("Energy (GeV)",fontsize=20)
    plt.legend(fontsize=10)
    if save:
        plt.savefig("%sTrueEnergyDistribution_CCNC.png"%savefolder)

    plt.figure(figsize=(10,7))
    plt.title("NN Energy Distribution",fontsize=25)
    plt.hist(reco[CC_mask], bins=100,color='b', alpha=0.5, label="CC");
    plt.hist(reco[NC_mask], bins=100,color='g', alpha=0.5, label="NC");
    plt.xlabel("Energy (GeV)",fontsize=20)
    plt.legend(fontsize=10)
    if save:
        plt.savefig("%sNNEnergyDistribution_CCNC.png"%savefolder)
    if not notebook:
        plt.close()

def plot_distributions(truth,reco=None,save=False,savefolder=None,old_reco=None,weights=None,variable="Energy",units="(GeV)",reco_name="Retro", minval=None, maxval=None,bins=100,cnn_name="CNN",ylog=False,xlog=False,old_reco_weights=None,title=None,xline=None,xline_label=None,flavor=None,sample=None,notebook=False,true_name="Truth"):
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

    if maxval is None:
        if reco is not None:
            if old_reco is None:
                maxval = numpy.max([numpy.max(truth),numpy.max(reco)])
            else:
                maxval = numpy.max([numpy.max([numpy.max(truth),numpy.max(reco)]),numpy.max(old_reco)])
        else:
            maxval = numpy.max(truth)
    if minval is None:
        if reco is not None:
            if old_reco is None:
                minval = numpy.min([numpy.min(truth),numpy.min(reco)])
            else:
                minval = numpy.min([numpy.min([numpy.min(truth),numpy.min(reco)]),numpy.min(old_reco)])
        else:
            minval = numpy.min(truth)
    print("Using", minval, maxval)
    
    plt.figure(figsize=(10,7))
    outname = ""
    if weights is not None:
        if old_reco_weights is None:
            old_reco_weights = weights
        #name += "Weighted"
        weights_factor = 1e7
    if title is not None:
        plt.title("%s"%(title),fontsize=25)
    else:
        name = "%s Distribution"%(variable)
        if flavor is not None:
            if flavor == "NuMu" or flavor == "numu":
                name += r' for $\nu_\mu$ '
            elif flavor == "NuE" or flavor == "nue":
                name += r' for $\nu_e$ '
            elif flavor == "NuTau" or flavor == "nutau":
                name += r' for $\nu_\tau$ '
            elif flavor == "Mu" or flavor == "mu":
                name += r' for $\mu$ '
            elif flavor == "Nu" or flavor == "nu":
                name += r' for $\nu$ '
            else:
                name += flavor
            name += sample
        plt.title(name,fontsize=25)

    if xlog:
        if minval <=0:
            print("MINVAL AT OR BELOW ZERO, USING ZERO FOR LOG SCALE")
            logmin = 0
        else:
            logmin = numpy.log(minval)
        number_bins = bins
        bins = 10**numpy.linspace(logmin, numpy.log10(maxval),number_bins)
        plt.xscale('log')
    plt.hist(truth, bins=bins,color='g',alpha=0.5,range=[minval,maxval],weights=weights,label=true_name);
    maskT = numpy.logical_and(truth > minval, truth < maxval)
    print("%s Total: %i, Events in Plot: %i, Overflow: %i"%(true_name,len(truth),sum(maskT),len(truth)-sum(maskT)))
    if weights is not None:
        print("WEIGHTED Truth Total: %.2f, Events in Plot: %.2f, Overflow: %.2f"%(sum(weights)*weights_factor,sum(weights[maskT])*weights_factor,(sum(weights)-sum(weights[maskT]))*weights_factor))
        plt.ylabel("weighted event count")
    else:
        plt.ylabel("event count")
    outname += "T"
    
    if reco is not None:
        plt.hist(reco, bins=bins,color='b', alpha=0.5,range=[minval,maxval],weights=weights,label=cnn_name);
        outname += "R"
        maskR = numpy.logical_and(reco > minval, reco < maxval)
        print("Reco Total: %i, Events in Plot: %i, Overflow: %i"%(len(reco),sum(maskR),len(reco)-sum(maskR)))
        if weights is not None:
            print("WEIGHTED Reco Total: %.2f, Events in Plot: %.2f, Overflow: %.2f"%(sum(weights)*weights_factor,sum(weights[maskR])*weights_factor,(sum(weights)-sum(weights[maskR]))*weights_factor))
    if old_reco is not None:
        plt.hist(old_reco, bins=bins,color='orange', alpha=0.5,range=[minval,maxval],weights=old_reco_weights,label=reco_name);
        outname += "OR"
        maskOR = numpy.logical_and(old_reco > minval, old_reco < maxval)
        print("Old Reco Total: %i, Events in Plot: %i, Overflow: %i"%(len(old_reco),sum(maskOR),len(old_reco)-sum(maskOR)))
        if weights is not None:
            print("WEIGHTED Old Reco Total: %.2f, Events in Plot: %.2f, Overflow: %.2f"%(sum(old_reco_weights)*weights_factor,sum(old_reco_weights[maskOR])*weights_factor,(sum(old_reco_weights)-sum(old_reco_weights[maskOR]))*weights_factor))
    plt.xlabel("%s %s"%(variable,units),fontsize=20)
    if ylog:
        plt.yscale("log")
    if xline is not None:
        plt.axvline(xline,linewidth=3,color='k',linestyle="-",label="%s"%xline_label)
    if reco is not None or old_reco is not None or xline is not None:
        plt.legend(fontsize=20)

    if title is not None:
        outname += "%s"%title.replace(" ", "")
    outname += "%s"%variable.replace(" ","")
    if flavor is not None:
        outname += "%s"%flavor.replace(" ","")
    if sample is not None:
        outname += "%s"%sample.replace(" ","")
    if save:
        plt.savefig("%s%sDistribution_%ito%i.png"%(savefolder,outname,int(minval),int(maxval)),bbox_inches='tight')
    if not notebook:
        plt.close()


def plot_2D_prediction(truth, nn_reco, \
                        save=False,savefolder=None,weights=None,syst_set="",\
                        bins=60,minval=None,maxval=None, switch_axis=False,\
                        cut_truth = False, axis_square =False,
                        zmin = None, zmax=None,log=True,
                        variable="Energy", units = "(GeV)", epochs=None,\
                        flavor="NuMu", sample=None,\
                        variable_type="True", reco_name="CNN",new_labels=None,
                        new_units=None,save_name=None,no_contours=False,
                        xline=None,yline=None,notebook=False):
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

    maxplotline = min([max(nn_reco),max(truth)])
    minplotline = max([min(nn_reco),min(truth)])
   
    truth = truth #[mask]
    nn_reco = nn_reco #[mask]
   
    #Cut axis
    if axis_square:
        xmin = minval
        ymin = minval
        xmax = maxval
        ymax = maxval
    else:
        xmin = min(truth)
        ymin = min(nn_reco)
        xmax = max(truth)
        ymax = max(nn_reco)
    if switch_axis:
        xmin, ymin = ymin, xmin
        xmax, ymax = ymax, xmax


    if weights is None:
        cmin = 1
    else:
        cmin = 1e-12
    if zmin is not None:
        cmin = zmin
 
    plt.figure(figsize=(10,7))
    if log:
        if switch_axis:
            cts,xbin,ybin,img = plt.hist2d(nn_reco, truth, bins=bins,range=[[xmin,xmax],[ymin,ymax]], cmap='viridis_r', norm=colors.LogNorm(), weights=weights, cmax=zmax, cmin=cmin)
        else:
            cts,xbin,ybin,img = plt.hist2d(truth, nn_reco, bins=bins,range=[[xmin,xmax],[ymin,ymax]],cmap='viridis_r', norm=colors.LogNorm(), weights=weights, cmax=zmax, cmin=cmin)
    else:
        if switch_axis:
            cts,xbin,ybin,img = plt.hist2d(nn_reco, truth, bins=bins,range=[[xmin,xmax],[ymin,ymax]],cmap='viridis_r', weights=weights, cmax=zmax, cmin=cmin)
        else:
            cts,xbin,ybin,img = plt.hist2d(truth, nn_reco, bins=bins,range=[[xmin,xmax],[ymin,ymax]],cmap='viridis_r', weights=weights, cmax=zmax, cmin=cmin)
    cbar = plt.colorbar()
    if weights is None:
        cbar.ax.set_ylabel('counts', rotation=90)
    else:
        cbar.ax.set_ylabel('Rate (Hz)', rotation=90)
    plt.xlabel("%s %s %s"%(variable_type,variable,units),fontsize=20)
    plt.ylabel("%s Reconstructed %s %s"%(reco_name,variable,units),fontsize=20)
    if switch_axis:
        plt.ylabel("%s %s %s"%(variable_type,variable,units),fontsize=20)
        plt.xlabel("%s Reconstructed %s %s"%(reco_name,variable,units),fontsize=20)
    if new_labels is not None:
        plt.ylabel("%s %s"%(new_labels[0],new_units[0]),fontsize=20)
        plt.xlabel("%s %s"%(new_labels[1],new_units[1]),fontsize=20)
    
    #NAMING
    title = "%s vs %s for %s %s"%(reco_name,variable_type,variable,syst_set)
    if flavor == "NuMu" or flavor == "numu":
        title += r' for $\nu_\mu$ '
    elif flavor == "NuE" or flavor == "nue":
        title += r' for $\nu_e$ '
    elif flavor == "NuTau" or flavor == "nutau":
        title += r' for $\nu_\tau$ '
    elif flavor == "Mu" or flavor == "mu":
        title += r' for $\mu$ '
    elif flavor == "Nu" or flavor == "nu":
        title += r' for $\nu$ '
    else:
        title += flavor
    if sample is not None:
        title += sample
    #if weights is not None:
    #    title += " Weighted"
    if epochs:
        title += " at %i Epochs"%epochs
    plt.suptitle(title,fontsize=25)
    #if cutting:
    #    plt.title("%s, plotted %i, overflow %i"%(name,len(truth),overflow),fontsize=20)
    
    #Plot 1:1 line
    if axis_square:
        plt.plot([minval,maxval],[minval,maxval],'k:',label="1:1")
    else:
        plt.plot([minplotline,maxplotline],[minplotline,maxplotline],'k:',label="1:1")
    
    if switch_axis:
        x, y, y_l, y_u = find_contours_2D(nn_reco,truth,xbin,weights=weights)
    else:
        x, y, y_l, y_u = find_contours_2D(truth,nn_reco,xbin,weights=weights)

    if not no_contours:
        plt.plot(x,y,color='r',label='Median')
        plt.plot(x,y_l,color='r',label='68% band',linestyle='dashed')
        plt.plot(x,y_u,color='r',linestyle='dashed')
        plt.legend(fontsize=20)
    if yline is not None:
        if type(yline) is list:
            for y_val in yline:
                plt.axhline(y_val,linewidth=3,color='red',label="Cut")
        else:
            plt.axhline(yline,linewidth=3,color='red',label="Cut")
        plt.legend(fontsize=20)
    if xline is not None:
        if type(xline) is list:
            for x_val in xline:
                plt.axvline(x_val,linewidth=3,color='magenta',linestyle="dashed",label="Cut")
        else:
            plt.axvline(xline,linewidth=3,color='magenta',linestyle="dashed",label="Cut")
        plt.legend(fontsize=20)

    reco_name = reco_name.replace(" ","")
    variable = variable.replace(" ","")
    variable_type = str(variable_type).replace(" ","")
    nocut_name = ""
    if weights is not None:
        nocut_name="Weighted"
    if flavor is not None:
        nocut_name += "%s"%flavor.replace(" ","")
    if sample is not None:
        nocut_name += "%s"%sample
    if not axis_square:
        nocut_name +="_nolim"
    if zmax:
        nocut_name += "_zmax%.1e"%zmax    
    if zmin:
        nocut_name += "_zmin%.1e"%zmin  
    if switch_axis:
        nocut_name +="_SwitchedAxis"
    if save_name is not None:
        nocut_name += "%s"%save_name.replace(" ","")
    if save:
        plt.savefig("%s%s%sReco%s_2DHist%s%s.png"%(savefolder,variable_type,reco_name,variable,syst_set,nocut_name),bbox_inches='tight')
    if not notebook:
        plt.close()

def plot_2D_prediction_fraction(truth, nn_reco, weights=None,\
                            save=False,savefolder=None,syst_set="",\
                            bins=60,xminval=None,xmaxval=None,notebook=False,\
                            yminval=None,ymaxval=None,log=True,zmax=None,\
                            variable="Energy", units = "(GeV)",reco_name="CNN"):
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
        variable = string, name of the variable you are plotting
        units = string, units for the variable you are plotting
    Returns:
        2D plot of True vs (True - Reco)/True
    """
    
    fractional_error = abs(truth - nn_reco)/ truth
   
    nolim=False
    if xminval is None:
        xminval = min(truth)
    if xmaxval is None:
        nolim = True
        xmaxval = max(truth)
    if yminval is None:
        yminval = min(fractional_error)
    if ymaxval is None:
        ymaxval = max(fractional_error)
    
    if weights is None:
        cmin = 1
    else:
        cmin = 1e-12
    
    plt.figure(figsize=(10,7))

    if log:
        cts,xbin,ybin,img = plt.hist2d(truth, fractional_error, bins=bins,range=[[xminval,xmaxval],[yminval,ymaxval]],cmap='viridis_r', norm=colors.LogNorm(), weights=weights, cmax=zmax, cmin=cmin)
    else:
        cts,xbin,ybin,img = plt.hist2d(truth, fractional_error, bins=bins,range=[[xminval,xmaxval],[yminval,ymaxval]],cmap='viridis_r', weights=weights, cmax=zmax, cmin=cmin)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('counts', rotation=90)
    
    plt.xlabel("True Neutrino %s %s"%(variable,units),fontsize=20)
    plt.ylabel(r'Fractional Resolution: $\frac{reconstruction - truth}{truth}$',fontsize=20)
    plt.title("%s Fractional Error vs. True %s %s"%(reco_name,variable,syst_set),fontsize=25)
    
    x, y, y_l, y_u = find_contours_2D(truth,fractional_error,xbin,weights=weights)
    plt.plot(x,y,color='r',label='Median')
    plt.plot(x,y_l,color='r',label='68% band',linestyle='dashed')
    plt.plot(x,y_u,color='r',linestyle='dashed')
    plt.legend(fontsize=12)
    
    nocut_name = ""
    variable = variable.replace(" ","")
    if weights is not None:
        nocut_name="Weighted"
    if nolim:
        nocut_name ="_nolim"
    if save:
        plt.savefig("%sTruth%sRecoFrac%s_2DHist%s%s.png"%(savefolder,reco_name,variable,syst_set,nocut_name),bbox_inches='tight')

def plot_resolution_CCNC(truth_all_labels,truth,reco,save=False,savefolder=None,variable="Energy", units = "(GeV)",notebook=False):
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

    variable = variable.replace(" ","")
    if save:
        plt.savefig("%s%sResolutionFrac_CCNC.png"%(savefolder,variable))
    if not notebook:
        plt.close()

def plot_single_resolution(truth,nn_reco,weights=None, \
                           bins=100, use_fraction=False,\
                           use_old_reco = False, old_reco=None,\
                           old_reco_truth=None,old_reco_weights=None,\
                           mintrue=None,maxtrue=None,\
                           minaxis=None,maxaxis=None,notebook=False,\
                           save=False,savefolder=None,
                           flavor="NuMu", sample=None,
                           variable="Energy", units = "GeV", epochs=None,
                           reco_name="CNN", old_reco_name="Retro"):
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
    weights_reco = old_reco_weights
    if weights is not None:
        import wquantiles as wq
        if old_reco_weights is None:
            weights_reco = numpy.array(weights)

    fig, ax = plt.subplots(figsize=(10,7))

    ## Assume old_reco truth is the same as test sample, option to set it otherwise
    if old_reco_truth is None:
        truth2 = truth
    else:
        truth2 = old_reco_truth
    # NAN CUT FOR OLD RECO
    #if old_reco is not None:
    #    not_nan = numpy.logical_not(numpy.isnan(old_reco))
    #    if sum(not_nan) != len(not_nan):
    #        print("CUTTING NAN VALUES FROM OLD RECO")
    #    old_reco = old_reco[not_nan]
    #    truth2 = truth2[not_nan]
    #    weights_reco = weights_reco[not_nan]
    #Check nan
    if old_reco is not None:
        is_nan = numpy.isnan(old_reco)
        assert sum(is_nan) == 0, "OLD RECO HAS NAN"
    is_nan = numpy.isnan(nn_reco)
    assert sum(is_nan) == 0, "CNN RECO HAS NAN"
    
    if use_fraction:
        nn_resolution = (nn_reco - truth)/truth
        if use_old_reco:
            old_reco_resolution = (old_reco - truth2)/truth2
        title = "Fractional %s Resolution"%variable
        xlabel = r'$\frac{reconstruction - truth}{truth}$'
    else:
        nn_resolution = nn_reco - truth
        if use_old_reco:
            old_reco_resolution = old_reco - truth2
        title = "%s Resolution"%variable
        xlabel = "reconstruction - truth %s"%units
    if epochs:
        title += " at %i Epochs"%epochs
    if flavor == "NuMu" or flavor == "numu":
        title += r' for $\nu_\mu$ ' 
    elif flavor == "NuE" or flavor == "nue":
        title += r' for $\nu_e$ '
    elif flavor == "NuTau" or flavor == "nutau":
        title += r' for $\nu_\tau$ '
    elif flavor == "Nu" or flavor == "nu":
        title += r' for $\nu$ '
    elif flavor == "Mu" or flavor == "mu":
        title += r' for $\mu$ '
    else:
        title += flavor
    if sample is not None:
        title += sample

    #if weights is not None:
    #    title += " Weighted"
    original_size = len(nn_resolution)
 
    
    #Get stats before axis cut!
    rms_nn = get_RMS(nn_resolution,weights)
    if weights is not None:
        r1 = wq.quantile(nn_resolution,weights,0.16)
        r2 = wq.quantile(nn_resolution,weights,0.84)
        median = wq.median(nn_resolution,weights)
    else:
        r1, r2 = numpy.percentile(nn_resolution, [16,84])
        median = numpy.median(nn_resolution)
    if use_old_reco:
        true_cut_size_reco=len(old_reco_resolution)
        rms_old_reco = get_RMS(old_reco_resolution,weights_reco)
        if weights is not None:
            r1_old_reco = wq.quantile(old_reco_resolution,weights_reco,0.16)
            r2_old_reco = wq.quantile(old_reco_resolution,weights_reco,0.84)
            median_old_reco = wq.median(old_reco_resolution,weights_reco)
        else:
            r1_old_reco, r2_old_reco = numpy.percentile(old_reco_resolution, [16,84])
            median_old_reco = numpy.median(old_reco_resolution)


    # Find cut for plot axis
    #print(minaxis,maxaxis)
    axis_cut = False
    if minaxis or maxaxis:
        axis_cut = True
    if not minaxis:
        minaxis = min(nn_resolution)
        if use_old_reco:
            if minaxis > min(old_reco_resolution):
                minaxis = min(old_reco_resolution)
    if not maxaxis:
        maxaxis = max(nn_resolution)
        if use_old_reco:
            if maxaxis < max(old_reco_resolution):
                maxaxis = max(old_reco_resolution)
    

    hist_nn, bins, p = ax.hist(nn_resolution, bins=bins, range=[minaxis,maxaxis], weights=weights, alpha=0.5, label=reco_name);
    weights_factor = 1 #1e7
    total_events = len(nn_resolution) #sum(weights)
    outside_range = numpy.logical_or(nn_resolution < minaxis, nn_resolution > maxaxis)
    overflow = sum(outside_range) #sum(weights[outside_range])

    #Statistics
    #weighted_avg_and_std(nn_resolution,weights)
    
    textstr = '\n'.join((
            r'%s' % (reco_name),
            r'$\mathrm{events}=%.0f$' % (total_events*weights_factor, ),
            r'$\mathrm{median}=%.2f$' % (median, ),
            r'$\mathrm{overflow}=%.0f$' % (overflow*weights_factor, ),
            #r'$\mathrm{RMS}=%.2f$' % (rms_nn, ),
            r'$\mathrm{1\sigma}=%.2f,%.2f$' % (r1,r2 )))
    props = dict(boxstyle='round', facecolor='blue', alpha=0.3)
    ax.text(0.6, 0.95, textstr, transform=ax.transAxes, fontsize=20,
        verticalalignment='top', bbox=props)

    if use_old_reco:
        ax.hist(old_reco_resolution, bins=bins, range=[minaxis,maxaxis], weights=weights_reco, alpha=0.5, label="%s"%old_reco_name);
        total_events_reco = len(old_reco_resolution) #sum(weights_reco) #len(weights_reco)
        outside_range_reco = numpy.logical_or(old_reco_resolution < minaxis, old_reco_resolution > maxaxis)
        overflow_reco = sum(outside_range_reco) #sum(weights_reco[outside_range_reco]) #sum(outside_range_reco)
        ax.legend(loc="upper left",fontsize=20)
        textstr = '\n'.join((
            '%s' % (old_reco_name),
            r'$\mathrm{events}=%.0f$' % (total_events_reco*weights_factor, ),
            r'$\mathrm{median}=%.2f$' % (median_old_reco, ),
            r'$\mathrm{overflow}=%.0f$' % (overflow_reco*weights_factor, ),
            #r'$\mathrm{RMS}=%.2f$' % (rms_old_reco, ),
            r'$\mathrm{1\sigma}=%.2f,%.2f$' % (r1_old_reco,r2_old_reco )))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.6, 0.55, textstr, transform=ax.transAxes, fontsize=20,
            verticalalignment='top', bbox=props)

    #if axis_cut:
    ax.set_xlim(minaxis,maxaxis)
    ax.set_xlabel(xlabel,fontsize=20)
    ax.set_title(title,fontsize=25)

    old_reco_name = old_reco_name.replace(" ","")
    variable = variable.replace(" ","")
    savename = "%sResolution"%variable
    if weights is not None:
        savename+="Weighted"
    if use_fraction:
        savename += "Frac"
    if use_old_reco:
        savename += "_Compare%sReco"%reco_name
    if axis_cut:
        savename += "_xlim"
    if maxtrue or mintrue:
        savename += "_truthcut"
    if save == True:
        plt.savefig("%s%s.png"%(savefolder,savename),bbox_inches='tight')
    if not notebook:
        plt.close()

def plot_compare_resolution(truth,nn_reco,namelist, weights_dict=None, savefolder=None,\
                            save=False,bins=100,use_fraction=False, mask_dict=None,mask_index=None,
                            minval=None,maxval=None,reco_name="CNN",variable="Energy",units="(GeV)",notebook=False):
    """Plots resolution for dict of inputs
    Receives:
        truth = dict of truth or Y_test labels
                (contents = [key name, energy], shape = [number syst sets, number of events])
        nn_reco = dict of NN predicted or Y_test_predicted results
                    (contents = [key name, energy], shape = [number syst sets, number of events])
        weights = dict of weights
        namelist = list of names for the dict, to use as pretty labels
        save_folder_name = string for output file
        save = bool where True saves and False does not save plot
        bins = int value
        use_fraction: bool, uses fractional resolution if True
    Returns:
        Histograms of resolutions for systematic sets, overlaid
        Prints statistics for all histograms into table
    """

    if weights_dict is not None:
        import wquantiles as wq

    
    print("Resolution")
    print('Name\t Events\t Overflow\t Median\t RMS\t Percentiles\t')
    plt.figure(figsize=(10,7)) 
    if use_fraction:
        title = "%s Fractional %s Resolution"%(reco_name,variable)
        xlabel = "(reconstruction - truth) / truth"
    else:
        title = "%s %s Resolution"%(reco_name, variable)
        xlabel = "reconstruction - truth %s"%units
   
    find_minval = True
    find_maxval = True
    if minval is not None:
        find_minval = False
    if maxval is not None:
        find_maxval = False
        
    resolution = {} 
    for index in range(0,len(namelist)):
        keyname = namelist[index]
        #if mask_dict is None:
        #    mask = numpy.ones(len(truth[keyname]),dtype=bool)
        #else:
        #    mask = mask_dict[keyname][mask_index]
        if use_fraction:
            resolution[keyname] = (nn_reco[keyname] - truth[keyname]) / truth[keyname]
        else:
            resolution[keyname] = nn_reco[keyname] - truth[keyname]
        
        if find_minval:
            if index == 0:
                minval = min(resolution[keyname])
            if minval > min(resolution[keyname]):
                minval = min(resolution[keyname])
        if find_maxval:
            if index == 0:
                maxval = max(resolution[keyname])
            if maxval < max(resolution[keyname]):
                maxval = max(resolution[keyname])
    
    #Get Statistics & Plot
    for index in range(0,len(namelist)):
        keyname = namelist[index]
        #if mask_dict is None:
        #    mask = numpy.ones(len(truth[keyname]),dtype=bool)
        #else:
        #    mask = mask_dict[keyname][mask_index]
        
        rms = get_RMS(resolution[keyname],weights_dict[keyname])
        if weights_dict is not None:
            r1 = wq.quantile(resolution[keyname],weights_dict[keyname],0.16)
            r2 = wq.quantile(resolution[keyname],weights_dict[keyname],0.84)
            median = wq.median(resolution[keyname],weights_dict[keyname])
        else:
            r1, r2 = numpy.percentile(resolution[keyname], [16,84])
            median = numpy.median(resolution[keyname])

        total_events = len(resolution[keyname]) #sum(weights_dict[keyname][mask])
        outside_range = numpy.logical_or(resolution[keyname] < minval, resolution[keyname] > maxval)
        #masked_weights=weights_dict[keyname]
        overflow = sum(outside_range) #sum(masked_weights[outside_range])
        weights_factor = 1 #1e7

        plt.hist(resolution[keyname], range=[minval,maxval],bins=bins, alpha=0.5, label="%s"%namelist[index]);
            
        print("%s\t %.0f\t %.0f\t %.2f\t %.2f\t %.2f, %.2f\t"%(namelist[index], \
                                                        total_events*weights_factor,\
                                                        overflow*weights_factor,\
                                                        median,\
                                                        rms,\
                                                        r1, r2))
    plt.title(title)    
    plt.xlabel(xlabel)
    if use_fraction:
        plt.legend(fontsize=20)

    else:
        plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=2)
    
    basename=""
    if weights_dict is not None:
        basename += "Weighted"
    if use_fraction:
        basename += "Frac"
    basename += "%sResolution_CompareSets%s"%(variable,reco_name)
    if save:
        plt.savefig("%s%s.png"%(savefolder,basename))
    if not notebook:
        plt.close()

def plot_systematic_slices(truth_dict, nn_reco_dict, namelist,
                           weights_dict = None, use_fraction=False,
                           mask_dict=None, mask_index=None,title=None,\
                           use_old_reco = False, old_reco_dict=None,notebook=False,
                           old_reco_weights_dict = None, old_reco_truth_dict = None,\
                           save=False,savefolder=None,cnn_name="CNN",old_reco_name="Retro"):
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
    if weights_dict is not None:
        if old_reco_weights_dict is None:
            old_reco_weights_dict = weights_dict
    if old_reco_dict is not None:
        if old_reco_truth_dict is None:
            old_reco_truth_dict = truth_dict
    
    number_sets = len(namelist)
    percentile_in_peak = 68.27

    left_tail_percentile  = (100.-percentile_in_peak)/2
    right_tail_percentile = 100.-left_tail_percentile
    
    medians  = numpy.zeros(number_sets)
    err_from = numpy.zeros(number_sets)
    err_to   = numpy.zeros(number_sets)
    
    if use_old_reco:
        medians_reco  = numpy.zeros(number_sets)
        err_from_reco = numpy.zeros(number_sets)
        err_to_reco   = numpy.zeros(number_sets)
    
    resolution = {}
    for index in range(0,number_sets):
        keyname = namelist[index]
        #if mask_dict is None:
        #    mask = numpy.ones(len(truth_dict[keyname]),dtype=bool)
        #else:
        #    mask = mask_dict[keyname][mask_index]
        if use_fraction:
            resolution = (nn_reco_dict[keyname] - truth_dict[keyname])/truth_dict[keyname]
        else:
            resolution = (nn_reco_dict[keyname] - truth_dict[keyname])
    
        if weights_dict is not None:
            import wquantiles as wq
            lower_lim = wq.quantile(resolution,weights_dict[keyname],0.16)
            upper_lim = wq.quantile(resolution,weights_dict[keyname],0.84)
            median = wq.median(resolution,weights_dict[keyname])
        else:
            lower_lim = numpy.percentile(resolution, left_tail_percentile)
            upper_lim = numpy.percentile(resolution, right_tail_percentile)
            median = numpy.percentile(resolution, 50.)
        
        medians[index] = median
        err_from[index] = lower_lim
        err_to[index] = upper_lim
    
        if use_old_reco:
            if use_fraction:
                resolution_old_reco = ((old_reco_dict[keyname]-old_reco_truth_dict[keyname])/old_reco_truth_dict[keyname])
            else:
                resolution_old_reco = (old_reco_dict[keyname]-old_reco_truth_dict[keyname])
            
            if weights_dict is not None:
                lower_lim_reco = wq.quantile(resolution_old_reco,old_reco_weights_dict[keyname],0.16)
                upper_lim_reco = wq.quantile(resolution_old_reco,old_reco_weights_dict[keyname],0.84)
                median_reco = wq.median(resolution_old_reco,old_reco_weights_dict[keyname])
            else:
                lower_lim_reco = numpy.percentile(resolution_old_reco, left_tail_percentile)
                upper_lim_reco = numpy.percentile(resolution_old_reco, right_tail_percentile)
                median_reco = numpy.percentile(resolution_old_reco, 50.)
            
            medians_reco[index] = median_reco
            err_from_reco[index] = lower_lim_reco
            err_to_reco[index] = upper_lim_reco


    x_range = numpy.linspace(1,number_sets,number_sets)
    
    fig, ax = plt.subplots(figsize=(10,7))
    plt.errorbar(x_range, medians, yerr=[medians-err_from, err_to-medians],  capsize=5.0, fmt='o',label="%s"%cnn_name)
    if use_old_reco:
        plt.errorbar(x_range, medians_reco, yerr=[medians_reco-err_from_reco, err_to_reco-medians_reco], capsize=5.0,fmt='o',label="%s"%old_reco_name)
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
        ax.set_ylabel(r'Fractional Resolution: $\frac{reconstruction - truth}{truth}$')
    else:
        ax.set_ylabel("Resolution: \n reconstruction - truth (GeV)")
    ax.set_title("%s"%title)
    
    savename = ""
    if weights_dict is not None:
        savename += "Weighted"
    if use_fraction:
        savename += "Frac"
    if use_old_reco:
        savename += "_CompareOldReco"
    savename += "SystematicResolutionCompare"
    if save == True:
        plt.savefig("%s%s.png"%(savefolder,savename))
    if not notebook:
        plt.close()

def plot_bin_slices(truth, nn_reco, energy_truth=None, weights=None,\
                       use_fraction = False, old_reco=None,old_reco_truth=None,\
                       reco_energy_truth=None,old_reco_weights=None,\
                       bins=10,min_val=0.,max_val=60., ylim = None,\
                       save=False,savefolder=None,vs_predict=False,\
                       flavor="NuMu", sample="CC",style="contours",\
                       variable="Energy",units="(GeV)",xlog=False,save_name=None,
                       xvariable="Energy",xunits="(GeV)",notebook=False,
                       specific_bins = None,xline=None,xline_name="DeepCore",
                       epochs=None,reco_name="Retro",cnn_name="CNN",
                       legend="upper center",add_contour=False,
                       print_bins=False,variable_type="True"):
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
        style= contours or errorbars
    Returns:
        Scatter plot with energy values on x axis (median of bin width)
        y axis has median of resolution with error bars containing 68% of resolution
    """
    reco_weights = old_reco_weights
    if weights is not None:
        import wquantiles as wq
        if reco_weights is None:
            reco_weights = numpy.array(weights)

    nn_reco = numpy.array(nn_reco)
    truth = numpy.array(truth)
     ## Assume old_reco truth is the same as test sample, option to set it otherwise
    if old_reco_truth is None:
        truth2 = numpy.array(truth)
    else:
        truth2 = numpy.array(old_reco_truth)
    if reco_energy_truth is None:
        energy_truth2 = numpy.array(energy_truth)
    else:
        energy_truth2 = numpy.array(reco_energy_truth)
    #Check nan
    if old_reco is not None:
        is_nan = numpy.isnan(old_reco)
        assert sum(is_nan) == 0, "OLD RECO HAS NAN"
    is_nan = numpy.isnan(nn_reco)
    assert sum(is_nan) == 0, "CNN RECO HAS NAN"
    
    # NAN CUT FOR OLD RECO
    #if old_reco is not None:
    #    not_nan = numpy.logical_not(numpy.isnan(old_reco))
    #    if sum(not_nan) != len(not_nan):
    #        print("CUTTING NAN VALUES FROM OLD RECO")
    #    old_reco = old_reco[not_nan]
    #    truth2 = truth2[not_nan]
    #    if reco_weights is not None:
    #        reco_weights = reco_weights[not_nan]
    #    if energy_truth is not None:
    #        energy_truth2 = energy_truth2[not_nan]


    if use_fraction:
        resolution = ((nn_reco-truth)/truth) # in fraction
    else:
        resolution = (nn_reco-truth)
    resolution = numpy.array(resolution)
    percentile_in_peak = 68.27
    second_percentile = 95.45

    left_tail_percentile  = (100.-percentile_in_peak)/2
    left2_tail_percentile  = (100.-second_percentile)/2
    right_tail_percentile = 100.-left_tail_percentile
    right2_tail_percentile = 100.-left2_tail_percentile

    if specific_bins is None:
        variable_ranges  = numpy.linspace(min_val,max_val, num=bins+1)
        variable_centers = (variable_ranges[1:] + variable_ranges[:-1])/2.
    else:
        max_val = specific_bins[-1]
        min_val = specific_bins[0]
        variable_ranges = specific_bins
        variable_centers = []
        for i in range(len(specific_bins)-1):
            variable_centers.append(specific_bins[i] + ((specific_bins[i+1] - specific_bins[i])/2.))

    medians  = numpy.zeros(len(variable_centers))
    err_from = numpy.zeros(len(variable_centers))
    err_to   = numpy.zeros(len(variable_centers))
    err2_from = numpy.zeros(len(variable_centers))
    err2_to   = numpy.zeros(len(variable_centers))

    if old_reco is not None:
        if use_fraction:
            resolution_reco = ((old_reco-truth2)/truth2)
        else:
            resolution_reco = (old_reco-truth2)
        resolution_reco = numpy.array(resolution_reco)
        medians_reco  = numpy.zeros(len(variable_centers))
        err_from_reco = numpy.zeros(len(variable_centers))
        err_to_reco   = numpy.zeros(len(variable_centers))
        err2_from_reco = numpy.zeros(len(variable_centers))
        err2_to_reco   = numpy.zeros(len(variable_centers))

    evt_per_bin = []
    evt_per_bin2 = []
    for i in range(len(variable_ranges)-1):
        var_from = variable_ranges[i]
        var_to   = variable_ranges[i+1]
        
        if vs_predict:
            x_axis_array = nn_reco
            x_axis_array2 = old_reco #nn_reco
            title="%s Resolution Dependence"%(variable)
        else:
            if energy_truth is None:
                title="%s Resolution Dependence"%(variable)
                x_axis_array = truth
                x_axis_array2 = truth2
            else:
                title="%s Resolution %s Dependence"%(variable,xvariable)
                energy_truth = numpy.array(energy_truth)
                x_axis_array = energy_truth
                x_axis_array2 = energy_truth2
                
        cut = (x_axis_array >= var_from) & (x_axis_array < var_to)
        if print_bins:
            evt_per_bin.append(sum(cut))
        #print("Events in ", var_from, " to ", var_to, sum(cut))
        if old_reco is not None:
            cut2 = (x_axis_array2 >= var_from) & (x_axis_array2 < var_to)
            if print_bins:
                evt_per_bin2.append(sum(cut2))
            #print("Events in ", var_from, " to ", var_to, sum(cut2))

        if weights is not None:
            lower_lim = wq.quantile(resolution[cut],weights[cut],left_tail_percentile/100.)
            lower2_lim = wq.quantile(resolution[cut],weights[cut],left2_tail_percentile/100.)
            upper_lim = wq.quantile(resolution[cut],weights[cut], right_tail_percentile/100.)
            upper2_lim = wq.quantile(resolution[cut],weights[cut], right2_tail_percentile/100.)
            median = wq.median(resolution[cut],weights[cut])
        else:
            lower_lim = numpy.percentile(resolution[cut], left_tail_percentile/100.)
            lower2_lim = numpy.percentile(resolution[cut], left2_tail_percentile/100.)
            upper_lim = numpy.percentile(resolution[cut], right_tail_percentile/100.)
            upper2_lim = numpy.percentile(resolution[cut], right2_tail_percentile/100.)
            median = numpy.percentile(resolution[cut], 0.50)

        medians[i] = median
        err_from[i] = lower_lim
        err_to[i] = upper_lim
        err2_from[i] = lower2_lim
        err2_to[i] = upper2_lim

        if old_reco is not None:
            if reco_weights is not None:
                lower_lim_reco = wq.quantile(resolution_reco[cut2],reco_weights[cut2],left_tail_percentile/100.)
                upper_lim_reco = wq.quantile(resolution_reco[cut2],reco_weights[cut2],right_tail_percentile/100.)
                lower2_lim_reco = wq.quantile(resolution_reco[cut2],reco_weights[cut2],left2_tail_percentile/100.)
                upper2_lim_reco = wq.quantile(resolution_reco[cut2],reco_weights[cut2],right2_tail_percentile/100.)
                median_reco = wq.median(resolution_reco[cut2],reco_weights[cut2])
            else:
                lower_lim_reco = numpy.percentile(resolution_reco[cut2], left_tail_percentile/100.)
                upper_lim_reco = numpy.percentile(resolution_reco[cut2], right_tail_percentile/100.)
                lower2_lim_reco = numpy.percentile(resolution_reco[cut2], left2_tail_percentile/100.)
                upper2_lim_reco = numpy.percentile(resolution_reco[cut2], right2_tail_percentile/100.)
                median_reco = numpy.percentile(resolution_reco[cut2], 0.50)

            medians_reco[i] = median_reco
            err_from_reco[i] = lower_lim_reco
            err_to_reco[i] = upper_lim_reco
            err2_from_reco[i] = lower2_lim_reco
            err2_to_reco[i] = upper2_lim_reco

    plt.figure(figsize=(10,7))
    plt.plot([min_val,max_val], [0,0], color='k')
    if style == "errorbars":
        if old_reco is not None:
            (_, caps_reco, _) = plt.errorbar(variable_centers, medians_reco, yerr=[medians_reco-err_from_reco, err_to_reco-medians_reco], xerr=[ variable_centers-variable_ranges[:-1], variable_ranges[1:]-variable_centers ], capsize=3.0, fmt='o',label="%s"%reco_name)
            for cap in caps_reco:
                cap.set_markeredgewidth(5)
        (_, caps, _) = plt.errorbar(variable_centers, medians, yerr=[medians-err_from, err_to-medians], xerr=[ variable_centers-variable_ranges[:-1], variable_ranges[1:]-variable_centers ], capsize=3.0, fmt='o',label=cnn_name)
        for cap in caps:
            cap.set_markeredgewidth(5)
        plt.legend(loc=legend)
        if xline is not None:
            if type(xline) is list:
                for x_val in xline:
                    plt.axvline(x_val,linewidth=3,color='k',linestyle="dashed",label="%s"%xline_name)
            else:
                plt.axvline(xline,linewidth=3,color='k',linestyle="dashed",label="%s"%xline_name)
            
            plt.legend(loc=legend)
    else: #countours
        alpha_extra=0.75
        alpha=0.5
        lwid=3
        #cmap = plt.get_cmap('Blues')
        #colors = cmap(numpy.linspace(0, 1, 2 + 2))[2:]
        #color=colors[0]
        color=['mediumblue','cornflowerblue','lightsteelblue']
        #cmap = plt.get_cmap('Oranges')
        #rcolors = cmap(numpy.linspace(0, 1, 2 + 2))[2:]
        #rcolor=rcolors[0] 
        rcolor=['darkorange','darkorange','moccasin']
        ax = plt.gca()
        if old_reco is not None:
            if add_contour:
                ax.fill_between(variable_centers,medians_reco,err2_from_reco, color=rcolor[2], alpha=alpha_extra,linestyle=':',linewidth=2)
                ax.fill_between(variable_centers,medians_reco,err2_to_reco, color=rcolor[2],alpha=alpha_extra,linestyle=':',linewidth=2,label=reco_name + ' 95%')
            ax.fill_between(variable_centers,medians_reco,err_from_reco, color=rcolor[1], alpha=alpha,linestyle='--',linewidth=2)
            ax.fill_between(variable_centers,medians_reco,err_to_reco, color=rcolor[1],alpha=alpha,linestyle='--',linewidth=2,label=reco_name + ' 68%')
            ax.plot(variable_centers,medians_reco, color=rcolor[0], linestyle='-', label="%s median"%reco_name, linewidth=lwid)

        #Plot CNN
        if add_contour:
            ax.fill_between(variable_centers,medians, err2_from,color=color[2], alpha=alpha_extra,linestyle=':',linewidth=2)
            ax.fill_between(variable_centers,medians, err2_to, color=color[2], alpha=alpha_extra,linestyle=':',linewidth=2, label=cnn_name + ' 95%')
        ax.fill_between(variable_centers,medians, err_from,color=color[1], alpha=0.3,linestyle='--',linewidth=2)
        ax.fill_between(variable_centers,medians, err_to, color=color[1], alpha=0.3,linestyle='--',linewidth=2,label=cnn_name + ' 68%')
        ax.plot(variable_centers, medians,linestyle='-',label="%s median"%(cnn_name), color=color[0], linewidth=lwid)
        if xline is not None:
            if type(xline) is list:
                for x_val in xline:
                    plt.axvline(x_val,linewidth=3,color='k',linestyle="dashed",label="%s"%xline_name)
            else:
                plt.axvline(xline,linewidth=3,color='k',linestyle="dashed",label="%s"%xline_name)
        plt.legend(loc=legend,bbox_to_anchor=(1.3, 0.7))
    plt.xlim(min_val,max_val)
    if ylim is not None:
        plt.ylim(ylim)
    if vs_predict:
        plt.xlabel("Reconstructed %s %s"%(variable,units),fontsize=20)
    elif energy_truth is not None:
        plt.xlabel("%s %s"%(xvariable,units),fontsize=20)
    else:
        plt.xlabel("%s %s %s"%(variable_type,variable,units),fontsize=20)
    if use_fraction:
        plt.ylabel(r'Fractional Resolution: $\frac{reconstruction - truth}{truth}$',fontsize=20)
    else:
         plt.ylabel("Resolution: \n reconstruction - truth %s"%units,fontsize=20)
    if xlog:
        plt.xscale('log')

    if print_bins:
        print(evt_per_bin)
        if old_reco is not None:
            print(evt_per_bin2)

    #if epochs:
    #    title += " at %i Epochs"%epochs
    if flavor == "NuMu" or flavor == "numu":
        title += r' for $\nu_\mu$ ' 
    elif flavor == "NuE" or flavor == "nue":
        title += r' for $\nu_e$ '
    elif flavor == "NuTau" or flavor == "nutau":
        title += r' for $\nu_\tau$ '
    elif flavor == "Mu" or flavor == "mu":
        title += r' for $\mu$ '
    elif flavor == "Nu" or flavor == "nu":
        title += r' for $\nu$ '
    else:
        title += flavor
    title += sample
    plt.title(title,fontsize=25)

    reco_name = reco_name.replace(" ","")
    variable = variable.replace(" ","")
    cnn_name = cnn_name.replace(" ","")
    savename = "%s%sResolutionSlices"%(variable,cnn_name)
    if vs_predict:
        savename +="VsPredict"
    if use_fraction:
        savename += "Frac"
    if weights is not None:
        savename += "Weighted"
    if flavor is not None:
        savename += "%s"%flavor.replace(" ","")
    if energy_truth is not None:
        xvar_no_space = xvariable.replace(" ","")
        savename += "_%sBinned"%xvar_no_space
    if style == "errorbars":
        savename += "ErrorBars"
    if xlog:
        savename +="_xlog"
    if old_reco is not None:
        savename += "_Compare%sReco"%reco_name
    if ylim is not None:
        savename += "_ylim"
    if save_name is not None:
        savename += "_%s"%savename
    if save == True:
        plt.savefig("%s%s.png"%(savefolder,savename),bbox_inches='tight')

    if not notebook:
        plt.close()

def plot_rms_slices(truth, nn_reco, energy_truth=None, use_fraction = False,  \
                       old_reco=None,old_reco_truth=None, reco_energy_truth=None,\
                       bins=10,min_val=0.,max_val=60., ylim = None,weights=None,\
                       old_reco_weights=None,save=False,savefolder=None,\
                       variable="Energy",units="(GeV)",epochs=None,reco_name="Retro",
                        flavor="NuMu", sample="CC",notebook=False):
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
     ## Assume old_reco truth is the same as test sample, option to set it otherwise
    if old_reco_truth is None:
        truth2 = numpy.array(truth)
    else:
        truth2 = numpy.array(old_reco_truth)
    if reco_energy_truth is None:
        energy_truth2 = numpy.array(energy_truth)
    else:
        energy_truth2 = numpy.array(reco_energy_truth)
    if weights is not None:
        if old_reco_weights is None:
            old_reco_weights = weights

    if use_fraction:
        resolution = ((nn_reco-truth)/truth) # in fraction
    else:
        resolution = (nn_reco-truth)
    resolution = numpy.array(resolution)

    variable_ranges  = numpy.linspace(min_val,max_val, num=bins+1)
    variable_centers = (variable_ranges[1:] + variable_ranges[:-1])/2.

    rms_all  = numpy.zeros(len(variable_centers))

    if old_reco is not None:
        if use_fraction:
            resolution_reco = ((old_reco-truth2)/truth2)
        else:
            resolution_reco = (old_reco-truth2)
        resolution_reco = numpy.array(resolution_reco)
        rms_reco_all = numpy.zeros(len(variable_centers))

    for i in range(len(variable_ranges)-1):
        var_from = variable_ranges[i]
        var_to   = variable_ranges[i+1]
        
        title=""
        #else:
        #    title="Weighted "
        if energy_truth is None:
            title+="%s RMS Resolution Dependence"%(variable)
            cut = (truth >= var_from) & (truth < var_to)
            cut2 = (truth2 >= var_from) & (truth2 < var_to)
        else:
            #print("Using energy for x-axis. Make sure your min_val and max_val are in terms of energy!")
            title+="%s RMS Resolution Energy Dependence"%(variable)
            energy_truth = numpy.array(energy_truth)
            cut = (energy_truth >= var_from) & (energy_truth < var_to)
            cut2 = (energy_truth2 >= var_from) & (energy_truth2 < var_to)

        if weights is not None:
            weight_here = weights[cut]
            reco_weight_here = old_reco_weights[cut2]
        else:
            weight_here = None
        rms = get_RMS(resolution[cut],weight_here)
        rms_all[i] = rms
       
        if old_reco is not None:
            rms_reco = get_RMS(resolution_reco[cut2],reco_weight_here)
            rms_reco_all[i] = rms_reco

    #cnn_name = "CNN"
    cnn_name = "Neural Network"
    diff_width=abs(variable_ranges[1] - variable_ranges[0])
    plt.figure(figsize=(10,7))

    if old_reco is not None:
        rms_reco_all = numpy.append(rms_reco_all, rms_reco_all[-1])
        plt.step(variable_ranges, rms_reco_all, where='post', color="orange",label="%s"%reco_name)
    rms_all = numpy.append(rms_all,rms_all[-1])
    plt.step(variable_ranges, rms_all, where='post', color="blue",label=cnn_name)
    plt.legend(fontsize=15)
    
    
    plt.ylim(bottom=0)
    plt.xlim(min_val,max_val)
    if type(ylim) is not None:
        plt.ylim(ylim)
    plt.xlabel("True %s %s"%(variable,units),fontsize=20)
    if use_fraction:
        if weights is not None:
            plt.ylabel(r'Weighted RMS of Fractional Resoltion: $\frac{reconstruction - truth}{truth}$',fontsize=20)
        else:
            plt.ylabel(r'RMS of Fractional Resoltion: $\frac{reconstruction - truth}{truth}$',fontsize=20)
    else:
        if weights is not None:
            plt.ylabel("Weighted RMS of Resolution: \n reconstruction - truth %s"%units,fontsize=20)
        else:
            plt.ylabel("RMS of Resolution: \n reconstruction - truth %s"%units,fontsize=20)
    #if epochs:
    #    title += " at %i Epochs"%epochs
    if flavor == "NuMu" or flavor == "numu":
        title += r' for $\nu_\mu$ ' 
    elif flavor == "NuE" or flavor == "nue":
        title += r' for $\nu_e$ '
    else:
        title += flavor
    title += sample
    plt.title(title,fontsize=25)
    
    #print(rms_all,rms_reco_all)
    
    reco_name = reco_name.replace(" ","")
    variable = variable.replace(" ","")
    savename = "%sRMSSlices"%variable
    if use_fraction:
        savename += "Frac"
    if weights is not None:
        savename += "Weighted"
    if energy_truth is not None:
        savename += "_EnergyBinned"
        plt.xlabel("True Energy (GeV)",fontsize=20)
    if old_reco is not None:
        savename += "_Compare%sReco"%reco_name
    if type(ylim) is not None:
        savename += "_ylim"
    if save == True:
        plt.savefig("%s%s.png"%(savefolder,savename),bbox_inches='tight')
    
    if not notebook:
        plt.close()

def imshow_plot(array,name,emin,emax,tmin,tmax,zlabel,savename):
    
    afig = plt.figure(figsize=(10,7))
    plt.imshow(array,origin='lower',extent=[emin,emax,tmin,tmax],aspect='auto')
    cbar = plt.colorbar()
    cbar.set_label(zlabel,rotation=90,fontsize=20)
    plt.set_cmap('viridis_r')
    cbar.ax.tick_params(labelsize=20) 
    plt.xlabel("True Neutrino Energy (GeV)",fontsize=20)
    plt.ylabel("True Track Length (m)",fontsize=20)
    plt.title("%s for Track Length vs. Energy"%name,fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig(savename,bbox_inches='tight')
    return afig
    
def plot_length_energy(truth, nn_reco, track_index=2,tfactor=200.,\
                        save=False,savefolder=None,use_fraction=False,\
                        ebins=10,tbins=10,emin=5.,emax=100.,tmin=0.,tmax=430.,\
                        cut_truth = False, axis_square =False, zmax=None,
                        variable="Energy", units = "(GeV)", epochs=None,reco_name="CNN"):
   

    true_energy = truth[:,0]*emax
    true_track =  truth[:,track_index]*tfactor
    #nn_reco = nn_reco[:,0]*emax
    
    #print(true_energy.shape,nn_reco.shape)
    if use_fraction:
        resolution = (nn_reco - true_energy)/true_energy
        title = "Fractional %s Resolution"%variable
        zlabel = "(reco - truth) / truth" 
    else:
        resolution = nn_reco - true_energy
        title = "%s Resolution"%variable
        zlabel = "reconstruction - truth (GeV)"
    #print(nn_reco[:10],true_energy[:10])    
        
    percentile_in_peak = 68.27
    left_tail_percentile  = (100.-percentile_in_peak)/2
    right_tail_percentile = 100.-left_tail_percentile
    
    
    energy_ranges  = numpy.linspace(emin,emax, num=ebins+1)
    energy_centers = (energy_ranges[1:] + energy_ranges[:-1])/2.
    track_ranges  = numpy.linspace(tmin,tmax, num=tbins+1)
    track_centers = (track_ranges[1:] + track_ranges[:-1])/2.

    medians  = numpy.zeros((len(energy_centers),len(track_centers)))
    err_from = numpy.zeros((len(energy_centers),len(track_centers)))
    err_to   = numpy.zeros((len(energy_centers),len(track_centers)))
    rms      = numpy.zeros((len(energy_centers),len(track_centers)))
    
    #print(energy_ranges,track_ranges)
    for e in range(len(energy_ranges)-1):
        e_from = energy_ranges[e]
        e_to   = energy_ranges[e+1]
        for t in range(len(track_ranges)-1):
            t_from = track_ranges[t]
            t_to   = track_ranges[t+1]
            
        
            e_cut = (true_energy >= e_from) & (true_energy < e_to)
            t_cut = (true_track >= t_from) & (true_track < t_to)
            cut = e_cut & t_cut

            subset = resolution[cut]
            #print(subset)
            #print(e_from,e_to,t_from,t_to,true_energy[cut],true_track[cut])
            if sum(cut)==0:
                lower_lim = numpy.nan
                upper_lim = numpy.nan
                median    = numpy.nan
                one_rms   = numpy.nan
            else:
                lower_lim = numpy.percentile(subset, left_tail_percentile)
                upper_lim = numpy.percentile(subset, right_tail_percentile)
                median = numpy.percentile(subset, 50.)
                mean_array = numpy.ones_like(subset)*numpy.mean(subset)
                one_rms = numpy.sqrt( sum((mean_array - subset)**2)/len(subset) )
            #Invert saving because imshow does (M,N) where M is rows and N is columns
            medians[t,e] = median
            err_from[t,e] = lower_lim
            err_to[t,e] = upper_lim
            rms[t,e] = one_rms
    
    stat=["Median", "Lower 1 sigma", "Upper 1 sigma", "RMS"]
    z_name = [zlabel, "lower 1 sigma of " + zlabel, "upper 1 sigma of " + zlabel, "RMS of " + zlabel ]
    
    savename = "%sTrueEnergyTrackReco%s_2DHist_%s.png"%(savefolder,reco_name,stat[0])
    imshow_plot(medians,stat[0],emin,emax,tmin,tmax,z_name[0],savename)
    
    savename="%sTrueEnergyTrackReco%s_2DHist_%s.png"%(savefolder,reco_name,"LowSigma")
    imshow_plot(err_from,stat[1],emin,emax,tmin,tmax,z_name[1],savename)
    
    savename="%sTrueEnergyTrackReco%s_2DHist_%s.png"%(savefolder,reco_name,"HighSigma")
    imshow_plot(err_to,stat[2],emin,emax,tmin,tmax,z_name[2],savename)
    
    savename="%sTrueEnergyTrackReco%s_2DHist_%s.png"%(savefolder,reco_name,stat[3])
    imshow_plot(rms,stat[3],emin,emax,tmin,tmax,z_name[3],savename)

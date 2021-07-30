import numpy as np

def plot_1d_binned_slices(truth, reco1, reco2=None,
                       xarray1=None,xarray2=None,truth2=None,\
                       plot_resolution=False, use_fraction = False,\
                       bins=10,xmin=None,xmax=None,style="contours",\
                       x_name = "Zenith", x_units = "",\
                       reco1_name = "Reco 1", reco2_name = "Reco 2",\
                       reco1_weight = None, reco2_weight = None,
                       save=True,savefolder=None):
    """Plots different energy slices vs each other (systematic set arrays)
    Receives:
        truth = 1D array with truth values
        reco1 = 1D array that has reconstructed results
        reco2 = optional, 1D array that has an alternate reconstructed results
        xarray1 = optional, 1D array that the reco1 variable (or resolution) will be plotted against, if none is given, will automatically use truth1
        xarray2 = optional, 1D array that the reco2 variable (or resolution2) will be plotted against, if none is given, will automatically use xarray1
        truth2 = 1D array with truth values used to calculate resolution2
        plot_resolution = use resolution (reco - truth) instead of just reconstructed values
        use_fraction = bool, use fractional resolution instead of absolute, where (reco - truth)/truth
        style = "errorbars" is only string that would trigger change (to errorbar version), default is contour plot version
        bins = integer number of data points you want (range/bins = width)
        xmin = minimum truth value to start cut at (default = find min)
        xmax = maximum truth value to end cut at (default = find max)
        x_name = variable for x axis (what is the truth)
        x_units = units for truth/x-axis variable
        reco1_name = name for reconstruction 1
        reco2_name = name for reconstruction 2
        reco1_weight = 1D array for reco1 weights, if left None, will not use
        reco2_weight = 1D array for reco2 weights, if left None, will not use
    Returns:
        Scatter plot with truth bins on x axis (median of bin width)
        y axis has median of resolution or absolute reconstructed value with error bars containing given percentile
    """
    
    truth = np.array(truth)
    reco1 = np.array(reco1)
    xarray1 = np.array(xarray1)
    if reco1_weight is not None:
        reco1_weight = np.array(reco1_weight)
    if truth2 is not None:
        truth2 = np.array(truth2)
    if reco2 is not None:
        reco2 = np.array(reco2)
    if xarray2 is not None:
        xarray2 = np.array(xarray2)
    if reco2_weight is not None:
        reco2_weight = np.array(reco2_weight)


    percentile_in_peak = 68.27 #CAN CHANGE
    left_tail_percentile  = (100.-percentile_in_peak)/2
    right_tail_percentile = 100.-left_tail_percentile
    
    # if no xarray given, automatically use truth
    if xarray1 is None:
        xarray1 = np.array(truth)
    if xmin is None:
        xmin = min(xarray1)
        print("Setting xmin based on xarray1 (or truth)--not taking into account xarray2")
    if xmax is None:
        xmax = max(xarray1)
        print("Setting xmax based on xarray1 (or truth)--not taking into account xarray2")

    ranges  = np.linspace(xmin,xmax, num=bins)
    centers = (ranges[1:] + ranges[:-1])/2.

    # Calculate resolution if plot_resolution flag == True
    if plot_resolution:
        if use_fraction:
            yvariable = ((reco1-truth)/truth) # in fraction
        else:
            yvariable = (reco1-truth)
    else: #use reco directly, not resolution
        y_variable = reco1
        assert use_fraction==False, "Flag for fractional resolution only, not doing resolution here"

    medians  = np.zeros(len(centers))
    err_from = np.zeros(len(centers))
    err_to   = np.zeros(len(centers))

    #Compare to second reconstruction if given    
    if reco2 is not None:
        #check if some variables exist, if not, set to match reco1's
        if truth2 is None:
            truth2 = np.array(truth1)
        if xarray2 is None:
            xarray2 = np.array(xarray1)

        if plot_resolution:
            if use_fraction:
                yvariable2 = ((reco2-truth2)/truth2)
            else:
                yvariable2 = (reco2-truth2)
        else:
            yvariable2 = reco2
        medians2  = np.zeros(len(centers))
        err_from2 = np.zeros(len(centers))
        err_to2   = np.zeros(len(centers))

    # Find median and percentile bounds for data
    for i in range(len(ranges)-1):

        # Make a cut based on the truth (binned on truth)
        var_to   = ranges[i+1]
        var_from = ranges[i]
        cut = (xarray1 >= var_from) & (xarray1 < var_to)
        assert sum(cut)>0, "No events in xbin from %s to %s for reco1, may need to change xmin, xmax, or number of bins or check truth/xarray1 inputs"%(var_from, var_to)
        if reco2 is not None:
            cut2 = (xarray2 >= var_from) & (xarray2 < var_to)
            assert sum(cut2)>0, "No events in xbin from %s to %s for reco2, may need to change xmin, xmax, or number of bins or check truth2/xarray2 inputs"%(var_from, var_to)
        
        #find number of reco1 (or resolution) in this bin
        if reco1_weight is None:
            lower_lim = np.percentile(yvariable[cut], left_tail_percentile/100.)
            upper_lim = np.percentile(yvariable[cut], right_tail_percentile/100.)
            median = np.percentile(yvariable[cut], 0.50)
        else:
            import wquantiles as wq
            lower_lim = wq.quantile(yvariable[cut], reco1_weight[cut], left_tail_percentile/100.)
            upper_lim = wq.quantile(yvariable[cut], reco1_weight[cut], right_tail_percentile/100.)
            median = wq.median(yvariable[cut], reco1_weight[cut])

        medians[i] = median
        err_from[i] = lower_lim
        err_to[i] = upper_lim
 
        #find number of reco2 (or resolution2) in this bin
        if reco2 is not None:
            if reco2_weight is None:
                lower_lim2 = np.percentile(yvariable2[cut2], left_tail_percentile)
                upper_lim2 = np.percentile(yvariable2[cut2], right_tail_percentile)
                median2 = np.percentile(yvariable2[cut2], 50.)
            else:
                import wquantiles as wq
                reco2_weight = np.array(reco2_weight)
                lower_lim2 = wq.quantile(yvariable2[cut2], reco2_weight[cut2], left_tail_percentile)
                upper_lim2 = wq.quantile(yvariable2[cut2], reco2_weight[cut2], right_tail_percentile)
                median2 = wq.median(yvariable2[cut2], reco2_weight[cut2])

            medians2[i] = median2
            err_from2[i] = lower_lim2
            err_to2[i] = upper_lim2

    # Make plot
    plt.figure(figsize=(10,7))
    
    # Median as datapoint
    # Percentile as y error bars
    # Bin size as x error bars
    if style is "errorbars":
        plt.errorbar(centers, medians, yerr=[medians-err_from, err_to-medians], xerr=[ centers-ranges[:-1], ranges[1:]-centers ], capsize=5.0, fmt='o',label="%s"%reco1_name)
        #Compare to second reconstruction, if given
        if reco2 is not None:
            plt.errorbar(centers, medians2, yerr=[medians2-err_from2, err_to2-medians2], xerr=[ centers-ranges[:-1], ranges[1:]-centers ], capsize=5.0, fmt='o',label="%s"%reco2_name)
            plt.legend(loc="upper center")
    # Make contour plot
    # Center solid line is median
    # Shaded region is percentile
    # NOTE: plotted using centers, so 0th and last bins look like they stop short (by 1/2*bin_size)
    else:
        alpha=0.5
        lwid=3
        cmap = plt.get_cmap('Blues')
        colors = cmap(np.linspace(0, 1, 2 + 2))[2:]
        color=colors[0]
        cmap = plt.get_cmap('Oranges')
        rcolors = cmap(np.linspace(0, 1, 2 + 2))[2:]
        rcolor=rcolors[0]
        ax = plt.gca()
        ax.plot(centers, medians,linestyle='-',label="%s median"%(reco1_name), color=color, linewidth=lwid)
        ax.fill_between(centers,medians, err_from,color=color, alpha=alpha)
        ax.fill_between(centers,medians, err_to, color=color, alpha=alpha,label=reco1_name + " %i"%percentile_in_peak +'%' )
        if reco2 is not None:
            ax.plot(centers,medians2, color=rcolor, linestyle='-', label="%s median"%reco2_name, linewidth=lwid)
            ax.fill_between(centers,medians2,err_from1, color=rcolor, alpha=alpha)
            ax.fill_between(centers,medians2,err_to2, color=rcolor,alpha=alpha,label=reco2_name + " %i"%percentile_in_peak +'%' )
    
    # Extra features to have a horizontal 0 line and trim the x axis
    plt.plot([xmin,xmax], [0,0], color='k')
    plt.xlim(xmin,xmax)
    
    #Make pretty labels
    plt.xlabel("%s %s"%(x_name,x_units))
    if plot_resolution:
        if use_fraction:
            plt.ylabel("Fractional Resolution: \n (reconstruction - truth)/truth")
        else:
            plt.ylabel("Resolution: \n reconstruction - truth %s"%x_units)
    else:
        plt.ylabel("Reconstructed %s %s"(x_name,x_units)) 

    # Make a pretty title
    title = "%s Dependence for %s"%(x_name,reco1_name)
    if reco2 is not None:
        title += " and %s"(reco2_name)
    if plot_resolution:
        title += " Resolution"
    plt.title("%s"%(title))

    # Make a pretty filename
    savename = "%s"%(x_name.replace(" ",""))
    if use_fraction:
        savename += "Frac"
    if plot_resolution:
        savename += "Resolution"
    if reco2 is not None:
        savename += "_Compare%s"%(reco2_name.replace(" ",""))
    if save == True:
        plt.savefig("%s/%s.png"%(savefolder,savename))

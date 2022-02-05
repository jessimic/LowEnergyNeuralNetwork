import matplotlib
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score, recall_score
from sklearn.metrics import confusion_matrix #, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve
import numpy as np
import os

#colorscale [dark blue, light blue, gray, yellow, orange, red]
colorscale = ['#4575b4', '#91bfdb', '#999999', '#fee090', '#fc8d59', '#d73027']
color_green = '#4daf4a'

def find_thresholds(truth,prediction,contamination=0.1):
    
    #Find 10% contamination track
    fpr, tpr, thresholds = roc_curve(truth, prediction)
    track_contam_index = (np.abs(fpr - contamination)).argmin()
    threshold_track = thresholds[track_contam_index]
    #Find 10% contamination cascade
    inverse = np.ones(len(prediction)) - prediction
    fpr_c, tpr_c, thresholds_c = roc_curve(np.logical_not(truth), inverse)
    casc_contam_index = (np.abs(fpr_c - contamination)).argmin()
    threshold_cascade = thresholds_c[casc_contam_index]

    return threshold_track, threshold_cascade, [fpr[track_contam_index],tpr[track_contam_index]], [fpr_c[casc_contam_index], tpr[casc_contam_index]]

def find_percision(truth,prediction,contamination=0.1):
    precision, recall, thresholds = precision_recall_curve(truth, prediction)
    index_track = (precision - (1.0 - contamination)).argmin()
    threshold_track = thresholds[index_track]
    inverse = np.ones(len(prediction)) - prediction
    p_casc, r_casc, t_casc = precision_recall_curve(np.logical_not(truth), inverse)
    index_casc = (p_casc - (1.0 - contamination)).argmin()
    threshold_casc = t_casc[index_casc]
    return [precision[index_track], recall[index_track], threshold_track], [p_casc[index_casc], r_casc[index_casc], t_casc[index_casc]]


def plot_classification_hist(truth,prediction,reco=None,reco_mask=None,reco_truth=None,reco_weights=None,mask=None,mask_name="", reco_name="CNN",units="",bins=50,log=False,save=True,save_folder_name=None,weights=None,contamination=0.1,normed=False,savename=None,name_prob1="Track",name_prob0="Cascade",notebook=False,ymax=None,xmin=None,xmax=None):

    if mask is not None:
        print("Masking, using %f of input"%(sum(mask)/len(truth)))
        truth = truth[mask]
        prediction = prediction[mask]
        if reco is not None:
            if reco_mask is not None:
                reco = reco[reco_mask]
            else:
                reco = reco[mask]
            if reco_truth is not None:
                reco_mask1 = reco_truth == 1
                reco_mask0 = reco_truth == 0
            else:
                reco_mask1 = truth == 1
                reco_mask0 = truth == 0
            if reco_weights is None:
                reco_weights = weights
            reco_weights1 = reco_weights[mask1]
            reco_weights0 = reco_weights[mask0]
        if weights is not None:
            weights = weights[mask]
        #save_folder_name += mask_name.replace(" ","") + "/"
        #if os.path.isdir(save_folder_name) != True:
        #            os.mkdir(save_folder_name)

    mask1 = truth == 1
    mask0 = truth == 0
    
    if xmin is None:
        xmin = 0
        if xmin > 0.9:
            matplotlib.rc('xtick', labelsize=10)
    if xmax is None:
        xmax = 1.

    fig,ax = plt.subplots(figsize=(10,7))
    name = "%s"%reco_name
    if weights is not None:
        name += "Weighted"
        weights1 = weights[mask1]
        weights0 = weights[mask0]
    else:
        weights1 = None
        weights0 = None
    ax.set_title("%s %s Classification %s"%(name,name_prob1,mask_name),fontsize=25)
    ax.set_xlabel("Probability %s"%(name_prob1),fontsize=20)
    if weights is not None:
        ax.set_ylabel("Rate (Hz)",fontsize=20)
    else:
        ax.set_ylabel("Counts",fontsize=20)

    if log:
        ax.set_yscale("log")

    if reco is not None:
        ax.hist(reco[reco_mask1], bins=bins,color=colorscale[-1],linestyle=":",alpha=1,
                range=[xmin,xmax],weights=reco_weights1,
                label="True Retro %s"%name_prob1); #,density=normed);
        ax.hist(reco[reco_mask0], bins=bins,color=colorscale[0],linestyle=":",alpha=1,
                range=[xmin,xmax],weights=reco_weights0,
                label="True Retro %s"%name_prob0); #,density=normed);
        label1 = "True CNN %s"%name_prob1
        label0 = "True CNN %s"%name_prob0
    else:
        label1 = "True %s"%name_prob1
        label0 = "True %s"%name_prob0
    print(sum(mask0),sum(mask1),len(mask1))
    ax.hist(prediction[mask1], bins=bins, color=colorscale[-1], alpha=0.7,
            range=[xmin,xmax], weights=weights1, label=label1); #,density=normed);
    ax.hist(prediction[mask0], bins=bins, color=colorscale[0], alpha=0.7,
            range=[xmin,xmax], weights=weights0, label=label0); #,density=normed);

    if ymax is not None:
        ax.set_ylim(0,ymax)

    #Plot contamination lines
    threshold1, threshold0, rates_t, rates_c = find_thresholds(truth, prediction, contamination)
    binary1 = prediction > threshold1
    binary0 = prediction < threshold0
#    print("True Track Rate: %.2f, False Track Rate: %.2f"%)
    print("Events predicted to be track: ", sum(binary1), "number of true tracks there: ", sum(np.logical_and(mask1,binary1)), "number of true cascades there: ", sum(np.logical_and(mask0,binary1)))
    #ax.axvline(threshold_track,linewidth=3,color='green',label=r'10% Track Contamination')
    #ax.axvline(threshold_casc,linewidth=3,color='blue',label=r'10% Cascade Contamination')

    ax.legend(fontsize=20)
    
    if savename is None:
        name += "%s%s"%(reco_name, name_prob1.replace(" ",""))
    else:
        name = savename
    end = "Hist"
    if reco is not None:
        end += "_compareReco"
    if normed:
        end += "Normalized"
    if mask is not None:
        end += "_%s"%(mask_name.replace(" ",""))
    if log:
        end+= "log"
    if save:
        plt.savefig("%s%s%s.png"%(save_folder_name,name,end),bbox_inches='tight')
    if not notebook:
        plt.close()

    return threshold1, threshold0

def precision(truth, prediction, reco=None, mask=None, mask_name="", reco_mask = None,save=True,save_folder_name=None,reco_name="Retro",contamination=0.1,notebook=False):

    if mask is not None:
        print(sum(mask)/len(truth))
        truth = truth[mask]
        prediction = prediction[mask]
        if reco is not None:
            if reco_mask is None:
                reco = reco[mask]
            else:
                reco = reco[reco_mask]

    p, r, t = precision_recall_curve(truth, prediction)
    index_track = (p - (1.0 - contamination)).argmin()
    threshold_track = t[index_track]
    if reco is not None:
        p2, r2, t2 = precision_recall_curve(truth, reco)
        index2 = (p2 - (1.0 - contamination)).argmin()
        best2 = t2[index2]
        

    fig, ax = plt.subplots(1,2,figsize=(10,7))
    ax[0].plot(t,p[:-1],'g-',label="CNN")
    #ax[0].axvline(threshold_track,linewidth=3,color='black',label=r'10% Contamination CNN')
    if reco is not None:
        ax[0].plot(t2,p2[:-1],'orange',linestyle="-",label="%s"%reco_name)
        #ax[0].axvline(best2,linewidth=3,label=r'10% Contamination' + " %s"%reco_name)
        ax[0].legend(fontsize=20)
    ax[0].set_ylabel("Precision = TP/(TP + FP)")
    ax[0].set_xlabel("Threshold Cut")
    ax[0].set_title("Track Precision")
   
    inverse = np.ones(len(prediction)) - prediction
    p_casc, r_casc, t_casc = precision_recall_curve(np.logical_not(truth), inverse)
    index_casc = (p_casc - (1.0 - contamination)).argmin()
    threshold_casc = t_casc[index_casc] 
    if reco is not None:    
        inverse_reco = np.ones(len(prediction)) - reco
        p4, r4, t4 = precision_recall_curve(np.logical_not(truth), inverse_reco)
        index4 = (p4 - (1.0 - contamination)).argmin()
        best4 = t4[index4]
    ax[1].plot(t_casc,p_casc[:-1],'b-',label="CNN")
    #ax[1].axvline(threshold_casc,linewidth=3,color='black',label=r'10% Contamination CNN')
    if reco is not None:
        ax[1].plot(t4,p4[:-1],'orange',linestyle="-",label="%s"%reco_name)
        #ax[1].axvline(best4,linewidth=3,label=r'10% Contamination' + " %s"%reco_name)
        ax[1].legend(fontsize=20)
    ax[1].set_ylabel("Precision = TP/(TP + FP)")
    ax[1].set_xlabel("Threshold Cut")
    ax[1].set_title("Cascade Precision")

    name="%s"%mask_name
    if reco is not None:
        name += "_%s"%reco_name
    if save:
        plt.savefig("%sPrecision%s.png"%(save_folder_name,name))

def ROC(truth, prediction,reco=None,reco_truth=None,mask=None,mask_name="",reco_mask=None,save=True,save_folder_name=None,reco_name="Retro",variable="Probability Track",contamination=0.1,notebook=False):

    if mask is not None:
        print(sum(mask)/len(truth))
        truth = truth[mask]
        prediction = prediction[mask]
        if reco is not None:
            if reco_mask is None:
                reco_mask = mask
            reco = reco[reco_mask]
            if reco_truth is None:
                reco_truth = truth
            else:
                reco_truth = reco_truth[reco_mask]

    print("Fraction of true label = 1: %.3f"%(sum(truth)/len(truth)))
    
    # Find ROC Curve + Stats
    fpr, tpr, thresholds = roc_curve(truth, prediction)
    auc = roc_auc_score(truth, prediction)
    threshold_track, threshold_casc, rates_t, rates_c = find_thresholds(truth, prediction, contamination)
    #print('AUC: %.3f' % auc,"best track threshold %.3f"%threshold_track)

    # Plot ROC Curve
    fig, ax = plt.subplots(figsize=(10,7))

    ax.plot([0,1],[0,1],'k:',label="random")
    ax.plot(fpr, tpr, marker='.', markersize=1,label="CNN")
    #Compare other reco
    if reco is not None:
        fpr_reco, tpr_reco, thresholds_reco = roc_curve(reco_truth, reco)
        auc_reco = roc_auc_score(reco_truth, reco)
        ax.plot(fpr_reco, tpr_reco, marker='.', markersize=1,label="%s"%reco_name)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('False Positive Rate',fontsize=20)
    ax.set_ylabel('True Positive Rate',fontsize=20)
    ax.set_title('ROC %s %s'%(variable,mask_name),fontsize=25)
    #ax.plot(rates_t[0],rates_t[1],"g*",markersize=10,label="10% Track Contamination")
    #ax.plot(rates_c[0],rates_c[1],"b*",markersize=10,label="10% Cascade Contamination")
    props = dict(boxstyle='round', facecolor='blue', alpha=0.3)
    ax.text(0.65, 0.45, r'CNN AUC:%.3f'%auc, transform=ax.transAxes, fontsize=20,
            verticalalignment='top', bbox=props)
    if reco is not None:
        props = dict(boxstyle='round', facecolor='blue', alpha=0.3)
        ax.text(0.65, 0.35, r'%s AUC:%.3f'%(reco_name,auc_reco), 
            transform=ax.transAxes, fontsize=20, verticalalignment='top', bbox=props)
    ax.legend(loc="lower right",fontsize=20)

    end = "ROC_%s"%variable.replace(" ","")
    if reco is not None:
        end += "_compare%s"%reco_name.replace(" ","")
    if mask is not None:
        end += "_%s"%mask_name.replace(" ","")
    if save:
        plt.savefig("%s%s.png"%(save_folder_name,end))
    if not notebook:
        plt.close()

    return threshold_track, threshold_casc, auc

def ROC_dict(truth_dict, prediction_dict,namelist, reco_dict=None,mask_dict=None,mask_name="",reco_mask_dict=None,save=True,save_folder_name=None,reco_name="Retro",contamination=0.1,notebook=False):

    print("Keyname\t AUC")
    # Plot ROC Curve
    fig, ax = plt.subplots(figsize=(10,7))
    ax.plot([0,1],[0,1],'k:',label="random")
    for index in range(0,len(namelist)):
        keyname = namelist[index]
        if mask_dict is not None:
            mask = mask_dict[keyname]
            print("Fraction events kept:",sum(mask[keyname])/len(truth_dict[keyname]))
            truth_dict[keyname] = truth_dict[keyname][mask]
            prediction_dict[keyname] = prediction_dict[keyname][mask]
            if reco_dict is not None:
                if reco_mask_dict is None:
                    reco_dict[keyname] = reco_dict[keyname][mask]
                else:
                    reco_dict[keyname] = reco_dict[keyname][reco_mask_dict[keyname]]
        print("Fraction of true tracks: %.3f"%(sum(truth_dict[keyname])/len(truth_dict[keyname])))
        
        # Find ROC Curve + Stats
        fpr, tpr, thresholds = roc_curve(truth_dict[keyname], prediction_dict[keyname])
        auc = roc_auc_score(truth_dict[keyname], prediction_dict[keyname])
        print("%s\t %.3f"%(keyname,auc))
        ax.plot(fpr, tpr, marker='.', markersize=1,label="%s"%keyname)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('False Positive Rate',fontsize=20)
    ax.set_ylabel('True Positive Rate',fontsize=20)
    ax.set_title('ROC Curve %s'%mask_name,fontsize=25)
    #ax.plot(rates_t[0],rates_t[1],"g*",markersize=10,label="10% Track Contamination")
    #ax.plot(rates_c[0],rates_c[1],"b*",markersize=10,label="10% Cascade Contamination")
    props = dict(boxstyle='round', facecolor='blue', alpha=0.3)
    ax.legend(loc="lower right",fontsize=20)

    end = "SystROC"
    if reco is not None:
        end += "_compare%s"%reco_name.replace(" ","")
    if mask is not None:
        end += "_%s"%mask_name.replace(" ","")
    if save:
        plt.savefig("%s%s.png"%(save_folder_name,end))
    if not notebook:
        plt.close()

    return threshold_track, threshold_casc

def my_confusion_matrix(binary_truth, binary_class, weights, mask=None, color="Blues",
                     label0="Muon",label1="Neutrino",ylabel="CNN Prediction",xlabel="Truth",
                     title="CNN Muon Cut",save=True,save_folder_name=None,notebook=False):

    if mask is None:
        mask = np.ones(len(binary_truth),dtype=bool)

    cm = confusion_matrix(binary_truth[mask], binary_class[mask], sample_weight=weights[mask])
    invert_binary_truth = np.invert(binary_truth[mask])
    
    weights_squared = weights*weights
    hist_squared, xbins_notused, ybins_notused = np.histogram2d(invert_binary_truth, binary_class[mask],bins=2,weights=weights_squared[mask]);
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_aspect("equal")
    blues=plt.get_cmap("%s"%color)
    minval = np.min(cm)
    maxval = np.max(cm)

    
    hist, xbins, ybins, im = ax.hist2d(invert_binary_truth, binary_class[mask], bins=2,
                                       cmap=blues,weights=weights[mask],
                                       norm=colors.LogNorm(vmin=minval, vmax=maxval));
    fig.colorbar(im, orientation='vertical')
    plt.yticks(ticks=[0.25,0.75],labels=["%s"%label0, "%s"%label1],fontsize=20)
    plt.xticks(ticks=[0.25,0.75],labels=["%s"%label1, "%s"%label0],fontsize=20)
    ax.set_ylabel("%s"%ylabel,fontsize=25)
    ax.set_xlabel("%s"%xlabel,fontsize=25)
    ax.set_title("%s"%title,fontsize=30)
     
    
    true_one = binary_truth == 1
    true_zero = binary_truth == 0
    mask_true_one = np.logical_and(true_one,mask)
    mask_true_zero = np.logical_and(true_zero,mask)
 
    transposed_hist_squared = np.transpose(hist_squared)
    
    save_percent = []
    for i in range(len(ybins)-1):
        for j in range(len(xbins)-1):
            c="k"
            if j == 0:
                total = sum(weights[mask_true_one])
            if j == 1:
                total = sum(weights[mask_true_zero])
            events = hist.T[i,j]
            error = np.sqrt(transposed_hist_squared[i,j])
            percent = (float(events)/float(total))*100
            save_percent.append(percent)
            s = "%.2e"%(events) + r'$\pm$' + "%.2e\n %.2f"%(error,percent) + '% of truth'
            if events > maxval/2.:
                c="w"
            ax.text(xbins[j]+0.25,ybins[i]+0.25,"%s"%s, 
                    color=c, ha="center", va="center", fontweight="bold",fontsize=18)

    name = title.replace(" ","")
    if save:
        plt.savefig("%s%sConfusionMaxtrix.png"%(save_folder_name,name),bbox_inches='tight')

    if not notebook:
        plt.close()

    # Order = (x=0,y=0), (x=1, y=0), (x=0, y=1), (x=1, y=1)
    return save_percent

def plot_osc_hist_given_hist(hist_here,label_factor=1,title="Counts",
                            label_factor_title=None,pid="CNN Track",
                            save_folder_name=None,save=True,notebook=False):
    
    if label_factor_title is None:
        label_factor_title = str(label_factor)
    
    fig, ax = plt.subplots(figsize=(15,13))
    ax.set_title("%s: True Muon, %s (label x %s)"%(title,pid,label_factor_title),fontsize=25)
    im = ax.imshow(hist_here,origin='lower', cmap='viridis_r')
    fig.colorbar(im, orientation='vertical')
    ax.set_xlabel("CNN Energy (GeV)",fontsize=20)
    ax.set_ylabel("CNN Cos Zenith",fontsize=20)

    xlabels=[]
    for i in range(0,len(energy_bins)):
        if i%2==0:
            xlabels.append("%.2f"%energy_bins[i])
    ylabels=[]
    for i in range(0,len(coszen_bins)):
        if i%2==0:
            ylabels.append("%.2f"%coszen_bins[i])
    ax.set_xticks([-0.5,1.5,3.5,5.5,7.5,9.5,11.5])
    ax.set_yticks([-0.5,1.5,3.5,5.5,7.5,9.5])
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels)
    
    maxhist = np.nanmax(hist_here)
    for i in range(len(ybins)-1):
        for j in range(len(xbins)-1):
            c="k"
            #total = sum(weights1[mask])
            events = hist_here[i,j]
            if events > maxhist/2.:
                c="w"
            s = "%.2f"%(events*label_factor)
            ax.text(j, i,"%s"%s, 
                    color=c, ha="center", va="center",fontsize=15)
    
    name = "%s"%title.replace(" ","")
    name += "%s"%pid.replace(" ","")
    
    if save:
        plt.savefig("%s%sOscMatrix.png"%(save_folder_name,name),bbox_inches='tight')

    if not notebook:
        plt.close()

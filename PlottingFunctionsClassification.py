import matplotlib
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score, recall_score
from sklearn.metrics import confusion_matrix #, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve
import numpy as np
import os

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


def plot_classification_hist(truth,prediction,reco=None,reco_mask=None,mask=None,mask_name="", variable="Classification",units="",bins=50,log=False,save=True,save_folder_name=None,weights=None,contamination=0.1):

    if mask is not None:
        print("Masking, using %f of input"%(sum(mask)/len(truth)))
        truth = truth[mask]
        prediction = prediction[mask]
        if reco is not None:
            if reco_mask is not None:
                reco = reco[reco_mask]
            else:
                reco = reco[mask]
        if weights is not None:
            weights = weights[mask]
        #save_folder_name += mask_name.replace(" ","") + "/"
        #if os.path.isdir(save_folder_name) != True:
        #            os.mkdir(save_folder_name)

    maskTrack = truth == 1
    maskCascade = truth == 0
    

    fig,ax = plt.subplots(figsize=(10,7))
    name = ""
    if weights is not None:
        name += "Weighted"
        weights_track = weights[maskTrack]
        weights_cascade = weights[maskCascade]
    else:
        weights_track = None
        weights_cascade = None
    ax.set_title("%s %s %s"%(name,variable,mask_name),fontsize=25)
    ax.set_xlabel("%s %s"%(variable,units),fontsize=20)
    if log:
        ax.set_yscale("log")

    if reco is not None:
        ax.hist(reco[maskTrack], bins=bins,color='r',alpha=1,range=[0.,1.],weights=weights_track,label="True Retro Track");
        ax.hist(reco[maskCascade], bins=bins,color='orange',alpha=1,range=[0.,1.],weights=weights_cascade,label="True Retro Cascade");
        track_label = "True CNN Track"
        casc_label = "True CNN Cascade"
    else:
        track_label = "True Track"
        casc_label = "True Cascade"
    ax.hist(prediction[maskTrack], bins=bins,color='g',alpha=0.5,range=[0.,1.],weights=weights_track,label=track_label);
    ax.hist(prediction[maskCascade], bins=bins,color='b',alpha=0.5,range=[0.,1.],weights=weights_cascade,label=casc_label);
    
    #Plot contamination lines
    threshold_track, threshold_casc, rates_t, rates_c = find_thresholds(truth, prediction, contamination)
    binary_track = prediction > threshold_track
    binary_casc = prediction < threshold_casc
#    print("True Track Rate: %.2f, False Track Rate: %.2f"%)
    print("Events predicted to be track: ", sum(binary_track), "number of true tracks there: ", sum(np.logical_and(maskTrack,binary_track)), "number of true cascades there: ", sum(np.logical_and(maskCascade,binary_track)))
    ax.axvline(threshold_track,linewidth=3,color='green',label=r'10% Track Contamination')
    ax.axvline(threshold_casc,linewidth=3,color='blue',label=r'10% Cascade Contamination')

    ax.legend(fontsize=20)

    name += "%s"%(variable.replace(" ",""))
    end = "Hist"
    if reco is not None:
        end += "_compareReco"
    if mask is not None:
        end += "_%s"%(mask_name.replace(" ",""))
    if save:
        plt.savefig("%s%s%s.png"%(save_folder_name,name,end))
    plt.close()

    return threshold_track, threshold_casc

def precision(truth, prediction, reco=None, mask=None, mask_name="", reco_mask = None,save=True,save_folder_name=None,reco_name="Retro",contamination=0.1):

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
    ax[0].plot(t,p,'g-',label="CNN")
    ax[0].axvline(threshold_track,linewidth=3,color='black',label=r'10% Contamination CNN')
    if reco is not None:
        ax[0].plot(t2,p2,'orange',linestyle="-",label="%s"%reco_name)
        ax[0].axvline(best2,'k:',linewidth=3,label=r'10% Contamination' + " %s"%reco_name)
        ax[0].legend(fontsize=20)
    ax[0].set_ylabel("Precision = TP/(TP + FP)",fontize=20)
    ax[0].set_xlabel("Threshold Cut",fontize=20)
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
    ax[1].plot(t_casc,p_casc,'b-',label="CNN")
    ax[1].axvline(t_casc,linewidth=3,color='black',label=r'10% Contamination CNN')
    if reco is not None:
        ax[1].plot(t4,p4,'orange',linestyle="-",label="%s"%reco_name)
        ax[1].axvline(best4,'k:',linewidth=3,label=r'10% Contamination' + " %s"%reco_name)
        ax[1].legend(fontsize=20)
    ax[1].set_ylabel("Precision = TP/(TP + FP)",fontize=20)
    ax[1].set_xlabel("Threshold Cut",fontize=20)
    ax[1].set_title("Cascade Precision")

    name="%s"%mask_name
    if reco is not None:
        name += "_%s"%reco_name
    if save:
        plt.savefig("%sPrecision%s.png"%(save_folder_name,name))

def ROC(truth, prediction,reco=None,mask=None,mask_name="",reco_mask=None,save=True,save_folder_name=None,reco_name="Retro",contamination=0.1):

    if mask is not None:
        print(sum(mask)/len(truth))
        truth = truth[mask]
        prediction = prediction[mask]
        if reco is not None:
            if reco_mask is None:
                reco = reco[mask]
            else:
                reco = reco[reco_mask]
    print("Fraction of true tracks: %.3f"%(sum(truth)/len(truth)))
    
    # Find ROC Curve + Stats
    fpr, tpr, thresholds = roc_curve(truth, prediction)
    auc = roc_auc_score(truth, prediction)
    threshold_track, threshold_casc, rates_t, rates_c = find_thresholds(truth, prediction, contamination)
    print('AUC: %.3f' % auc,"best track threshold %.3f"%threshold_track)

    # Plot ROC Curve
    fig, ax = plt.subplots(figsize=(10,7))

    ax.plot([0,1],[0,1],'k:',label="random")
    ax.plot(fpr, tpr, marker='.', markersize=1,label="CNN")
    #Compare other reco
    if reco is not None:
        fpr_reco, tpr_reco, thresholds_reco = roc_curve(truth, reco)
        auc_reco = roc_auc_score(truth, reco)
        ax.plot(fpr_reco, tpr_reco, marker='.', markersize=1,label="%s"%reco_name)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('False Positive Rate',fontsize=20)
    ax.set_ylabel('True Positive Rate',fontsize=20)
    ax.set_title('ROC Curve %s'%mask_name,fontsize=25)
    #ax.plot(rates_t[0],rates_t[1],"g*",markersize=10,label="10% Track Contamination")
    #ax.plot(rates_c[0],rates_c[1],"b*",markersize=10,label="10% Cascade Contamination")
    props = dict(boxstyle='round', facecolor='blue', alpha=0.3)
    ax.text(0.1, 0.95, r'CNN AUC:%.3f'%auc, transform=ax.transAxes, fontsize=20,
            verticalalignment='top', bbox=props)
    if reco is not None:
        props = dict(boxstyle='round', facecolor='blue', alpha=0.3)
        ax.text(0.1, 0.85, r'%s AUC:%.3f'%(reco_name,auc_reco), 
            transform=ax.transAxes, fontsize=20, verticalalignment='top', bbox=props)
    ax.legend(loc="lower right",fontsize=20)

    end = "ROC"
    if reco is not None:
        end += "_compare%s"%reco_name.replace(" ","")
    if mask is not None:
        end += "_%s"%mask_name.replace(" ","")
    if save:
        plt.savefig("%s%s.png"%(save_folder_name,end))
    plt.close()

    return threshold_track, threshold_casc

def confusion_matrix(truth, prediction, threshold, mask=None, mask_name="", weights=None,save=True, save_folder_name=None):
    if mask is not None:
        truth = truth[mask]
        prediction = prediction[mask]
        if weights is not None:
            weights = weights[mask]

    #Change to 0 or 1
    predictionCascade = prediction < threshold
    predictionTrack = prediction >= threshold
    prediction[predictionCascade] = 0
    prediction[predictionTrack] = 1
    #isTrack = truth == 1
    #isCasc = truth == 0
    #truth[isCasc] = .5
    #truth[isTrack] = 1.5

    
    cm = confusion_matrix(truth, prediction)
    cm_display = ConfusionMatrixDisplay(cm,display_labels=["Cascade","Track"]).plot()
    fig, ax = plt.subplots()
    #cts,xbin,ybin,img = plt.hist2d(prediction, truth, bins=2,range=[[0,2],[0,2]],cmap='viridis_r', weights=weights,density=True)
    #cbar = plt.colorbar()
    #cbar.ax.set_ylabel('Counts', rotation=90)

    name = ""
    if weights is not None:
        name +="Weighted"
    ax.set_title("%s Normalized Confusion Matrix"%name,fontsize=20)
    ax.set_xlabel("Predicted Label",fontsize=20)
    ax.set_ylabel("True Label",fontsize=20)
    ax.set_xticks([0.5,1.5])
    ax.set_yticks([0.5,1.5])
    ax.set_xticklabels(["Cascade","Track"],fontsize=15)
    ax.set_yticklabels(["Cascade","Track"],fontsize=15)
    for i in range(len(ybin)-1):
        for j in range(len(xbin)-1):
            ax.text(xbin[j]+0.5,ybin[i]+0.5, "%.4f"%cts.T[i,j],
            color="w", ha="center", va="center", fontweight="bold",fontsize=20)
    if save:
        plt.savefig("%s%sConfusionMaxtrix.png"%(save_folder_name,name),bbox_inches='tight')

    plt.close()

import matplotlib
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score, recall_score
from sklearn.metrics import confusion_matrix
import numpy as np
import os

def plot_classification_hist(truth,prediction,reco=None,reco_mask=None,mask=None,mask_name="", variable="Classification",units="",bins=50,log=False,save=True,save_folder_name=None,weights=None):

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

    plt.figure(figsize=(10,7))
    name = ""
    if weights is not None:
        name += "Weighted"
        weights_track = weights[maskTrack]
        weights_cascade = weights[maskCascade]
    else:
        weights_track = None
        weights_cascade = None
    plt.title("%s %s %s"%(name,variable,mask_name),fontsize=25)
    plt.xlabel("%s %s"%(variable,units),fontsize=20)
    if log:
        plt.yscale("log")

    if reco is not None:
        plt.hist(reco[maskTrack], bins=bins,color='r',alpha=1,range=[0.,1.],weights=weights_track,label="True Retro Track");
        plt.hist(reco[maskCascade], bins=bins,color='orange',alpha=1,range=[0.,1.],weights=weights_cascade,label="True Retro Cascade");
        track_label = "True CNN Track"
        casc_label = "True CNN Cascade"
    else:
        track_label = "True Track"
        casc_label = "True Cascade"
    plt.hist(prediction[maskTrack], bins=bins,color='g',alpha=0.5,range=[0.,1.],weights=weights_track,label=track_label);
    plt.hist(prediction[maskCascade], bins=bins,color='b',alpha=0.5,range=[0.,1.],weights=weights_cascade,label=casc_label);
    plt.legend(fontsize=20)

    name += "%s"%(variable.replace(" ",""))
    end = "Hist"
    if reco is not None:
        end += "_compareReco"
    if mask is not None:
        end += "_%s"%(mask_name.replace(" ",""))
    if save:
        plt.savefig("%s%s%s.png"%(save_folder_name,name,end))
    plt.close()

def ROC(truth, prediction,reco=None,mask=None,mask_name="",reco_mask=None,save=True,save_folder_name=None,reco_name="Retro"):

    if mask is not None:
        print(sum(mask)/len(truth))
        truth = truth[mask]
        prediction = prediction[mask]
        if reco is not None:
            if reco_mask is None:
                reco = reco[mask]
            else:
                reco = reco[reco_mask]
    print("Fraction of true tracks: %i"%(sum(truth)/len(truth)))
    fpr, tpr, thresholds = roc_curve(truth, prediction)
    tnr = 1 - fpr #true negative rate
    fnr = 1 - tpr #false nagtive rate
    auc = roc_auc_score(truth, prediction)
    sumrates = tnr + tpr
    best_index = np.where(sumrates == max(sumrates))
    best_thres = thresholds[best_index]
    print('AUC: %.3f' % auc,"best sumrates: %.3f"%max(sumrates),best_thres, "BEST THRES IS PROBS BROKEN")



    # ROC Curve
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
    #ax.plot(fpr[best_index],tpr[best_index],marker="*",markersize=10)
    props = dict(boxstyle='round', facecolor='blue', alpha=0.3)
    ax.text(0.1, 0.95, r'CNN AUC:%.3f'%auc, transform=ax.transAxes, fontsize=20,
            verticalalignment='top', bbox=props)
    if reco is not None:
        ax.legend(loc="lower right",fontsize=20)
        props = dict(boxstyle='round', facecolor='blue', alpha=0.3)
        ax.text(0.1, 0.85, r'%s AUC:%.3f'%(reco_name,auc_reco), 
            transform=ax.transAxes, fontsize=20, verticalalignment='top', bbox=props)

    end = "ROC"
    if reco is not None:
        end += "_compare%s"%reco_name.replace(" ","")
    if mask is not None:
        end += "_%s"%mask_name.replace(" ","")
    if save:
        plt.savefig("%s%s.png"%(save_folder_name,end))
    plt.close()

    return best_thres

def confusion_matrix(truth, prediction, best_thres, mask=None, mask_name="", weights=None,save=True, save_folder_name=None):
    if mask is not None:
        truth = truth[mask]
        prediction = prediction[mask]
        if weights is not None:
            weights = weights[mask]

    #Change to 0 or 1
    predictionCascade = prediction < best_thres
    predictionTrack = prediction >= best_thres
    prediction[predictionCascade] = .5
    prediction[predictionTrack] = 1.5
    isTrack = truth == 1
    isCasc = truth == 0
    truth[isCascade] = .5
    truth[isTrack] = 1.5

    fig, ax = plt.subplots()
    cts,xbin,ybin,img = plt.hist2d(prediction, truth, bins=2,range=[[0,2],[0,2]],cmap='viridis_r', weights=weights,density=True)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Counts', rotation=90)
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

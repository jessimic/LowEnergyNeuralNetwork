import matplotlib
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score, recall_score
from sklearn.metrics import confusion_matrix
import numpy as np
import os

def plot_classification_hist(truth,prediction,mask=None,mask_name="", variable="Classification",units="",bins=50,log=False,save=True,save_folder_name=None,weights=None):

    if mask is not None:
        print("Masking, using %f of input"%(sum(mask)/len(truth)))
        truth = truth[mask]
        prediction = prediction[mask]
        if weights is not None:
            weights = weights[mask]
        save_folder_name += mask_name.replace(" ","") + "/"
        if os.path.isdir(save_folder_name) != True:
                    os.mkdir(save_folder_name)

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
    plt.title("%s %s Distribution"%(name,variable),fontsize=25)
    plt.xlabel("%s %s"%(variable,units),fontsize=20)
    if log:
        plt.yscale("log")

    plt.hist(prediction[maskTrack], bins=bins,color='g',alpha=0.5,range=[0.,1.],weights=weights_track,label="True Track");
    plt.hist(prediction[maskCascade], bins=bins,color='b',alpha=0.5,range=[0.,1.],weights=weights_cascade,label="True Cascade");
    plt.legend(fontsize=20)

    name += "%s"%(variable.replace(" ",""))
    if save:
        plt.savefig("%s%sHist.png"%(save_folder_name,name))
    plt.close()

def ROC(truth, prediction,mask=None,mask_name="",save=True,save_folder_name=None):

    if mask is not None:
        print(sum(mask)/len(truth))
        truth = truth[mask]
        prediction = prediction[mask]
        save_folder_name += mask_name.replace(" ","") + "/"
        if os.path.isdir(save_folder_name) != True:
                    os.mkdir(save_folder_name)

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
    ax.plot(fpr, tpr, marker='.', markersize=1,label="ROC")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('False Positive Rate',fontsize=20)
    ax.set_ylabel('True Positive Rate',fontsize=20)
    ax.set_title('ROC Curve',fontsize=25)
    #ax.plot(fpr[best_index],tpr[best_index],marker="*",markersize=10)
    props = dict(boxstyle='round', facecolor='blue', alpha=0.3)
    ax.text(0.1, 0.95, r'AUC:%.3f'%auc, transform=ax.transAxes, fontsize=20,
            verticalalignment='top', bbox=props)

    if save:
        plt.savefig("%sROC.png"%(save_folder_name))
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

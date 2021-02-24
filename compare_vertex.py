import numpy as np
import h5py
import matplotlib.pyplot as plt

input_file = "/mnt/home/micall12/LowEnergyNeuralNetwork/vertex_recos.hdf5"
f = h5py.File(input_file, 'r')
SANTA_fit = f['SANTA_fit'][:]
SANTA = f['SANTA_vertex'][:]
Finite = f['Finite_vertex'][:]
LEERA_EM = f['LEERA_EM_vertex'][:]
LEERA_Had = f['LEERA_Had_vertex'][:]
LEERA_Mu = f['LEERA_Mu_vertex'][:]
true = f['True_vertex'][:]
weights = f['weights'][:]
f.close()
del f

save = True
save_folder = '/mnt/home/micall12/LowEnergyNeuralNetwork/output_plots/Vertex_Test/'

def get_RMS(resolution,weights=None):
    if weights is not None:
        import wquantiles as wq

    mean_array = np.ones_like(resolution)*np.mean(resolution)
    if weights is None:
        rms = np.sqrt( sum((mean_array - resolution)**2)/len(resolution) )
    else:
        rms = np.zeros_like(resolution)
        rms = np.sqrt( sum(weights*(mean_array - resolution)**2)/sum(weights) )
    return rms

def plot_resolution_from_dict(truth,reco,keylist,\
                            cut=None,weights=None,suptitle="Compare Vertex",\
                            savefolder=None,save=False,bins=100,use_fraction=False):
    
    variables = ["x", "y", "z"]
    fig, ax = plt.subplots(1,3,figsize=(20,10))
    fig.suptitle(suptitle)
    if cut is None:
        all_truth = np.ones(len(truth))
        cut = all_truth == 1
        assert sum(cut) == truth.shape[0], "Accidentally cutting, check mask"


    for var in range(0,3):
        if use_fraction:
            title = "Fractional %s Resolution"%variables[var]
            xlabel = "(reco - truth) / truth"
        else:
            title = "%s Resolution"%variables[var]
            xlabel = "reco - truth (m)"
            #plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=2)
    
        print("Resolution %s"%variables[var])
        print('Name\t Entries\t Mean\t Median\t RMS\t Percentiles\t')
        for index in range(0,len(keylist)):
            keyname = keylist[index]
            if use_fraction:
                resolution = (reco[keyname][:,var] - truth[:,var]) / truth[:,var]
            else:
                resolution = reco[keyname][:,var] - truth[:,var]
            ax[var].hist(resolution, bins=bins, weights=weights, \
                    alpha=0.5, label="%s"%keyname);

            #Statistics
            rms = get_RMS(resolution)
            r1, r2 = np.percentile(resolution, [16,84])

            #textstr = '\n'.join((
            #r'%s' % (keyname),
            #r'$\mathrm{events}=%i$' % (len(resolution), ),
            #r'$\mathrm{median}=%.2f$' % (np.median(resolution), ),
            #r'$\mathrm{RMS}=%.2f$' % (rms, ),
            #r'$\mathrm{1\sigma}=%.2f,%.2f$' % (r1,r2 )))
            #props = dict(boxstyle='round', facecolor='blue', alpha=0.3)
            #ax[var].text(0.6, 0.95, textstr, transform=ax.transAxes, fontsize=20,
            #         verticalalignment='top', bbox=props)
            
            print("%s\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f, %.2f\t"%(keyname, \
                                                            len(resolution),
                                                            np.mean(resolution),\
                                                            np.median(resolution),\
                                                            rms,\
                                                            r1, r2))
        ax[var].set_title(title)
        ax[var].set_xlabel(xlabel)
        ax[var].legend(fontsize=20)

    sup = suptitle.replace(" ","")
    if save:
        if use_fraction:
            plt.savefig("%sFractionalVertexResolution_%s.png"%(savefolder,sup),bbox_inches='tight')
        else:
            plt.savefig("%sVertexResolution_%s.png"%(savefolder,sup),bbox_inches='tight')
    plt.close()


LEERA_dict = {}
LEERA_dict['Cascade EM'] = LEERA_EM
LEERA_dict['Cascade Had'] = LEERA_Had
LEERA_dict['Muon'] = LEERA_Mu
keys = ['Cascade EM', 'Cascade Had', 'Muon']
Finite_dict = {}
Finite_dict["Finite Reco"] = Finite
#keys = ["Finite Reco"]
plot_resolution_from_dict(true,LEERA_dict,keys,\
                            cut=None,weights=None,suptitle="LEERA Vertex",\
                            savefolder=save_folder,save=save,bins=100,use_fraction=False)

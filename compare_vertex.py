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
            rms = get_RMS(resolution,weights)
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
variables = ["x", "y", "z"]
print(LEERA_dict['Cascade EM'])
eps = 1e-8
for var in range(0,3):
    delta1 = abs(LEERA_dict['Cascade EM'][:,var] - LEERA_dict['Cascade Had'][:,var])
    delta2 = abs(LEERA_dict['Cascade EM'][:,var] - LEERA_dict['Muon'][:,var])
    delta3 = abs(LEERA_dict['Muon'][:,var] - LEERA_dict['Cascade Had'][:,var])
    check1 = delta1 > eps
    check2 = delta1 > eps
    check3 = delta1 > eps
    assert sum(check1) == 0, "Casc EM - Had not small"
    assert sum(check2) == 0, "Casc EM - Muon not small"
    assert sum(check3) == 0, "Muon - Casc Had not small"

passed_dict = {}
passed_SANTA = SANTA_fit > 1
passed_Finite = np.logical_not(np.isnan(Finite[:,0]))
passed_LEERA = np.logical_not(np.isnan(LEERA_EM[:,0]))
passed = np.logical_and(passed_SANTA,np.logical_and(passed_Finite,passed_LEERA))
print("Passed SANTA = %.3f"%(sum(passed_SANTA)/len(passed_SANTA)))
print("Passed LEERA = %.3f"%(sum(passed_LEERA)/len(passed_LEERA)))
print("Passed Finite = %.3f"%(sum(passed_Finite)/len(passed_Finite)))
passed_dict["Finite"] = Finite[passed]
passed_dict["LEERA"] = LEERA_Had[passed]
passed_dict["SANTA"] = SANTA[passed]
passed_truth = true[passed]
keys = ["Finite", "LEERA", "SANTA"]

delta1 = np.isnan(passed_dict["Finite"])
delta2 = np.isnan(passed_dict["LEERA"])
delta3 = np.isnan(passed_dict["SANTA"])
print(sum(delta1), "Finite has nans left")
print(sum(delta2),"LEERA has nans left")
print(sum(delta3), "SANTA")
"""

plot_resolution_from_dict(passed_truth,passed_dict,keys,\
                            cut=None,weights=None,suptitle="All Passed SANTA Vertex",\
                            savefolder=save_folder,save=save,bins=100,use_fraction=False)
"""
reco_name = ["Finite", "SANTA", "LEERA"]
final_vertex = Finite
reco_type = np.zeros(len(final_vertex))
final_vertex[passed_SANTA] = SANTA[passed_SANTA]
reco_type[passed_SANTA] = 1  
#final_vertex[passed_LEERA] = LEERA_Had[passed_LEERA]
#reco_type[passed_LEERA] = 2
delta1 = np.isnan(final_vertex)
print(sum(delta1), "Finite has nans left")

final_dict = {}
final_dict["Mixed"] = final_vertex[passed_Finite]
reco_type = reco_type[passed_Finite]
final_truth = true[passed_Finite]
num_finite = (reco_type == 0).sum()
num_santa = (reco_type == 1).sum()
#num_leera = (reco_type == 2).sum()
#print("Finite: %i, SANTA: %i, LEERA: %i"%(num_finite,num_santa,num_leera))
print("Finite: %i, SANTA: %i"%(num_finite,num_santa))

plot_resolution_from_dict(final_truth,final_dict,["Mixed"],\
                            cut=None,weights=None,suptitle="Finite SANTA",\
                                                        savefolder=save_folder,save=save,bins=100,use_fraction=False)



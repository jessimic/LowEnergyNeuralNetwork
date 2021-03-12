import numpy as np
import h5py
import matplotlib.pyplot as plt

input_file = "/mnt/home/micall12/LowEnergyNeuralNetwork/test_10_monopod.hdf5"
f = h5py.File(input_file, 'r')
SANTA_fit = f['SANTA_fit'][:]
SANTA = f['SANTA_vertex'][:]
Finite = f['Finite_vertex'][:]
#LEERA_EM = f['LEERA_EM_vertex'][:]
#LEERA_Had = f['LEERA_Had_vertex'][:]
#LEERA_Mu = f['LEERA_Mu_vertex'][:]
Monopod = f['Monopod_vertex'][:]
Corridor = f['CorridorWide_vertex'][:]
HLC = f['HLC_vertex'][:]
L3 = f['L3_vertex'][:]
true = f['True_vertex'][:]
weights = f['weights'][:]
f.close()
del f

input_file2 = "/mnt/home/micall12/LowEnergyNeuralNetwork/output_plots/energy_numu_flat_1_500_level6_cleanedpulses_IC19_CC_20000evtperbin_lrEpochs50_extended/L7_148885/prediction_values_oldGCD.hdf5"
f = h5py.File(input_file2, 'r')
retro = f["reco_test"][:]
retro_true = f["Y_test_use"][:]
retro_weights = f["weights_test"][:]
f.close()
del f
retro_weights = retro_weights[:,8]/9.

save = True
save_folder = '/mnt/home/micall12/LowEnergyNeuralNetwork/output_plots/Vertex_Test/'
weights = weights/9.


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
            if weights is not None:
                import wquantiles as wq
                r1 = wq.quantile(resolution,weights,0.16)
                r2 = wq.quantile(resolution,weights,0.84)
            else:
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

"""
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
    not_nan1 = np.logical_not(np.isnan(delta1))
    not_nan2 = np.logical_not(np.isnan(delta2))
    not_nan3 = np.logical_not(np.isnan(delta3))
    check1 = delta1[not_nan1] > eps
    check2 = delta2[not_nan2] > eps
    check3 = delta3[not_nan3] > eps
    #for i in range(0,len(check1)):
    #    if check1 [i]!= 0:
    #        print(check1[i], delta1[i], LEERA_dict['Cascade EM'][i,var], LEERA_dict['Cascade Had'][i,var])
    assert sum(check1) == 0, "Casc EM - Had not small"
    assert sum(check2) == 0, "Casc EM - Muon not small"
    assert sum(check3) == 0, "Muon - Casc Had not small"
"""

passed_dict = {}
passed_SANTA = SANTA_fit > 0
passed_Finite = np.logical_not(np.isnan(Finite[:,0]))
passed_L3 = np.logical_not(np.isnan(L3[:,0]))
passed_HLC = np.logical_not(np.isnan(HLC[:,0]))
passed_Cor = np.logical_not(np.isnan(Corridor[:,0]))
passed_Mono = np.logical_not(np.isnan(Monopod[:,0]))
passed =  np.logical_and(passed_SANTA,np.logical_and(passed_Finite, np.logical_and(passed_L3, np.logical_and(passed_HLC, np.logical_and(passed_Cor, passed_Mono)))))

#passed_LEERA = np.logical_not(np.isnan(LEERA_EM[:,0]))
#passed = np.logical_and(passed_SANTA,np.logical_and(passed_Finite,passed_LEERA))
#print("Passed LEERA = %.3f"%(sum(passed_LEERA)/len(passed_LEERA)))

print("Passed SANTA = %.3f"%(sum(passed_SANTA)/len(passed_SANTA)))
print("Passed Finite = %.3f"%(sum(passed_Finite)/len(passed_Finite)))
print("Passed L3 = %.3f"%(sum(passed_L3)/len(passed_L3)))
print("Passed HLC = %.3f"%(sum(passed_HLC)/len(passed_HLC)))
print("Passed Cor = %.3f"%(sum(passed_Cor)/len(passed_Cor)))
print("Passed Mono = %.3f"%(sum(passed_Mono)/len(passed_Mono)))

passed_dict["Finite"] = Finite[passed]
#passed_dict["LEERA"] = LEERA_Had[passed]
passed_dict["SANTA"] = SANTA[passed]
passed_dict["L3Guess"] = L3[passed]
passed_dict["HLC"] = HLC[passed]
passed_dict["CorridorWide"] = Corridor[passed]
passed_dict["Monopod"] = Monopod[passed]
passed_truth = true[passed]
passed_weights = weights[passed]
#keys = ["Finite", "LEERA", "SANTA"]
keys = ["Finite", "L3Guess", "SANTA", "HLC", "CorridorWide", "Monopod"]

#delta1 = np.isnan(passed_dict["Finite"])
#delta2 = np.isnan(passed_dict["LEERA"])
#delta3 = np.isnan(passed_dict["SANTA"])
#print(sum(delta1), "Finite has nans left")
#print(sum(delta2),"LEERA has nans left")
#print(sum(delta3), "SANTA")

plot_resolution_from_dict(passed_truth,passed_dict,keys,\
                            cut=None,weights=passed_weights,suptitle="All Passed SANTA Vertex",\
                            savefolder=save_folder,save=save,bins=100,use_fraction=False)
"""
reco_name = ["Finite", "SANTA", "LEERA"]
final_vertex = Finite
reco_type = np.zeros(len(final_vertex))
final_vertex[passed_SANTA] = SANTA[passed_SANTA]
reco_type[passed_SANTA] = 1  
santa_mix = np.copy(final_vertex)
santa_type = np.copy(reco_type)
final_vertex[passed_LEERA] = LEERA_Had[passed_LEERA]
reco_type[passed_LEERA] = 2
delta1 = np.isnan(final_vertex)
print(sum(delta1), "Finite has nans left")

final_dict = {}
final_dict["LEERA SANTA FINITE"] = final_vertex[passed_Finite]
reco_type = reco_type[passed_Finite]
final_dict["SANTA FINITE"] = santa_mix[passed_Finite]
santa_type = santa_type[passed_Finite]

final_truth = true[passed_Finite]
final_weights = weights[passed_Finite]
num_finite = (reco_type == 0).sum()
num_santa = (reco_type == 1).sum()
num_leera = (reco_type == 2).sum()
print("Finite: %i, SANTA: %i, LEERA: %i"%(num_finite,num_santa,num_leera))
num_finite = (santa_type == 0).sum()
num_santa = (santa_type == 1).sum()
print("Finite: %i, SANTA: %i"%(num_finite,num_santa))

plot_resolution_from_dict(final_truth,final_dict,["LEERA SANTA FINITE","SANTA FINITE"],\
                            cut=None,weights=final_weights,suptitle="Compare Mixed",\
                                                        savefolder=save_folder,save=save,bins=100,use_fraction=False)
"""
retro_dict = {}
retro_dict["Retro"] = retro[:,4:7]
print(retro[:,4:7].shape)
retro_truth = retro_true[:,4:7]
plot_resolution_from_dict(retro_truth,retro_dict,["Retro"],\
                            cut=None,weights=retro_weights,suptitle="Retro",\
                                                                                    savefolder=save_folder,save=save,bins=100,use_fraction=False)

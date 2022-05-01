import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import itertools

indir = "/data/user/jmicallef/LowEnergyNeuralNetwork/output_plots/systematics_test"
compare_retro = True
"""
save_path = "/data/user/jmicallef/LowEnergyNeuralNetwork/output_plots/systematics_test/domeff/"
syst_sets = ["0000", "0001", "0002", "0003", "0004"]
set_name = ["baseline", '-10%', '-5%', '+5%', '+10%']
title_sets = "DOM Efficiency"
plot_save_name = "DOMEff"

save_path = "/data/user/jmicallef/LowEnergyNeuralNetwork/output_plots/systematics_test/domeff/"
syst_sets = ["0000", "0005", "0007", "0001", "0002", "0003", "0004", "0006"]
set_name = ["baseline", '-50%', '-30%', '-10%', '-5%', '+5%', '+10%', '+20%']
title_sets = "Extreme DOM Efficiency"
plot_save_name = "ExtremeDOMEff"

save_path = "/data/user/jmicallef/LowEnergyNeuralNetwork/output_plots/systematics_test/holeice/"
#syst_sets = ["0000", "0100", "0101", "0102", "0103", "0104","0105", "0106", "0107", "0109"]
#set_name = ["p0=0.10,p1=-0.05", "-0.06,-0.11", "-0.48,-0.02", "0.28,-0.08", "0.11,0.004", "","0105", "0106", "0107", "0109"]
syst_sets = ["0000", "0100", "0101", "0102", "0103","0151"]
set_name = ["baseline (0.10,-0.05)", "(-0.06,-0.11)", "(-0.48,-0.02)", "(0.28,-0.08)", "(0.11,0.004)","(-1.00,-0.10)"]
title_sets = "Hole Ice p0/p1"
plot_save_name = "HoleIce"
"""
save_path = "/data/user/jmicallef/LowEnergyNeuralNetwork/output_plots/systematics_test/domholeice/"
syst_sets = ["0000", "0104","0105", "0106", "0107", "0109", "0150", "0152"]
#set_name = ["baseline","0150", "0151", "0152"]
set_name = ["baseline (1.0, 0.10,-0.05)","(0.88,-0.05,-0.05)", "(0.90,0.5,0.15)","(0.93,-0.37,0.03)", "(0.95,-1.3,0.15)", "(0.97,0.30,-0.04)", "(1.03,0.12,-0.11)", "(1.12,-0.31,-0.08)"]
title_sets = "DOM Efficiency + Hole Ice"
plot_save_name = "DOMHoleIce"
"""
save_path = "/data/user/jmicallef/LowEnergyNeuralNetwork/output_plots/systematics_test/holeice/"
syst_sets = ["0000", "0300", "0301", "0302", "0303", "0311"]
set_name = ["baseline (0.10,-0.05)", "(-1.0,-0.05)", "(-0.5,-0.05)", "(-0.20,-0.05)", "(0.1,-0.05)", "(0.3,-0.05)"]
title_sets = "Constant P1 Hole Ice"
plot_save_name = "ConstP1HoleIce"

save_path = "/data/user/jmicallef/LowEnergyNeuralNetwork/output_plots/systematics_test/holeice/"
syst_sets = ["0000","0305", "0306", "0307", "0308", "0309", "0310"]
set_name = ["baseline (0.10,-0.05)", "(0.10,-0.15)", "(0.10,-0.10)", "(0.10,0)", "(0.10,0.05)","(0.10,0.10)", "(0.10,0.15)"]
title_sets = "Constant P0 Hole Ice"
plot_save_name = "ConstP0HoleIce"

save_path = "/data/user/jmicallef/LowEnergyNeuralNetwork/output_plots/systematics_test/bulkice/"
syst_sets = ["0000", "0500", "0501", "0502", "0503"]
set_name = ["baseline (scatter, absorp)", "(1.05,1.05)", "(1.05,0.95)", "(0.95,1.05)", "(0.95,0.95)"]
title_sets = "Bulk Ice"
plot_save_name = "BulkIce"

save_path = "/data/user/jmicallef/LowEnergyNeuralNetwork/output_plots/systematics_test/bulkice/"
syst_sets = ["0000", "0509", "0505", "0504", "0508"]
set_name = ["baseline (scatter, absorp)", "(1,0.7)", "(1,0.9)", "(1,1.1)", "(1,1.3)"]
title_sets = "Constant Scatter Bulk Ice"
plot_save_name = "ConstScatterBulkIce"

save_path = "/data/user/jmicallef/LowEnergyNeuralNetwork/output_plots/systematics_test/bulkice/"
syst_sets = ["0000", "0511", "0507", "0506", "0510"]
set_name = ["baseline (scatter, absorp)", "(0.7,1)", "(0.9,1)", "(1.1,1)", "(1.3,1)"]
title_sets = "Constant Absorp Bulk Ice"
plot_save_name = "ConstAbsorpBulkIce"

save_path = "/data/user/jmicallef/LowEnergyNeuralNetwork/output_plots/systematics_test/bulkice/"
syst_sets = ["0000", "0500", "0501", "0502", "0503", "0504", "0505", "0506", "0507", "0508", "0509", "0510", "0511"]
set_name = ["baseline", "0500", "0501", "0502", "0503", "0504", "0505", "0506", "0507", "0508", "0509", "0510", "0511"]
title_sets = "Extreme Bulk Ice"
plot_save_name = "ExtremeBulkIce"
"""

flavor_key = ["nuecc", "numucc", "nutaucc", "nunc", "nu"]
flavors = 5
e_step = (100-5)/20
z_step = (.3+1)/20
cnn = {}
retro = {}
selection = ["numucc", "nuecc"]
variables = ["energy", "coszen"]
input_var = ["median", "q3", "q1", "diff"]
for s in selection:
    cnn[s] = {}
    retro[s] = {}
    for v in variables:
        cnn[s][v] = {}
        retro[s][v] = {}
        for i in input_var:
            cnn[s][v][i] = {}
            retro[s][v][i] = {}
            if v == "energy":
                cnn[s][v][i]['x'] = np.arange(5,100+e_step,e_step)
                retro[s][v][i]['x'] = np.arange(5,100+e_step,e_step)
            if v == "coszen":
                cnn[s][v][i]['x'] = np.arange(-1,0.3+z_step,z_step)
                retro[s][v][i]['x'] = np.arange(-1,0.3+z_step,z_step)
pid = {}
pid["nuecc"] = {}
pid["numucc"] = {}
pid["nutaucc"] = {}
pid["nunc"] = {}
pid["nu"] = {}
rpid = {}
rpid["nuecc"] = {}
rpid["numucc"] = {}
rpid["nutaucc"] = {}
rpid["nunc"] = {}
rpid["nu"] = {}

for a_set in syst_sets:
    for v in variables:
        if v == "energy":
            no = "E"
        if v == "coszen":
            no = "Zen"
        for s in selection:
            if s == "numucc":
                start_count = 0
            else:
                start_count = 3
            c = start_count
            check_vars = 0
            for i in input_var:
                if check_vars < (len(input_var)-1):
                    cnn[s][v][i][a_set] = np.genfromtxt("/data/user/jmicallef/LowEnergyNeuralNetwork/output_plots/systematics_test/%s_median_no%scut_%s.txt"%(v,no,a_set),skip_header=1,usecols=c)
                    if compare_retro:
                        retro[s][v][i][a_set] = np.genfromtxt("/data/user/jmicallef/LowEnergyNeuralNetwork/output_plots/systematics_test/retro_%s_median_noEcut_%s.txt"%(v,a_set),skip_header=1,usecols=c)
                else:
                    cnn[s][v][i][a_set] = cnn[s][v]["q3"][a_set] - cnn[s][v]["q1"][a_set]
                    if compare_retro:
                        retro[s][v][i][a_set] = retro[s][v]["q3"][a_set] - retro[s][v]["q1"][a_set]
                c += 1 
                check_vars +=1

    for j, f in enumerate(flavor_key):                        
        index = j + 1
        pid[f][a_set] = np.genfromtxt("%s/pid_mHz_%s.txt"%(indir,a_set),skip_header=index,usecols=range(1,4),max_rows=1)
        if compare_retro:
            rpid[f][a_set] = np.genfromtxt("%s/retro_pid_mHz_%s.txt"%(indir,a_set),skip_header=index,usecols=range(1,4),max_rows=1)


color = ["red","blue","orange","green","purple","gray","magenta", "brown", "lime", "mediumslateblue", "goldenrod", "cyan", "hotpink"]
marker = ["o","s","*","X","^","p","v","3","D","P","1", "H","2"]

def plot_syst_resolution(plot_variables, syst_sets, title="Energy", ylabel="Fractional Resolution", xlabel="Energy GeV",savename="Energy",savepath=None,zeroline=True,retro_variables=None):
    fig, ax = plt.subplots(figsize=(10,7))
    ax.set_title(title,fontsize=25)
    ax.tick_params(axis="y",labelsize=15)
    ax.tick_params(axis="x",labelsize=15)
    ax.set_xlabel(xlabel,fontsize=20)
    ax.set_ylabel(ylabel,fontsize=20)
    ax.grid()
    if zeroline:
        ax.axhline(0,color="k",linewidth=2)
    for i in range(0,len(syst_sets)):
        name = syst_sets[i]
        x_var = plot_variables["x"]
        y_var = plot_variables[name]
        x = list(itertools.chain(*zip(x_var,x_var[1:])))
        y = list(itertools.chain(*zip(y_var,y_var)))
        ax.plot(x,y,label=set_name[i],marker=None,color=color[i],linewidth=2)
        if retro_variables is not None:
            ry_var = retro_variables[name]
            ry = list(itertools.chain(*zip(ry_var,ry_var)))
            if i == 0: #label first entry
                ax.plot(x,ry,label="retro",marker=None,color=color[i],linewidth=2,linestyle=":")
                savename += "_vsretro"
            else:
                ax.plot(x,ry,marker=None,color=color[i],linewidth=2,linestyle=":")
    plt.legend(fontsize=15)
    plt.savefig("%s/%s.png"%(savepath,savename))

pid_num = [1, 2, 3]
pid_names = ["track", "mixed", "cascade"]
flavor_names = [r'$\nu_e$' + " CC", r'$\nu_\mu$' + " CC", r'$\nu_\tau$' + " CC", r'$\nu$' + " NC",  r'$\nu$' + " All"]

def plot_pid_sets(pid_dict, syst_sets, title="DOM Efficiency", savename="DOMEff",savepath=None,retro_pid=None):
	
    if retro_pid is not None:
            savename += "_vsretro"
    for particle in range(flavors):
        flavor = flavor_key[particle]
        fig, ax = plt.subplots(figsize=(10,7))
        ax.set_title("%s PID Rates - %s"%(flavor_names[particle],title),fontsize=20)
        ax.tick_params(axis="y",labelsize=15)
        ax.set_xticks(pid_num)
        ax.set_xticklabels(pid_names,fontsize=15)
        ax.set_ylabel("Rate Syst/Baseline",fontsize=20)
        ax.axhline(1,color="k",linewidth=2)
        for i in range(1,len(syst_sets)):
            name = syst_sets[i]
            ax.plot(pid_num, pid[flavor][name]/pid[flavor]["0000"], label=set_name[i], marker=marker[i],markersize=10,color=color[i],linestyle=":")
            if retro_pid is not None:
                if flavor == flavor_key[0]: #label first entry
                    ax.plot(pid_num,retro_pid[flavor][name]/retro_pid[flavor]["0000"],label="retro",marker=marker[i],markersize=10,color=color[i],alpha=0.5,linestyle=(0, (1, 10)))
                else:
                    ax.plot(pid_num,retro_pid[flavor][name]/retro_pid[flavor]["0000"],marker=marker[i],markersize=10,color=color[i],alpha=0.5,linestyle=(0, (1, 10)))
        if flavor == "nu":
            ax.set_ylim(0.8,1.2)
        plt.legend(fontsize=15)
        plt.savefig("%s/%s_%s.png"%(savepath,savename,flavor))
	

a_title = "%s"%title_sets
a_savename = "%s_PIDRates"%plot_save_name
if compare_retro:
    plot_pid_sets(pid, syst_sets, title=a_title,savename=a_savename,savepath=save_path,retro_pid=rpid)
else:
    plot_pid_sets(pid, syst_sets, title=a_title,savename=a_savename,savepath=save_path)

a_title = "%s - Energy NuMu CC"%title_sets
a_ylabel= "Bias"
a_xlabel = "Energy (GeV)"
a_savename = "%s_EnergyNuMuCC_Bias"%plot_save_name
if compare_retro:
    plot_syst_resolution(cnn["numucc"]["energy"]["median"], syst_sets, title=a_title, ylabel=a_ylabel, xlabel=a_xlabel,savename=a_savename,savepath=save_path,retro_variables=retro["numucc"]["energy"]["median"])
else:
    plot_syst_resolution(cnn["numucc"]["energy"]["median"], syst_sets, title=a_title, ylabel=a_ylabel, xlabel=a_xlabel,savename=a_savename,savepath=save_path)

a_title = "%s - Energy NuMu CC"%title_sets
a_ylabel= '68% Spread'
a_xlabel = "Energy (GeV)"
a_savename = "%s_EnergyNuMuCC_Width"%plot_save_name
if compare_retro:
    plot_syst_resolution(cnn["numucc"]["energy"]["diff"], syst_sets, title=a_title, ylabel=a_ylabel, xlabel=a_xlabel,savename=a_savename,savepath=save_path,zeroline=False,retro_variables=retro["numucc"]["energy"]["diff"])
else:
    plot_syst_resolution(cnn["numucc"]["energy"]["diff"], syst_sets, title=a_title, ylabel=a_ylabel, xlabel=a_xlabel,savename=a_savename,savepath=save_path,zeroline=False)

a_title = "%s - Energy NuE CC"%title_sets
a_ylabel= "Bias"
a_xlabel = "Energy (GeV)"
a_savename = "%s_EnergyNuECC_Bias"%plot_save_name
if compare_retro:
    plot_syst_resolution(cnn["nuecc"]["energy"]["median"], syst_sets, title=a_title, ylabel=a_ylabel, xlabel=a_xlabel,savename=a_savename,savepath=save_path,retro_variables=retro["nuecc"]["energy"]["median"])
else:
    plot_syst_resolution(cnn["nuecc"]["energy"]["median"], syst_sets, title=a_title, ylabel=a_ylabel, xlabel=a_xlabel,savename=a_savename,savepath=save_path)

a_title = "%s - Energy NuE CC"%title_sets
a_ylabel= '68% Spread'
a_xlabel = "Energy (GeV)"
a_savename = "%s_EnergyNuECC_Width"%plot_save_name
if compare_retro:
    plot_syst_resolution(cnn["nuecc"]["energy"]["diff"], syst_sets, title=a_title, ylabel=a_ylabel, xlabel=a_xlabel,savename=a_savename,savepath=save_path,zeroline=False,retro_variables=retro["nuecc"]["energy"]["diff"])
else:
    plot_syst_resolution(cnn["nuecc"]["energy"]["diff"], syst_sets, title=a_title, ylabel=a_ylabel, xlabel=a_xlabel,savename=a_savename,savepath=save_path,zeroline=False)

a_title = "%s - Cosine Zenith NuMu CC"%title_sets
a_ylabel= "Bias"
a_xlabel = "Cosine Zenith"
a_savename = "%s_CosZenNuMuCC_Bias"%plot_save_name
if compare_retro:
    plot_syst_resolution(cnn["numucc"]["coszen"]["median"], syst_sets, title=a_title, ylabel=a_ylabel, xlabel=a_xlabel,savename=a_savename,savepath=save_path,retro_variables=retro["numucc"]["coszen"]["median"])
else:
    plot_syst_resolution(cnn["numucc"]["coszen"]["median"], syst_sets, title=a_title, ylabel=a_ylabel, xlabel=a_xlabel,savename=a_savename,savepath=save_path)

a_title = "%s - Cosine Zenith NuMu CC"%title_sets
a_ylabel= '68% Spread'
a_xlabel = "Cosine Zenith"
a_savename = "%s_CosZenNuMuCC_Width"%plot_save_name
if compare_retro:
    plot_syst_resolution(cnn["numucc"]["coszen"]["diff"], syst_sets, title=a_title, ylabel=a_ylabel, xlabel=a_xlabel,savename=a_savename,savepath=save_path,zeroline=False,retro_variables=retro["numucc"]["coszen"]["diff"])
else:
    plot_syst_resolution(cnn["numucc"]["coszen"]["diff"], syst_sets, title=a_title, ylabel=a_ylabel, xlabel=a_xlabel,savename=a_savename,savepath=save_path,zeroline=False)

a_title = "%s - Cosine Zenith NuE CC"%title_sets
a_ylabel= "Bias"
a_xlabel = "Cosine Zenith"
a_savename = "%s_CosZenNuECC_Bias"%plot_save_name
if compare_retro:
    plot_syst_resolution(cnn["nuecc"]["coszen"]["median"], syst_sets, title=a_title, ylabel=a_ylabel, xlabel=a_xlabel,savename=a_savename,savepath=save_path,retro_variables=retro["nuecc"]["coszen"]["median"])
else:
    plot_syst_resolution(cnn["nuecc"]["coszen"]["median"], syst_sets, title=a_title, ylabel=a_ylabel, xlabel=a_xlabel,savename=a_savename,savepath=save_path)

a_title = "%s - Cosine Zenith NuE CC"%title_sets
a_ylabel= '68% Spread'
a_xlabel = "Cosine Zenith"
a_savename = "%s_CosZenNuECC_Width"%plot_save_name
if compare_retro:
    plot_syst_resolution(cnn["nuecc"]["coszen"]["diff"], syst_sets, title=a_title, ylabel=a_ylabel, xlabel=a_xlabel,savename=a_savename,savepath=save_path,zeroline=False,retro_variables=retro["nuecc"]["coszen"]["diff"])
else:
    plot_syst_resolution(cnn["nuecc"]["coszen"]["diff"], syst_sets, title=a_title, ylabel=a_ylabel, xlabel=a_xlabel,savename=a_savename,savepath=save_path,zeroline=False)


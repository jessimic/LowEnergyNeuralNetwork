import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

levels = np.arange(2,7,1)
#level_names = ["Onsite Filter", "Fast Data-MC \n Agreement", r'Fast Noise & $\mu$' +"\nRejection BDTs", "Tighter Muon \nCut", "Final CNN \n Cuts"]
level_names = ["Level2", "Level3", "Level4", "Level5", "FLERCNN \n Selection"]
flavor_names = [r'$\nu_e$' + " CC", r'$\nu_\mu$' + " CC", r'$\nu_\tau$' + " CC", r'$\nu$' + " NC", "atm. " + r'$\mu$', "noise"]
flavors = ["nuecc", "numucc", "nutaucc", "nunc", "mu", "noise"]

rates = {}
rates["nuecc"]   = [1.61, 0.95, 0.84, 0.48, 0.1411]
rates["numucc"]  = [6.16, 3.77, 3.11, 1.39, 0.3522]
rates["nutaucc"] = [0.193, 0.129, 0.12, 0.071, 0.0348]
rates["nunc"]    = [0.86, 0.53, 0.46, 0.23, 0.0667]
rates["mu"]      = [7273, 505, 28.1, 0.93, 0.0033]
rates["noise"]   = [6621, 36.6, 0.26, 0.08,0.0006]

fig, ax = plt.subplots(figsize=(10,7))
ax.set_title("Signal and Background Rates",fontsize=25)
#ax.set_xlabel("Cuts",fontsize=20)
ax.tick_params(axis="y",labelsize=15)
ax.set_xticks(levels)
ax.set_xticklabels(level_names,fontsize=15)
plt.yscale('log')
ax.set_ylabel("Rate (mHz)",fontsize=20)
color = ["red","blue","orange","green","purple","gray"]
marker = ["o","s","*","x","^","p"]
style = ["-","-","-","-","-","-"]
for i in range(0,6):
    name = flavors[i]
    ax.plot(levels,rates[name],label=flavor_names[i],marker=marker[i],markersize=10,color=color[i],linestyle=style[i])
ax.plot(levels[:-1],rates['noise'][:-1],marker=marker[i],markersize=10,color=color[i])
plt.ylim(0.0001,10000)
plt.legend(fontsize=15)
plt.savefig("rates.png")

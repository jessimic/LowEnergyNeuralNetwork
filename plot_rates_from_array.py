import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

levels = np.arange(2,7,1)
#level_names = ["Onsite Filter", "Fast Data-MC \n Agreement", 'Fast Noise &' +"\nMuon Rejection", "Tighter Muon \nCut", "Final CNN \n Cuts"]
level_names = ["Level2", "Level3", "Level4", "Level5", "Final CNN \n Cuts"]
flavor_names = [r'$\nu_e$' + " CC", r'$\nu_\mu$' + " CC", r'$\nu_\tau$' + " CC", r'$\nu$' + " NC", "atm. " + r'$\mu$', "noise"]
flavors = ["nuecc", "numucc", "nutaucc", "nunc", "mu", "noise"]

rates = {}
rates["nuecc"]   = [1.61, 0.95, 0.84, 0.48, 0.174]
rates["numucc"]  = [6.16, 3.77, 3.11, 1.39, 0.453]
rates["nutaucc"] = [0.193, 0.129, 0.12, 0.071, 0.037]
rates["nunc"]    = [0.86, 0.53, 0.46, 0.23, 0.082]
rates["mu"]      = [7273, 505, 28.1, 0.93, 0.004]
rates["noise"]   = [6621, 36.6, 0.26, 0.08,0.00091618]

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
for i in range(0,6):
    name = flavors[i]
    ax.plot(levels,rates[name],label=flavor_names[i],marker=marker[i],markersize=10,color=color[i])
plt.legend(fontsize=15)
plt.savefig("rates.png")

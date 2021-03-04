#!/bin/bash

########## SBATCH Lines for Resource Request ##########
#SBATCH --time=3:59:00             # limit of wall clock time - how long the job will run (same as -t)
#SBATCH --mem=100G            # memory required per allocated CPU (or core) - amount of memory (in bytes)
#SBATCH --job-name pull_labels      # you can give your job a name for easier identification (same as -J)
#SBATCH --output /mnt/home/micall12/LowEnergyNeuralNetwork/make_jobs/plot_CNN/pull_labels.out

########## Command Lines to Run ##########

source /mnt/home/micall12/setup_combo_stable.sh

python /mnt/home/micall12/LowEnergyNeuralNetwork/pull_test_labels_i3_to_hdf5.py -i "/mnt/research/IceCube/jmicallef/FLERCNN_i3_output/oscNext_genie_level7_v02.00_pass2.1?0000.00????_FLERCNN_class.i3.zst" -o /mnt/home/micall12/LowEnergyNeuralNetwork/output_plots/PID_level6_cleanedpulses_IC19_E5to200_30000evtperbin_sigmoid_binarycross_LRe-3DROPe-1EPOCHS50/L7_official/ --variable class

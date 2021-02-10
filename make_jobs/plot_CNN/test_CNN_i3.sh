#!/bin/bash

########## SBATCH Lines for Resource Request ##########
#SBATCH --time=1:59:00             # limit of wall clock time - how long the job will run (same as -t)
#SBATCH --gres=gpu:1                # number of different nodes - could be an exact number or a range of nodes (same as -N)
#SBATCH --mem=27G            # memory required per allocated CPU (or core) - amount of memory (in bytes)
#SBATCH --job-name plot_epochs      # you can give your job a name for easier identification (same as -J)
#SBATCH --output /mnt/home/micall12/LowEnergyNeuralNetwork/make_jobs/plot_CNN/test_CNN_i3.out

########## Command Lines to Run ##########

singularity exec -B /mnt/scratch/micall12:/mnt/scratch/micall12 --nv /mnt/research/IceCube/Software/icetray_stable-tensorflow.sif sh -c '/usr/local/icetray/env-shell.sh python /mnt/home/micall12/LowEnergyNeuralNetwork/CNN_Test_i3.py -i /mnt/scratch/micall12/oscnext_official/149999_v2/oscNext_genie_level6_v02.00_pass2.149999.000025.i3.zst --model_name energy_numu_flat_5_150_CC_26659evtperbin_IC19_oldstartDC_lrEpochs50 -e 152 -n test_add_frame'

#!/bin/bash --login

########### SBATCH Lines for Resource Request ##########

#SBATCH --time=00:59:00             # limit of wall clock time - how long the job will run (same as -t)
#SBATCH --nodes=1                 # number of different nodes - could be an exact number or a range of nodes (same as -N)
#SBATCH --mem=30G            # memory required per allocated CPU (or core) - amount of memory (in bytes)
#SBATCH --job-name @@name@@      # you can give your job a name for easier identification (same as -J)
#SBATCH --output @@log@@

########### Command Lines to Run ##################

source ~/setup_combo_stable.sh

python /mnt/home/micall12/LowEnergyNeuralNetwork/create_training_file_perDOM_nocuts.py  -i $1 -n $2 --emax 200

exit $?

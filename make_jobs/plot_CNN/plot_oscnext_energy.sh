#!/bin/bash

########## SBATCH Lines for Resource Request ##########
#SBATCH --time=1:59:00             # limit of wall clock time - how long the job will run (same as -t)
#SBATCH --gres=gpu:1                # number of different nodes - could be an exact number or a range of nodes (same as -N)
#SBATCH --mem=20G            # memory required per allocated CPU (or core) - amount of memory (in bytes)
#SBATCH --job-name plot_epochs      # you can give your job a name for easier identification (same as -J)
#SBATCH --output /mnt/home/micall12/LowEnergyNeuralNetwork/make_jobs/plot_CNN/plot_epoch_energy.out

########## Command Lines to Run ##########
#source /mnt/home/micall12/setup_anaconda.sh
#source activate tfgpu

INPUT="NuMu_140000_level2_uncleaned_cleanedpulsesonly_newvertexIC7_IC19_contained_flat_95bins_77227evtperbin_CC.lt100.testonly.hdf5"
INDIR="/mnt/research/IceCube/jmicallef/DNN_files/"
OUTDIR="/mnt/home/micall12/LowEnergyNeuralNetwork/"
NAME="numu_flat_E_5_100_CC_uncleaned_cleanedpulsesonly_3millevents_nologcharge_oldvertexDC_lrEpochs50_containedIC19_illume"
TEST="oscnext"
VARIABLE="energy"
EPOCH=119

singularity exec -B /mnt/home/micall12:/mnt/home/micall12 -B /mnt/research/IceCube:/mnt/research/IceCube --nv /mnt/research/IceCube/Software/icetray_stable-tensorflow.sif python ${OUTDIR}CNN_TestOnly.py -i $INPUT -d $INDIR -o $OUTDIR --name $NAME --epoch $EPOCH -t $TEST --first_variable $VARIABLE

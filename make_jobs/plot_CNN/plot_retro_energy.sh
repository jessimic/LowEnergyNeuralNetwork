#!/bin/bash

########## SBATCH Lines for Resource Request ##########
#SBATCH --time=1:59:00             # limit of wall clock time - how long the job will run (same as -t)
#SBATCH --gres=gpu:1                # number of different nodes - could be an exact number or a range of nodes (same as -N)
#SBATCH --mem=27G            # memory required per allocated CPU (or core) - amount of memory (in bytes)
#SBATCH --job-name plot_epochs      # you can give your job a name for easier identification (same as -J)
#SBATCH --output /mnt/home/micall12/LowEnergyNeuralNetwork/make_jobs/plot_CNN/plot_epoch_energy.out

########## Command Lines to Run ##########
#source /mnt/home/micall12/setup_anaconda.sh
#source activate tfgpu

INPUT=$1
INDIR=/mnt/scratch/micall12/training_files/
OUTDIR=/mnt/home/micall12/LowEnergyNeuralNetwork/
NAME=energy_numu_flat_5_150_CC_26659evtperbin_IC19_oldstartDC_lrEpochs50
TEST=$2
VARIABLE="energy"
EPOCH=152

source /mnt/home/micall12/setup_anaconda.sh
source activate tfgpu

#srun singularity exec -B /mnt/home/micall12:/mnt/home/micall12 -B /mnt/research/IceCube:/mnt/research/IceCube --nv /mnt/research/IceCube/Software/icetray_stable-tensorflow.sif 

srun python ${OUTDIR}CNN_TestOnly.py -i $INPUT -d $INDIR -o $OUTDIR --name $NAME --epoch $EPOCH -t $TEST --first_variable $VARIABLE --compare_reco

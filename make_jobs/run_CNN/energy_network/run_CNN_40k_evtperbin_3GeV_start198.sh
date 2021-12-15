#!/bin/bash --login

########## SBATCH Lines for Resource Request ##########

#SBATCH --time=119:59:00             # limit of wall clock time - how long the job will run (same as -t)
#SBATCH --gres=gpu:1                # number of different nodes - could be an exact number or a range of nodes (same as -N)
#SBATCH --mem=40G            # memory required per allocated CPU (or core) - amount of memory (in bytes)
#SBATCH --job-name CNN_energy      # you can give your job a name for easier identification (same as -J)
#SBATCH --output /mnt/home/micall12/LowEnergyNeuralNetwork/make_jobs/run_CNN/CNN_energy_extended4_40k_3GeV_start198epoch.out

########## Command Lines to Run ##########

INPUT="NuMu_genie_149999_final_level6_cleanedpulses_tranformed_IC19_E1to500_CC_all_start_all_end_flat_499bins_40000evtperbin_file??.hdf5"
#INDIR="/mnt/research/IceCube/jmicallef/DNN_files/"
INDIR="/mnt/scratch/micall12/training_files/batch_file/149999/"
OUTDIR="/mnt/home/micall12/LowEnergyNeuralNetwork"
NUMVAR=1
LR_EPOCH=200
LR_DROP=0.5
LR=0.001
OUTNAME="energy_numu_level6_cleanedpulses_IC19_E1to500_40000evtperbin_extended4_LRe-3DROPe-.5EPOCHS${LR_EPOCH}_3GeV_start198epoch"

START=871
END=1600
STEP=7
for ((EPOCH=$START;EPOCH<=$END;EPOCH+=$STEP));
do
    #MODELNAME="$OUTDIR/output_plots/${OUTNAME}/${OUTNAME}_model_final.hdf5"
    MODELNAME="$OUTDIR/output_plots/${OUTNAME}/current_model_while_running.hdf5"
    
    case $EPOCH in
    198)
        singularity exec -B /mnt/home/micall12:/mnt/home/micall12 -B /mnt/research/IceCube:/mnt/research/IceCube -B /mnt/scratch/micall12:/mnt/scratch/micall12 --nv /mnt/research/IceCube/Software/icetray_stable-tensorflow.sif python ${OUTDIR}/CNN_Training.py --input_files $INPUT --path $INDIR --output_dir $OUTDIR --name $OUTNAME -e $STEP --start $EPOCH --variables $NUMVAR --first_variable energy --lr $LR --lr_epoch $LR_EPOCH --lr_drop $LR_DROP --model "$OUTDIR/output_plots/${OUTNAME}/${OUTNAME}_${START}epochs_model.hdf5" --chop_energy --ecut 3
    ;;
    *)
        singularity exec -B /mnt/home/micall12:/mnt/home/micall12 -B /mnt/scratch/micall12:/mnt/scratch/micall12 -B /mnt/research/IceCube:/mnt/research/IceCube --nv /mnt/research/IceCube/Software/icetray_stable-tensorflow.sif python ${OUTDIR}/CNN_Training.py --input_files $INPUT --path $INDIR --output_dir $OUTDIR --name $OUTNAME -e $STEP --start $EPOCH --variables $NUMVAR --model $MODELNAME --first_variable energy --lr $LR --lr_epoch $LR_EPOCH --lr_drop $LR_DROP --chop_energy --ecut 3
    ;;
    esac
done

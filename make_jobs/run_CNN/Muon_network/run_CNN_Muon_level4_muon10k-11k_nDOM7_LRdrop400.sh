#!/bin/bash --login

########## SBATCH Lines for Resource Request ##########

#SBATCH --time=119:59:00             # limit of wall clock time - how long the job will run (same as -t)
#SBATCH --gres=gpu:v100                # number of different nodes - could be an exact number or a range of nodes (same as -N)
#SBATCH --mem=100G            # memory required per allocated CPU (or core) - amount of memory (in bytes)
#SBATCH --job-name CNN_classification      # you can give your job a name for easier identification (same as -J)
#SBATCH --output /mnt/home/micall12/LowEnergyNeuralNetwork/make_jobs/run_CNN/Muon_network/CNN_Muon_level4_7mill_nDOM7_LRdrop400.out

########## Command Lines to Run ##########

INPUT="oscNext_level6_flercnn_pass2.120000_02h-06h_140000_06h-15h_130000_10k-11k.cleanedpulses_transformed_IC19_nocut8hit.no_cuts_file??.hdf5"
#INDIR="/mnt/research/IceCube/jmicallef/DNN_files/"
INDIR="/mnt/scratch/micall12/training_files/batch_file/"
OUTDIR="/mnt/home/micall12/LowEnergyNeuralNetwork"
LR_EPOCH=400
LR_DROP=0.5
LR=0.001
DOM=7
OUTNAME="MuonClassification_level4_7millevents_130000_10k-11k_nDOM${DOM}_LRdrop${LR_EPOCH}"

START=196
END=600
STEP=7
for ((EPOCH=$START;EPOCH<=$END;EPOCH+=$STEP));
do
    #MODELNAME="$OUTDIR/output_plots/${OUTNAME}/${OUTNAME}_${EPOCH}epochs_model.hdf5"
    MODELNAME="$OUTDIR/output_plots/${OUTNAME}/current_model_while_running.hdf5"
    
    case $EPOCH in
    0)
        singularity exec -B /mnt/home/micall12:/mnt/home/micall12 -B /mnt/research/IceCube:/mnt/research/IceCube -B /mnt/scratch/micall12:/mnt/scratch/micall12 --nv /mnt/research/IceCube/Software/icetray_stable-tensorflow.sif python ${OUTDIR}/CNN_Training.py --input_file $INPUT --path $INDIR --output_dir $OUTDIR --name $OUTNAME -e $STEP --start $EPOCH --lr $LR --lr_epoch $LR_EPOCH --lr_drop $LR_DROP --first_var muon --cut_nDOM --nDOM ${DOM}
    ;;
    *)
        singularity exec -B /mnt/home/micall12:/mnt/home/micall12 -B /mnt/research/IceCube:/mnt/research/IceCube -B /mnt/scratch/micall12:/mnt/scratch/micall12 --nv /mnt/research/IceCube/Software/icetray_stable-tensorflow.sif python ${OUTDIR}/CNN_Training.py --input_file $INPUT --path $INDIR --output_dir $OUTDIR --name $OUTNAME -e $STEP --start $EPOCH --model $MODELNAME --lr $LR --lr_epoch $LR_EPOCH --lr_drop $LR_DROP --first_var muon --cut_nDOM --nDOM ${DOM}
    ;;
    esac
done

#!/bin/bash

source /home/users/jmicallef/setup_anaconda.sh 
source activate tfgpu

INPUT="NuMu_140000_level2_uncleaned_cleanedpulsesonly_vertexDC_IC19_flat_95bins_36034evtperbin_file0?_CC.lt100.transformedinputstatic_transformed3output.hdf5"
INDIR="/data/icecube/jmicallef/processed_CNN_files/"
OUTNAME="numu_flat_E_5_100_CC_uncleaned_cleanedpulsesonly_3600kevents_nologcharge_IC19"
OUTDIR="/home/users/jmicallef/LowEnergyNeuralNetwork"
NUMVAR=1

START=0
END=3
STEP=1
for ((EPOCH=$START;EPOCH<=$END;EPOCH+=$STEP));
do
    MODELNAME="$OUTDIR/output_plots/${OUTNAME}/${OUTNAME}_${EPOCH}epochs_model.hdf5"
    echo $MODELNAME
 
    case $EPOCH in
    0)
        python /home/users/jmicallef/LowEnergyNeuralNetwork/CNN_LoadMultipleFiles.py --input_files $INPUT --path $INDIR --output_dir $OUTDIR --name $OUTNAME -e $STEP --start $EPOCH --variables $NUMVAR --no_test True
    ;;
    $END)
        python /home/users/jmicallef/LowEnergyNeuralNetwork/CNN_LoadMultipleFiles.py --input_files $INPUT --path $INDIR --output_dir $OUTDIR --name $OUTNAME -e $STEP --start $EPOCH --variables $NUMVAR --model $MODELNAME --no_test False
    ;;
    *)
        python /home/users/jmicallef/LowEnergyNeuralNetwork/CNN_LoadMultipleFiles.py --input_files $INPUT --path $INDIR --output_dir $OUTDIR --name $OUTNAME -e $STEP --start $EPOCH --variables $NUMVAR --model $MODELNAME --no_test True
    ;;
    esac
done

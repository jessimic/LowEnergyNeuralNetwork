#!/bin/bash

source ~/setup_anaconda.sh

MYDIR="/mnt/home/micall12/LowEnergyNeuralNetwork/"
OUTDIR="/mnt/home/micall12/LowEnergyNeuralNetwork/output_plots/"
PLOT1="TruthRecoCosZenith_2DHist_zmax400.png"
PLOT2="CosZenithResolutionSlices_EnergyBinned_ylim.png"
PLOT3="CosZenithResolutionSlices_ylim.png"
PLOT4="CosZenithResolution.png"
PLOT5="TruthRecoCosZenith_2DHist.png"
PLOT6="CosZenithResolution_xlim.png"
DIR="numu_flat_Z_5_100_CC_uncleaned_cleanedpulsesonly_3600kevents_nologcharge_IC19_lre-4"
ROWS=2
EPOCHS="399 504 602 700"

python ${MYDIR}put_plots_together.py -f $PLOT1 -i $DIR -d $OUTDIR --rows $ROWS -e $EPOCHS

python ${MYDIR}put_plots_together.py -f $PLOT2 -i $DIR -d $OUTDIR --rows $ROWS -e $EPOCHS

python ${MYDIR}put_plots_together.py -f $PLOT3 -i $DIR -d $OUTDIR --rows $ROWS -e $EPOCHS

python ${MYDIR}put_plots_together.py -f $PLOT4 -i $DIR -d $OUTDIR --rows $ROWS -e $EPOCHS

python ${MYDIR}put_plots_together.py -f $PLOT5 -i $DIR -d $OUTDIR --rows $ROWS -e $EPOCHS

python ${MYDIR}put_plots_together.py -f $PLOT6 -i $DIR -d $OUTDIR --rows $ROWS -e $EPOCHS

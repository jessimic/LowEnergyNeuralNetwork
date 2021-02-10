#!/bin/bash

source ~/setup_anaconda.sh

MYDIR="/mnt/home/micall12/LowEnergyNeuralNetwork/"
OUTDIR="/mnt/home/micall12/LowEnergyNeuralNetwork/output_plots/"
PLOT1="TruthRecoEnergy_2DHist.png"
PLOT2="EnergyResolutionSlicesFrac.png"
PLOT3="EnergyResolution.png"
PLOT4="EnergyResolutionSlicesFrac_ylim.png"
PLOT5="TruthRecoEnergy_2DHist_zmax2000.png"
PLOT6="EnergyResolution_xlim.png"
DIR="numu_flat_E_5_100_CC_uncleaned_cleanedpulsesonly_3600kevents_nologcharge_IC19_lre-4"
ROWS=2
EPOCHS="504 602 700 798"

python ${MYDIR}put_plots_together.py -f $PLOT1 -i $DIR -d $OUTDIR --rows $ROWS -e $EPOCHS

python ${MYDIR}put_plots_together.py -f $PLOT3 -i $DIR -d $OUTDIR --rows $ROWS -e $EPOCHS

python ${MYDIR}put_plots_together.py -f $PLOT4 -i $DIR -d $OUTDIR --rows $ROWS -e $EPOCHS

python ${MYDIR}put_plots_together.py -f $PLOT5 -i $DIR -d $OUTDIR --rows $ROWS -e $EPOCHS

python ${MYDIR}put_plots_together.py -f $PLOT6 -i $DIR -d $OUTDIR --rows $ROWS -e $EPOCHS

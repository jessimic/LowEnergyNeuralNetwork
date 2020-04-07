#!/bin/bash

source /home/users/jmicallef/setup_anaconda.sh

PLOT1="TruthRecoEnergy_2DHist.png"
PLOT2="EnergyResolutionSlicesFrac.png"
PLOT3="EnergyResolution.png"
DIR="numu_flat_E_5_100_CC_uncleaned_cleanedpulsesonly_3600kevents_nologcharge_IC19"
ROWS=3
EPOCHS="98 203 301 399 504 602 700 798 903"

python /home/users/jmicallef/LowEnergyNeuralNetwork/put_plots_together.py -f $PLOT1 -i $DIR --rows $ROWS -e $EPOCHS

python /home/users/jmicallef/LowEnergyNeuralNetwork/put_plots_together.py -f $PLOT2 -i $DIR --rows $ROWS -e $EPOCHS

python /home/users/jmicallef/LowEnergyNeuralNetwork/put_plots_together.py -f $PLOT3 -i $DIR --rows $ROWS -e $EPOCHS

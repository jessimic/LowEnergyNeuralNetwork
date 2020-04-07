#!/bin/bash

source /home/users/jmicallef/setup_anaconda.sh
source activate tfgpu

INPUT="NuMu_140000_level2_uncleaned_cleanedpulsesonly_vertexDC_IC19_flat_95bins_36034evtperbin_CC.lt100.transformedinputstatic_transformed3output.testonly.hdf5"
OUTDIR="/home/users/jmicallef/LowEnergyNeuralNetwork/"
NAME="numu_flat_E_5_100_CC_uncleaned_cleanedpulsesonly_3600kevents_nologcharge_IC19"
#EPOCH=406
TEST="oscnext"
VARIABLE="energy"

#98 203 301 399 504 602 700 798 903
for EPOCH in 98 203 301 399 504 602 700 798 903;
do
    python /home/users/jmicallef/LowEnergyNeuralNetwork/CNN_TestOnly.py -i $INPUT -o $OUTDIR --name $NAME --epoch $EPOCH -t $TEST --first_variable $VARIABLE
done

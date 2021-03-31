#!/bin/bash

#INPUTFILES="/mnt/research/IceCube/le_osc/forJessie/L6/NuMu_genie_149999_0[0,1]????_*.zst"
#INPUTFILES="/mnt/research/IceCube/jmicallef/simulation/level6/129999/NuE_genie_129999_00*.zst"
INPUTFILES="/mnt/research/IceCube/jmicallef/official_oscnext/level7/140000/oscNext_genie_level7_v02.00_pass2.140000.*.i3.zst"
#INPUTFILES="/mnt/research/IceCube/jmicallef/official_oscnext/level6/1[2,4]0000/oscNext_genie_level6.5_v02.00_pass2.1[2,4]0000.*.i3.bz2"
#INPUTFILES="/mnt/research/IceCube/jmicallef/official_oscnext/newGCD/1[2,4]8885/oscNext_genie_level7_v02.00_pass2.1[2,4]8885.00000?.i3.zst"
#INPUTFILES="/mnt/research/IceCube/jmicallef/FLERCNN_i3_output/oscNext_genie_level7_v02.00_pass2.1?0000.00????_FLERCNN_class.i3.zst"
FILEPATH=/mnt/home/micall12/LowEnergyNeuralNetwork/make_jobs/i3_test
LOG_FOLDER=$FILEPATH/logs
JOB_FOLDER=$FILEPATH/slurm

#Settings for test
VARIABLE="energy"
FACTOR=100
MODEL_NAME=energy_numu_flat_1_500_level6_cleanedpulses_IC19_CC_20000evtperbin_lrEpochs50_extended
EPOCH=152
MODEL_NAME2=PID_level6_cleanedpulses_IC19_E5to200_30000evtperbin_sigmoid_binarycross_LRe-3DROPe-1EPOCHS50
EPOCH2=192
MODEL_NAME3=zenith_numu_lrEpochs64_lrInit0.001_lrDrop0.6_weighted
EPOCH3=656

[ ! -d $LOG_FOLDER ] && mkdir $LOG_FOLDER
[ ! -d $JOB_FOLDER ] && mkdir $JOB_FOLDER

COUNT=0
echo $JOB_FOLDER
for file in $INPUTFILES;
do
    name=`basename $file`
    sed -e "s|@@file@@|${file}|g" \
        -e "s|@@log@@|${LOG_FOLDER}/${name}_cpu.log|g" \
        -e "s|@@model_name@@|${MODEL_NAME}|g" \
        -e "s|@@epoch@@|${EPOCH}|g" \
        -e "s|@@variable@@|${VARIABLE}|g" \
        -e "s|@@factor@@|${FACTOR}|g" \
        -e "s|@@model_name2@@|${MODEL_NAME2}|g" \
        -e "s|@@epoch2@@|${EPOCH2}|g" \
        -e "s|@@model_name3@@|${MODEL_NAME3}|g" \
        -e "s|@@epoch3@@|${EPOCH3}|g" \
        < cpu_job_template.sb > $JOB_FOLDER/${name}_cpu.sb
    let COUNT=$COUNT+1
done
echo $COUNT
#cp run_all_here.sh $JOB_FOLDER

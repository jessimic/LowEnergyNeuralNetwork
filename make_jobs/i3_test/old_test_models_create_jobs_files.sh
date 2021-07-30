#!/bin/bash

#INPUTFILES="/mnt/research/IceCube/le_osc/forJessie/L6/NuMu_genie_149999_0[0,1]????_*.zst"
#INPUTFILES="/mnt/research/IceCube/jmicallef/simulation/level6/129999/NuE_genie_129999_00*.zst"
#INPUTFILES="/mnt/research/IceCube/jmicallef/official_oscnext/level7/140000/oscNext_genie_level7_v02.00_pass2.140000.*.i3.zst"
#INPUTFILES="/mnt/research/IceCube/jmicallef/official_oscnext/newGCD/1[2,4]8885/oscNext_genie_level7_v02.00_pass2.1[2,4]8885.00000?.i3.zst"
#INPUTFILES="/mnt/research/IceCube/jmicallef/FLERCNN_i3_output/oscNext_genie_level7_v02.00_pass2.1?0000.00????_FLERCNN_class.i3.zst"
#INPUTFILES="/mnt/research/IceCube/jmicallef/official_oscnext/level6.5/1[2,4]0000/oscNext_genie_level6.5_v02.00_pass2.1[2,4]0000.*.i3.bz2"
#INPUTFILES="/mnt/scratch/micall12/oscnext_official/1?0000/oscNext_*_level6_flercnn_pass2.1?0000.*.i3.zst"
INPUTFILES="/mnt/scratch/micall12/official_oscnext/1?0000/oscNext_*_level7_v02.00_pass2.*.zst"
FILEPATH=/mnt/home/micall12/LowEnergyNeuralNetwork/make_jobs/i3_test
LOG_FOLDER=$FILEPATH/logs
JOB_FOLDER=$FILEPATH/slurm

#Settings for test
VARIABLE="energy"
FACTOR=100
#MODEL_NAME=energy_numu_flat_1_500_level6_cleanedpulses_IC19_CC_20000evtperbin_lrEpochs50_extended
#MODEL_NAME=energy_numu_flat_1_500_level6_cleanedpulses_IC19_CC_20000evtperbin_lrEpochs50_extended2_Apr2021
#MODEL_NAME=energy_numu_level6_cleanedpulses_IC19_E1to500_20000evtperbin_extended4_LRe-3DROPe-1EPOCHS50
#MODEL_NAME=energy_numu_level6_cleanedpulses_IC19_E1to500_20000evtperbin_extended4_LRe-3DROPe-1EPOCHS200
#MODEL_NAME=energy_numu_level6_cleanedpulses_IC19_E1to500_20000evtperbin_extended4_LRe-3DROPe-.5EPOCHS200
#EPOCH=152
#EPOCH=144
#EPOCH=110
#EPOCH=132
#EPOCH=168
#EPOCH=432
MODEL_NAME2=PID_level6_cleanedpulses_IC19_E5to200_30000evtperbin_sigmoid_binarycross_LRe-3DROPe-1EPOCHS50
EPOCH2=192
#MODEL_NAME3=zenith_numu_lrEpochs64_lrInit0.001_lrDrop0.6_weighted
MODEL_NAME3=zenith_FLERCNN
EPOCH3=0
MODEL_NAME4=vertex_FLERCNN
EPOCH4=0
MODEL_NAME5=MuonClassification_level6_flercnn_pass2_285474events
EPOCH5=5

[ ! -d $LOG_FOLDER ] && mkdir $LOG_FOLDER
[ ! -d $JOB_FOLDER ] && mkdir $JOB_FOLDER

COUNT=0
echo $JOB_FOLDER
for file in $INPUTFILES;
do
    name=`basename $file`
    sed -e "s|@@file@@|${file}|g" \
        -e "s|@@log@@|${LOG_FOLDER}/${name}.log|g" \
        -e "s|@@model_name@@|${MODEL_NAME}|g" \
        -e "s|@@epoch@@|${EPOCH}|g" \
        -e "s|@@variable@@|${VARIABLE}|g" \
        -e "s|@@factor@@|${FACTOR}|g" \
        -e "s|@@model_name2@@|${MODEL_NAME2}|g" \
        -e "s|@@epoch2@@|${EPOCH2}|g" \
        -e "s|@@model_name3@@|${MODEL_NAME3}|g" \
        -e "s|@@epoch3@@|${EPOCH3}|g" \
        -e "s|@@model_name4@@|${MODEL_NAME4}|g" \
        -e "s|@@epoch4@@|${EPOCH4}|g" \
        -e "s|@@model_name5@@|${MODEL_NAME5}|g" \
        -e "s|@@epoch5@@|${EPOCH5}|g" \
        < job_template.sb > $JOB_FOLDER/${name}.sb
    let COUNT=$COUNT+1
done
echo $COUNT
#cp run_all_here.sh $JOB_FOLDER

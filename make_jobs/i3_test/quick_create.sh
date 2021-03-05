#!/bin/bash

#INPUTFILES="/mnt/scratch/micall12/oscnext_official/149999/oscNext_genie_level7_v02.00_pass2.149999.*.i3.zst"
#INPUTFILES="/mnt/scratch/micall12/oscnext_official/169999/oscNext_genie_level7_v02.00_pass2.169999.*.i3.zst"
#INPUTFILES="/mnt/research/IceCube/jmicallef/FLERCNN_i3_output/oscNext_genie_level7_v02.00_pass2.149999.00????_FLERCNN.i3.zst"
#INPUTFILES="/mnt/research/IceCube/jmicallef/official_oscnext/oscNext_genie_level7_v02.00_pass2.140000.*.i3.zst"
#INPUTFILES="/mnt/research/IceCube/jmicallef/official_oscnext/oscNext_genie_level7_v02.00_pass2.120000.*.i3.zst"
#INPUTFILES="/mnt/research/IceCube/le_osc/forJessie/L6/NuMu_genie_149999_0[0,1]????_*.zst"
#INPUTFILES="/mnt/research/IceCube/jmicallef/simulation/level6/129999/NuE_genie_129999_00*.zst"
#INPUTFILES="/mnt/research/IceCube/jmicallef/official_oscnext/level7/1[2,4]0000/oscNext_genie_level7_v02.00_pass2.1[2,4]0000.*.i3.zst"
INPUTFILES="/mnt/research/IceCube/jmicallef/official_oscnext/newGCD/1[2,4]8885/oscNext_genie_level7_v02.00_pass2.1[2,4]8885.00000?.i3.zst"
#INPUTFILES="/mnt/research/IceCube/jmicallef/FLERCNN_i3_output/oscNext_genie_level7_v02.00_pass2.1?0000.00????_FLERCNN_class.i3.zst"
FILEPATH=/mnt/home/micall12/LowEnergyNeuralNetwork/make_jobs/i3_test
LOG_FOLDER=$FILEPATH/logs
JOB_FOLDER=$FILEPATH/slurm

[ ! -d $LOG_FOLDER ] && mkdir $LOG_FOLDER
[ ! -d $JOB_FOLDER ] && mkdir $JOB_FOLDER

COUNT=0
echo $JOB_FOLDER
for file in $INPUTFILES;
do
    name=`basename $file`
    sed -e "s|@@file@@|${file}|g" \
        -e "s|@@log@@|${LOG_FOLDER}/$name.log|g" \
        < quick_job.sb > $JOB_FOLDER/${name}.sb
    let COUNT=$COUNT+1
done
echo $COUNT
#cp run_all_here.sh $JOB_FOLDER

#!/bin/bash

#INPUTFILES="/mnt/research/IceCube/le_osc/forJessie/L6/NuMu_genie_149999_0[0,1]????_*.zst"
#INPUTFILES="/mnt/research/IceCube/jmicallef/simulation/level6/129999/NuE_genie_129999_00*.zst"
#INPUTFILES="/mnt/research/IceCube/jmicallef/official_oscnext/level7/140000/oscNext_genie_level7_v02.00_pass2.140000.*.i3.zst"
#INPUTFILES="/mnt/research/IceCube/jmicallef/official_oscnext/newGCD/1[2,4]8885/oscNext_genie_level7_v02.00_pass2.1[2,4]8885.00000?.i3.zst"
#INPUTFILES="/mnt/research/IceCube/jmicallef/FLERCNN_i3_output/oscNext_genie_level7_v02.00_pass2.1?0000.00????_FLERCNN_class.i3.zst"
#INPUTFILES="/mnt/research/IceCube/jmicallef/official_oscnext/level7/1[2,4]0000/oscNext_genie_level7_v02.00_pass2.1[2,4]0000.*.i3.zst"
INPUTFILES="/mnt/scratch/micall12/oscnext_official/140004/oscNext_genie_level6_flercnn_pass2.140004.00????.i3.zst"
#INPUTFILES="/mnt/scratch/micall12/oscnext_official/1?0000/oscNext_*_level6_flercnn_pass2.1?0000.*.i3.zst"
#INPUTFILES="/mnt/scratch/micall12/oscnext_official/1?0000/oscNext_*_level7_v02.00_pass2.*.zst"
#INPUTFILES="/mnt/scratch/micall12/training_files/i3_files/oscNext_genie_level6_flercnn_pass2.1?0000.*i3.zst"
FILEPATH=/mnt/home/micall12/LowEnergyNeuralNetwork/make_jobs/i3_test
LOG_FOLDER=$FILEPATH/logs
JOB_FOLDER=$FILEPATH/slurm/140004_l6
#OUTDIR=/mnt/scratch/micall12/training_files/i3_files/
OUTDIR=/mnt/scratch/micall12/training_files/i3_files/

#Settings for test

[ ! -d $LOG_FOLDER ] && mkdir $LOG_FOLDER
[ ! -d $JOB_FOLDER ] && mkdir $JOB_FOLDER

COUNT=0
echo $JOB_FOLDER
for file in $INPUTFILES;
do
    name=`basename $file`
    sed -e "s|@@file@@|${file}|g" \
        -e "s|@@log@@|${LOG_FOLDER}/${name}.log|g" \
        -e "s|@@outdir@@|${OUTDIR}|g" \
        < job_template_cvmfs.sb > $JOB_FOLDER/${name}.sb
    let COUNT=$COUNT+1
done
echo $COUNT
#cp run_all_here.sh $JOB_FOLDER

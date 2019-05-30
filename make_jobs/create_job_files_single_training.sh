#!/bin/bash

SYST_SET=603
INPUTFILES="/mnt/research/IceCube/jpandre/Matt/level5p/nue/12${SYST_SET}/*"
#INPUTFILES="/mnt/scratch/neergarr/level2/nue/12640/*"
FILEPATH=/mnt/scratch/micall12/training_files


LOG_FOLDER=$FILEPATH/job_logs/L5p_12${SYST_SET}
mkdir $LOG_FOLDER
mkdir $LOG_FOLDER/slurm

COUNT=0
for file in $INPUTFILES;
do
    name=`basename $file`
    echo $name
    sed     -e "s|@@file@@|${file}|g" \
        -e "s|@@log@@|${LOG_FOLDER}/$name.log|g" \
        -e "s|@@name@@|$name|g" \
        < job_template_single_file.sb > $LOG_FOLDER/slurm/$name.slurm
    let COUNT=$COUNT+1
done

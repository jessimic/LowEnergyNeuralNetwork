#!/bin/bash

INPUT="Level5_IC86.2013_genie_numu.014640."
FILEPATH=/mnt/scratch/micall12/training_files

if [[ "x$INPUT" == "x" ]]; then
        INPUT=""
fi

LOG_FOLDER=$FILEPATH/job_logs

COUNT=0
for file in /mnt/research/IceCube/jpandre/Matt/level5/numu/14640/${INPUT}*.i3.bz2;
do
    name=`basename $file`
    #echo $name
    sed     -e "s|@@file@@|${file}|g" \
        -e "s|@@log@@|${LOG_FOLDER}/$name.log|g" \
        -e "s|@@name@@|$name|g" \
        < job_template_single_file.sb > $LOG_FOLDER/pbs/$name.pbs
    let COUNT=$COUNT+1
done

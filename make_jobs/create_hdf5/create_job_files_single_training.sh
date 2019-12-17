#!/bin/bash

SYST_SET=640
#INPUTFILES="/mnt/scratch/f0008480/simulation/level12/NuMu/NuMu_140000_0000??_level2.zst"
#INPUTFILES="/mnt/research/IceCube/jpandre/Matt/level5p/nue/12${SYST_SET}/*"
#INPUTFILES="/mnt/scratch/micall12/oscnext_official/140000/*"
INPUTFILES="/mnt/research/IceCube/jpandre/Matt/level5p/numu/14${SYST_SET}/*"
#INPUTFILES="/mnt/scratch/micall12/mlarson/NuMu_140000_00*_level2.i3.bz2"
#INPUTFILES="/mnt/scratch/neergarr/level2/nue/12640/*"
#INPUTFILES="/mnt/research/IceCube/jmicallef/simulation/NuMu_140000_0000??_level2.zst"
#INPUTFILES="/mnt/research/IceCube/jmicallef/simulation/range_14_412_12100_100200/NuMu_140000_0000??_level2.zst"
#INPUTFILES="/mnt/scratch/micall12/simulation/level12/NuMu/NuMu_140000_0000??_level2.zst"
FILEPATH=/mnt/scratch/micall12/training_files
LOG_FOLDER=$FILEPATH/job_logs/L5_numu_cleaned

[ ! -d $LOG_FOLDER ] && mkdir $LOG_FOLDER
[ ! -d $LOG_FOLDER/slurm ] && mkdir $LOG_FOLDER/slurm

COUNT=0
for file in $INPUTFILES;
do
    name=`basename $file`
    echo $LOG_FOLDER/slurm/$name
    sed -e "s|@@file@@|${file}|g" \
        -e "s|@@log@@|${LOG_FOLDER}/$name.log|g" \
        -e "s|@@name@@|${name}_cleaned|g" \
        < job_template_single_file.sb > $LOG_FOLDER/slurm/$name.sb
    let COUNT=$COUNT+1
done

cp run_all_here.sh $LOG_FOLDER/slurm/

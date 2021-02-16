#!/bin/bash

FILES="NuMu_genie_149999_03[4,5,6,7,8,9]_level6.zst.sb"

for f in $FILES
do
    INDEX=$(echo $FILES | cut -c19-24)
    if [ ! -f "../NuMu_genie_149999_${INDEX}_level6.zst.log" ];
        then
        sbatch $FILE
    fi
done

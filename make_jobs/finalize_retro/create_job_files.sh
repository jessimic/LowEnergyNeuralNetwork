#!/bin/bash

START=1
END=1549

SYST=140000
for ((INDEX=${START};INDEX<=${END};INDEX+=1));
do
    FILE_NR=`printf "%06d\n" $INDEX`
    OUTFILE="/mnt/research/IceCube/jmicallef/official_oscnext/level6/${SYST}/oscNext_genie_level6.5_v02.00_pass2.${SYST}.${FILE_NR}.i3.bz2"
    sed -e "s|@@index@@|${INDEX}|g" \
        -e "s|@@syst@@|${SYST}|g" \
        -e "s|@@file@@|${OUTFILE}|g" \
        < job_template_finalize_retro.sb > job_scripts/slurm/finalize_retro_${SYST}_${INDEX}.sb
done

#        -e "s|@@dir@@|${DIR}|g" \

#!/bin/bash

START=1
END=1549

SYST=140150
#DIR=149999_v2

for ((INDEX=${START};INDEX<=${END};INDEX+=1));
do
    sed -e "s|@@index@@|${INDEX}|g" \
        -e "s|@@syst@@|${SYST}|g" \
        < job_template_finalize_retro.sb > job_scripts/finalize_retro_${SYST}_${INDEX}.sb
done

#        -e "s|@@dir@@|${DIR}|g" \

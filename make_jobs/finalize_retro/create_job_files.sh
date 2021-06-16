#!/bin/bash

#for SYST_SET in 120001 120002 120003 120004 120100 120101 120102 120150;
#for SYST_SET in 121118 141118; #140501 140502 140503 140500;
for SYST_SET in 160000;
do 
    SYST=$SYST_SET
    #FILEDIR="/mnt/scratch/micall12/oscnext_official/${SYST}"
    FILEDIR="/mnt/research/IceCube/jmicallef/official_oscnext/level6/${SYST}"
    FILENAME="oscNext_genie_level6_v02.00_pass2.${SYST}.*.i3.zst"
    COUNTER=0
    FILES=$FILEDIR/$FILENAME

    for f in $FILES;
    do
        INDEX=$(echo $f | cut -c88-93)
        OUTFILE=$FILEDIR/oscNext_genie_level6.5_v02.00_pass2.${SYST}.${INDEX}.i3.bz2
        sed -e "s|@@index@@|${INDEX}|g" \
            -e "s|@@syst@@|${SYST}|g" \
            -e "s|@@dir@@|${FILEDIR}|g" \
            -e "s|@@outfile@@|${OUTFILE}|g" \
            < syst_template.sb > job_scripts/slurm/finalize_retro_${SYST}_${INDEX}.sb
        #let COUNTER=COUNTER+1
        if [[ $COUNTER -gt 100 ]];
        then
            break
        fi
    done

done

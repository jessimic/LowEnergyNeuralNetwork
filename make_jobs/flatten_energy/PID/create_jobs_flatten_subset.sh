#!/bin/bash

FLV="E"
NUM=12
BIN_MAX=15000
CUTS="all"

#for THOUS in 0 3 4 5 6 7 8 9 10 11 12 13 14 15 16;
for THOUS in 0 1 2 3 4 5 6 7 8;
do
    FILENUM=`printf "%02d\n" $THOUS`
    infile=Nu${FLV}_genie_${NUM}9999_0${FILENUM}???_level6.zst_cleanedpulses_transformed_IC19.hdf5
    name=PID_Nu${FLV}_genie_${NUM}9999_${FILENUM}k_level6.zst_cleanedpulses_transformed_IC19
    sed -e "s|@@infile@@|${infile}|g" \
        -e "s|@@bin_max@@|$BIN_MAX|g" \
        -e "s|@@name@@|${name}|g" \
        -e "s|@@cuts@@|${CUTS}|g" \
        < job_template_flatten_subset.sb > flatten_Nu${FLV}_${FILENUM}k.sb
done


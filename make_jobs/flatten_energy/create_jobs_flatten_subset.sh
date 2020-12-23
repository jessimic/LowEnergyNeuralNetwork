#!/bin/bash

FLV="Mu"
NUM=14
BIN_MAX=20000
CUTS="CC"

for THOUS in 0 3 4 5 6 7 8 9 10 11 12 13 14 15 16 20 21 22 23 24 30 31 32 33;
do
    FILENUM=`printf "%02d\n" $THOUS`
    infile=Nu${FLV}_genie_${NUM}9999_0${FILENUM}???_level6.zst_cleanedpulses_transformed_IC19.hdf5
    name=Nu${FLV}_genie_${NUM}9999_${FILENUM}k_level6.zst_cleanedpulses_transformed_IC19
    sed -e "s|@@infile@@|${infile}|g" \
        -e "s|@@bin_max@@|$BIN_MAX|g" \
        -e "s|@@name@@|${name}|g" \
        -e "s|@@cuts@@|${CUTS}|g" \
        < job_template_flatten_subset.sb > flatten_Nu${FLV}_${FILENUM}k.sb
done


#!/bin/bash

FLV="Mu"
NUM=14
BIN_MAX=250000
#CUTS="NC"
OUTDIR="single_file/${NUM}9999/"

#for THOUS in 0 3 4 5 6 7 8 9 10 11 12 13 14 15 16 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45;
#for THOUS in 45 46 47 48 49;
#for THOUS in 0 1 2 3 4 5 6 7 8;
for CUTS in NC CC;
do
    #for THOUS in 52 53;
    #for THOUS in 54 55 56 57 58 59 60 61;
    for THOUS in 62 63 64 65;
    do
        FILENUM=`printf "%02d\n" $THOUS`
        infile=Nu${FLV}_genie_${NUM}9999_0${FILENUM}???_level6.zst_cleanedpulses_transformed_IC19.hdf5
        name=Nu${FLV}_genie_${NUM}9999_${FILENUM}k_level6.zst_cleanedpulses_transformed_IC19
        sed -e "s|@@infile@@|${infile}|g" \
            -e "s|@@bin_max@@|$BIN_MAX|g" \
            -e "s|@@name@@|${name}|g" \
            -e "s|@@folder@@|${OUTDIR}|g" \
            -e "s|@@cuts@@|${CUTS}|g" \
            < job_template_flatten_subset.sb > flatten_Nu${FLV}_${CUTS}_${FILENUM}k.sb
    done
done

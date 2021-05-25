#!/bin/bash

#SET=130000
SET=140000

#for THOUS in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19;
for THOUS in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15;
do
    FILENUM=`printf "%02d\n" $THOUS`
    #infile=oscNext_muongun_level6_v02.00_pass2.${SET}.0${FILENUM}???.i3.zst_cleanedpulses_transformed_IC19.hdf5
    #name=oscNext_muongun_level6_v02.00_pass2.${SET}.${FILENUM}k_cleanedpulses_transformed_IC19
    infile=oscNext_genie_level6.5_v02.00_pass2.${SET}.00${FILENUM}??.i3.bz2_cleanedpulses_transformed_IC19.hdf5
    name=oscNext_genie_level6.5_v02.00_pass2.${SET}.${FILENUM}h.cleanedpulses_transformed_IC19
    sed -e "s|@@files@@|${infile}|g" \
        -e "s|@@syst_set@@|${SET}|g" \
        -e "s|@@name@@|${name}|g" \
        < job_template_subset.sb > concat_${SET}_${FILENUM}.sb
done


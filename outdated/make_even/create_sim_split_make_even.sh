#!/bin/bash

FLV="Mu"
NUM=14
BIN_MAX=30000
EMAX=150
CUTS="CC"

for VERTEX in "start_DC"; #"start_DC" "old_start_DC";
    do
    for SIM in 2 3 4 6 7 8;
    do
        infile=Nu${FLV}_${NUM}0000_0000??_level2_sim${SIM}.zst_precleaning_lt200_NOvertex_IC19.hdf5
        name=Nu${FLV}_${NUM}0000_level2_sim${SIM}_IC19_${VERTEX}_precleaning
        sed -e "s|@@infile@@|${infile}|g" \
            -e "s|@@bin_max@@|$BIN_MAX|g" \
            -e "s|@@name@@|${name}|g" \
            -e "s|@@emax@@|${EMAX}|g" \
            -e "s|@@vertex@@|${VERTEX}|g" \
            -e "s|@@cuts@@|${CUTS}|g" \
            < job_template_make_even_simX.sb > make_even_Nu${FLV}_sim${SIM}_${VERTEX}.sb
        infile2=Nu${FLV}_${NUM}0000_0001??_level2_sim${SIM}.zst_precleaning_lt200_NOvertex_IC19.hdf5
        name2=Nu${FLV}_${NUM}0000_level2_sim${SIM}B_IC19_${VERTEX}_precleaning
        sed -e "s|@@infile@@|${infile2}|g" \
            -e "s|@@bin_max@@|$BIN_MAX|g" \
            -e "s|@@name@@|${name2}|g" \
            -e "s|@@emax@@|${EMAX}|g" \
            -e "s|@@vertex@@|${VERTEX}|g" \
            -e "s|@@cuts@@|${CUTS}|g" \
            < job_template_make_even_simX.sb > make_even_Nu${FLV}_sim${SIM}B_${VERTEX}.sb
  done      
done


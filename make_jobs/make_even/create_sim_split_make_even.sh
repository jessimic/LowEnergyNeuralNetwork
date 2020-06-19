#!/bin/bash

FLV="Mu"
NUM=14
BIN_MAX=36499
VERTEX="start_IC7"

for VERTEX in "start_IC7" "start_DC";
    do
    for SIM in 2 3 4 6 7 8;
    do
        infile=Nu${FLV}_${NUM}0000_00????_level2_sim${SIM}.zst_lt200_NOvertex_IC19.hdf5
        name=Nu${FLV}_${NUM}0000_level2_sim${SIM}_IC19_${VERTEX}
        sed -e "s|@@infile@@|${infile}|g" \
            -e "s|@@bin_max@@|$BIN_MAX|g" \
            -e "s|@@name@@|${name}|g" \
            -e "s|@@vertex@@|${VERTEX}|g" \
            < job_template_make_even_simX.sb > make_even_Nu${FLV}_sim${SIM}_${VERTEX}.sb
    done
done


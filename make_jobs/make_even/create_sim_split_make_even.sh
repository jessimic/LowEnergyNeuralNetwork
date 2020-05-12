#!/bin/bash

FLV="E"
NUM=12
BIN_MAX=10000

for SIM in 2 3; #4 6;
do
    infile=Nu${FLV}_${NUM}0000_00????_level2_sim${SIM}.zst_lt200_NOvertex_IC19.hdf5
    name=Nu${FLV}_${NUM}0000_level2_sim${SIM}_IC19
    sed -e "s|@@infile@@|${infile}|g" \
        -e "s|@@bin_max@@|$BIN_MAX|g" \
        -e "s|@@name@@|${name}|g" \
        < job_template_make_even_simX.sb > make_even_Nu${FLV}_sim${SIM}.sb
done


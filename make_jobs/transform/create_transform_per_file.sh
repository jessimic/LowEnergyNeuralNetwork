#!/bin/bash

FILEBASE=Level5p_IC86.2013_genie_numu.014640.000184.i3.bz2_lt200_NOvertex_IC19.hdf5
OUTNAME=Level5p_IC86.2013_genie_numu.014640.IC19
VERTEX="start_IC7"
EMAX=150
START=0
END=7
STEP=1

for ((FILENUM=$START;FILENUM<=$END;FILENUM+=$STEP));
do
    infile=Level5p_IC86.2013_genie_numu.014640.000${FILENUM}??.i3.bz2_lt200_NOvertex_IC19.hdf5
    name=${OUTNAME}_set${FILENUM}_
    sed -e "s|@@file@@|${infile}|g" \
        -e "s|@@emax@@|$EMAX|g" \
        -e "s|@@name@@|${name}|g" \
        -e "s|@@vertex@@|${VERTEX}|g" \
        < job_template_transform.sb > transform_emax${EMAX}_file${FILENUM}.sb
done


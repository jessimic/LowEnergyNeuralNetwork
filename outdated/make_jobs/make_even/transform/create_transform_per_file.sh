#!/bin/bash

FILEBASE=NuMu_140000_level2_extended_IC19_lt200_CC_start_IC7_all_end_flat_195bins_42953evtperbin
OUTNAME=NuMu_140000_level2_extended_IC19_flat_195bins_42953evtperbin
#FILEBASE=NuMu_140000_level2_extended_IC19_old_start_DC_lt150_CC_all_start_all_end_flat_145bins_26659evtperbin
#OUTNAME=NuMu_140000_level2_extended_IC19_old_start_DC_lt150_flat_145bins_26659evtperbin
VERTEX="start_IC7"
EMAX=200
START=0
END=17
STEP=1

for ((FILENUM=$START;FILENUM<=$END;FILENUM+=$STEP));
do
    NUM=$(printf "%02d" $FILENUM)
    infile=${FILEBASE}_file${NUM}.hdf5
    name=${OUTNAME}_file${NUM}_
    sed -e "s|@@file@@|${infile}|g" \
        -e "s|@@emax@@|$EMAX|g" \
        -e "s|@@name@@|${name}|g" \
        -e "s|@@vertex@@|${VERTEX}|g" \
        < job_template_transform.sb > transform_emax${EMAX}_file${NUM}.sb
done


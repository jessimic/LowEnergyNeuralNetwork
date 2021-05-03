#!/bin/bash

#OUTNAME=Level5p_IC86.2013_genie_numu.014640
#Level5p_IC86.2013_genie_numu.014640.000005.i3.bz2_lt200_NOvertex_IC19.hdf5
NAME=oscNext_genie_level7_v01.04_pass2.129999
#NAME=Level5p_IC86.2013_genie_numu.014640
EMAX=200
START=0
END=6
STEP=1

for ((FILENUM=$START;FILENUM<=$END;FILENUM+=$STEP));
do
    NUM=$(printf "%02d" $FILENUM)
    FILES=00${NUM}??
    infile=${NAME}.${FILES}.i3.zst_lt200_NOvertex_IC19.hdf5
    name=${NAME}.IC19_files${NUM}XX.
    sed -e "s|@@file@@|${infile}|g" \
        -e "s|@@emax@@|$EMAX|g" \
        -e "s|@@name@@|${name}|g" \
        < job_template_cut_concat.sb > cut_concat_emax${EMAX}_file${NUM}.sb
done

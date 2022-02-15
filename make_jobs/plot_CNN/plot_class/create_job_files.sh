#!/bin/bash

MODELDIR=MuonClassification_level4_7millevents_130000_10k-11k_nDOM4_LRdrop400
INFILE=compare_muon_epochs_180_to_300.hdf5
#EPOCHS=(180 192 204 216 228 240 252 264 276 288 300)
EPOCHS=(204 216 228 240 252 264 276 288 300)
#INFILE=compare_muon_epochs_60_to_168.hdf5
#EPOCHS=(60 72 84 96 108 120 132 144 156 168)
MU_IDX=(8 9 10 11 12 13 14 15 16 17 18)
T="test"
MU=1999
NUMU=97
NUE=91

for i in 0 1 2 3 4 5 6 7 8; # 9 10;
do
    OUTNAME=${T}sample_${EPOCHS[$i]}epochs
    sed -e "s|@@infile@@|${INFILE}|g" \
        -e "s|@@outname@@|${OUTNAME}|g" \
        -e "s|@@model_name@@|${MODELDIR}|g" \
        -e "s|@@muon_index@@|${MU_IDX[$i]}|g" \
        -e "s|@@numu@@|${NUMU}|g" \
        -e "s|@@nue@@|${NUE}|g" \
        -e "s|@@muon@@|${MU}|g" \
        < plot_template.sb > ${MODELDIR}_${OUTNAME}.sb
done

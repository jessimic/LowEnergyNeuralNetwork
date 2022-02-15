#!/bin/bash

FILES="MuonClassification_level4_7millevents_130000_10k-11k_nDOM4_testsample_*.sb"

for f in $FILES
do
    bash $f
done

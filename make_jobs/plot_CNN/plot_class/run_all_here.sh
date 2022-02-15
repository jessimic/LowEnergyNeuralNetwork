#!/bin/bash

#FILES="MuonClassification_level4_7millevents_130000_10k-11k_nDOM7_testsample_*.sb"
FILES="MuonClassification_level4_7millevents_130000_10k-11k_nDOM4_LRdrop400*sb"

for f in $FILES
do
    bash $f
done

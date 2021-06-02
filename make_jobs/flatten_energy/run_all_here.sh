#!/bin/bash

#FILES="flatten_*.sb"
FILES="flatten_NuMu_CC_7?k.sb"

for f in $FILES
do
    sbatch $f
done

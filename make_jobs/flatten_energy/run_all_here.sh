#!/bin/bash

#FILES="flatten_*.sb"
FILES="flatten_NuMu_?C_*k.sb"

for f in $FILES
do
    sbatch $f
done

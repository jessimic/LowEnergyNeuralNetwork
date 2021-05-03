#!/bin/bash

#FILES="flatten_*.sb"
FILES="flatten_NuE_?C_0?k.sb"

for f in $FILES
do
    sbatch $f
done

#!/bin/bash

FILES="concat_120000_*.sb"

for f in $FILES
do
    sbatch $f
done

#!/bin/bash

FILES="concat_130000_*.sb"

for f in $FILES
do
    sbatch $f
done

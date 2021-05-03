#!/bin/bash

FILES="transform_*200*.sb"

for f in $FILES
do
    sbatch $f
done

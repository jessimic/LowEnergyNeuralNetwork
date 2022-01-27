#!/bin/bash

FILES="*.sb"

for f in $FILES
do
    sbatch $f
done

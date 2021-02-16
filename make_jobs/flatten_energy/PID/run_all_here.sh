#!/bin/bash

FILES="flatten_NuEMu*.sb "

for f in $FILES
do
    sbatch $f
done

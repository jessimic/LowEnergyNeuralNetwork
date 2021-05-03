#!/bin/bash

FILES="make_even_NuMu_sim*.sb"

for f in $FILES
do
    sbatch $f
done

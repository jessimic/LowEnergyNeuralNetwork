#!/bin/bash

FILES="flatten_*.sb "

for f in $FILES
do
    sbatch $f
done

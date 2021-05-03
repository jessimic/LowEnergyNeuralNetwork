#!/bin/bash

FILES="concat_140000_*.sb"

for f in $FILES
do
    sbatch $f
done

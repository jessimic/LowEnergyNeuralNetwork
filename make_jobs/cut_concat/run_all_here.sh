#!/bin/bash

FILES="concat_130000_0?.sb"

for f in $FILES
do
    sbatch $f
done

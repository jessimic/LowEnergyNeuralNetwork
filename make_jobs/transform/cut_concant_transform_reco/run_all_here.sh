#!/bin/bash

FILES="cut_concat_emax200_file??.sb"

for f in $FILES
do
    sbatch $f
done

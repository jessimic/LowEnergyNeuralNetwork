#!/bin/bash

FILES="flatten_NuE*.sb "

for f in $FILES
do
    sbatch $f
done

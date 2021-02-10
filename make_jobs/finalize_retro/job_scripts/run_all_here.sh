#!/bin/bash

FILES="finalize_retro_140003_???.sb"

for f in $FILES
do
    sbatch $f
done

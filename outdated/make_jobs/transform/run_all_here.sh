#!/bin/bash
FILES=transform_emax??0_file?.sb

for FILE in $FILES;
do
    sbatch $FILE
done

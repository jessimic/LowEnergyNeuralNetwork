#!/bin/bash

#FILES="oscNext_genie_level7_v02.00_pass2.120000.000???.i3.zst.sb"
FILES="Nu*_genie_1?9999_000[4,5,6,7,8,9]??_level6.zst.sb"

for f in $FILES
do
    sbatch $f
done

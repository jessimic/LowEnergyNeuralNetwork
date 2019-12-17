#!/bin/bash

for ((i=0;i<30;i++));
do
    sbatch run_subfile_transform.sb $i
done

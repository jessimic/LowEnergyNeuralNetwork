#!/bin/bash

for fileset in 08 09 10 12 13;
do
    sbatch job_template_concat.sb $fileset
done

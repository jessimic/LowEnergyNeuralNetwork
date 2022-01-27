#!/bin/bash

job_file=dagman_createhdf5.dag
INFILE_DIR='/data/ana/LE/oscNext/pass2/genie/level3/149999/'
BASEFILE='oscNext_genie_level3_v02.00_pass2.149999.'
INPUTFILES=${INFILE_DIR}${FILENAMES}
START=1
END=2

echo 'VARS ALL_NODES INFILE_DIR="'${INFILE_DIR}'"' >> $job_file
echo '' >> $job_file
echo 'CONFIG dagman.config' >> $job_file
echo '' >> $job_file

COUNT=1

for ((NUM=${START};NUM<=${END};NUM+=1));
do
    FILE_NR=`printf "%06d\n" $NUM`
    NAME=${BASEFILE}${FILE_NR}.i3.zst
    echo 'JOB file'${COUNT} job_single_file.sub >> $job_file
    echo 'VARS file'${COUNT}' FILENAME="'${NAME}'"' >> $job_file
    echo '' >> $job_file

    let COUNT=$COUNT+1
done


#!/bin/bash

INFILE_DIR=$1
FILENAME=$2
OUTNAME=${FILENAME}

eval `/cvmfs/icecube.opensciencegrid.org/py3-v4.1.0/setup.sh`

CODE_DIR=/data/user/jmicallef/LowEnergyNeuralNetwork/
SCRIPT=create_training_file_perDOM_nocuts.py
OUTFILE_DIR=/data/user/jmicallef/LowEnergyNeuralNetwork/training_files/

/cvmfs/icecube.opensciencegrid.org/py3-v4.1.0/RHEL_7_x86_64/metaprojects/combo/stable/env-shell.sh \
python ${CODE_DIR}${SCRIPT} -i ${INFILE_DIR}/${FILENAME} -n ${OUTNAME} -o ${OUTFILE_DIR} --emax 200

exit $?

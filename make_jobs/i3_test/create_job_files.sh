#!/bin/bash

#INPUTFILES="/mnt/research/IceCube/le_osc/forJessie/L6/NuMu_genie_149999_0[0,1]????_*.zst"
#INPUTFILES="/mnt/research/IceCube/jmicallef/simulation/level6/129999/NuE_genie_129999_00*.zst"
#INPUTFILES="/mnt/research/IceCube/jmicallef/official_oscnext/level7/140000/oscNext_genie_level7_v02.00_pass2.140000.*.i3.zst"
#INPUTFILES="/mnt/research/IceCube/jmicallef/official_oscnext/newGCD/1[2,4]8885/oscNext_genie_level7_v02.00_pass2.1[2,4]8885.00000?.i3.zst"
#INPUTFILES="/mnt/research/IceCube/jmicallef/FLERCNN_i3_output/oscNext_genie_level7_v02.00_pass2.1?0000.00????_FLERCNN_class.i3.zst"
#INPUTFILES="/mnt/research/IceCube/jmicallef/official_oscnext/level7/1[2,4]0000/oscNext_genie_level7_v02.00_pass2.1[2,4]0000.*.i3.zst"
INPUTFILES="/mnt/scratch/micall12/oscnext_official/1?0000/oscNext_*_level6_flercnn_pass2.1?0000.*.i3.zst"
#INPUTFILES="/mnt/scratch/micall12/oscnext_official/1?0000/oscNext_*_level7_v02.00_pass2.*.zst"
FILEPATH=/mnt/home/micall12/LowEnergyNeuralNetwork/make_jobs/i3_test
LOG_FOLDER=$FILEPATH/logs
JOB_FOLDER=$FILEPATH/slurm
#OUTDIR=/mnt/scratch/micall12/training_files/i3_files/
OUTDIR=/mnt/scratch/micall12/training_files/i3_files/muontest_356k/

#Settings for test
VARIABLE="energy"
MODEL_NAME=energy_FLERCNN
MODEL_NAME2=PID_FLERCNN
MODEL_NAME3=zenith_FLERCNN
MODEL_NAME4=Vertex_XYZ_FLERCNN
MODEL_NAME5="../../../../../mnt/home/micall12/LowEnergyNeuralNetwork/output_plots/MuonClassification_level6_flercnn_pass2_356843events//MuonClassification_level6_flercnn_pass2_356843events_30epochs_model"

[ ! -d $LOG_FOLDER ] && mkdir $LOG_FOLDER
[ ! -d $JOB_FOLDER ] && mkdir $JOB_FOLDER

COUNT=0
echo $JOB_FOLDER
for file in $INPUTFILES;
do
    name=`basename $file`
    sed -e "s|@@file@@|${file}|g" \
        -e "s|@@log@@|${LOG_FOLDER}/${name}.log|g" \
        -e "s|@@outdir@@|${OUTDIR}|g" \
        -e "s|@@model_name@@|${MODEL_NAME}|g" \
        -e "s|@@model_name2@@|${MODEL_NAME2}|g" \
        -e "s|@@model_name3@@|${MODEL_NAME3}|g" \
        -e "s|@@model_name4@@|${MODEL_NAME4}|g" \
        -e "s|@@model_name5@@|${MODEL_NAME5}|g" \
        < job_template.sb > $JOB_FOLDER/${name}.sb
    let COUNT=$COUNT+1
done
echo $COUNT
#cp run_all_here.sh $JOB_FOLDER

#!/bin/bash

SYST_SET=0000
#INPUTFILES="/mnt/scratch/f0008480/simulation/level12/NuMu/NuMu_140000_0000??_level2.zst"
#INPUTFILES="/mnt/research/IceCube/jpandre/Matt/level5p/nue/12${SYST_SET}/*"
#INPUTFILES="/mnt/research/IceCube/jpandre/Matt/level5p/numu/14${SYST_SET}/Level*"
#INPUTFILES="/mnt/research/IceCube/jmicallef/simulation/129999/NuE_129999_002???_level2.zst"
#INPUTFILES="/mnt/research/IceCube/jmicallef/simulation/149999/NuMu_149999_0?????_level2.zst"
#INPUTFILES="/mnt/research/IceCube/jmicallef/simulation/NuE_120000_0000??_level2_sim?.zst"
#INPUTFILES="/mnt/scratch/micall12/oscnext_official/149999_v2/oscNext_genie_level3_v02.00_pass2.149999.00????.i3.zst"
#INPUTFILES="/mnt/scratch/micall12/oscnext_official/14${SYST_SET}/oscNext_genie_level7*"
#INPUTFILES="/mnt/scratch/micall12/oscnext_official/149999_v2/oscNext_genie_level6.5*"
#INPUTFILES="/mnt/scratch/micall12/oscnext_official/120000/oscNext_genie_level6_flercnn_pass2.1?0000.*"
#INPUTFILES="/mnt/scratch/micall12/training_files/i3_files/oscNext_*_level7_flercnn_pass2.130000.*"
#INPUTFILES="/mnt/research/IceCube/le_osc/forJessie/L6/NuMu_genie_149999_*.zst"
#INPUTFILES="/mnt/research/IceCube/jmicallef/simulation/level6/1?9999/Nu*_genie_149999_*_level6.zst"
#INPUTFILES="/mnt/scratch/micall12/oscnext_official/139008/oscNext_muongun_level4_v02.00_pass2.139008.*.i3.zst"
#INPUTFILES="/mnt/research/IceCube/jmicallef/official_oscnext/level6/140000/oscNext_genie_level6.5_v02.00_pass2.140000.00????.i3.bz2"
#INPUTFILES="/mnt/research/IceCube/jmicallef/official_oscnext/level7/140000/oscNext_genie_level7_v02.00_pass2.1?0000.00????.i3.*"
#INPUTFILES="/mnt/scratch/micall12/oscnext_official/160000/oscNext_genie_level7_v02.00_pass2.160000.000*.i3*"
#INPUTFILES="/mnt/scratch/micall12/oscnext_official/130000/oscNext_muongun_level6_flercnn_pass2.130000.0*.i3*"
INPUTFILES="/mnt/scratch/micall12/oscnext_official/140000/oscNext_genie_level6_flercnn_pass2.140000.001[2,3,4]??.i3*"
FILEPATH=/mnt/scratch/micall12/training_files
LOG_FOLDER=$FILEPATH/job_logs/120000_3Feb22
outdir=single_file/140000

[ ! -d $LOG_FOLDER ] && mkdir $LOG_FOLDER
[ ! -d $LOG_FOLDER/slurm ] && mkdir $LOG_FOLDER/slurm

COUNT=0
echo $LOG_FOLDER/slurm
for file in $INPUTFILES;
do
    name=`basename $file`
    #echo $LOG_FOLDER/slurm/${name}
    sed -e "s|@@file@@|${file}|g" \
        -e "s|@@log@@|${LOG_FOLDER}/$name.log|g" \
        -e "s|@@name@@|${name}|g" \
        -e "s|@@folder@@|${outdir}|g" \
        < job_i3.sb > $LOG_FOLDER/slurm/${name}.sb
    let COUNT=$COUNT+1
done
echo $COUNT
cp run_all_here.sh $LOG_FOLDER/slurm/

job_file=dagman.dag

#!/bin/bash
FILE_DIR=/data/ana/LE/oscNext/pass2/genie/level7_v01.04
FILE_BASE=oscNext_genie_level7_v01.04_pass2.
FLV_SETS=( 12 14 16)
SYST_SETS=( 0001 0002 )

echo 'CONFIG dagman.config' >> $job_file
echo '' >> $job_file

for FLV in "${FLV_SETS[@]}"; do
	for SYST in "${SYST_SETS[@]}"; do
		INPUTFILES=${FILE_DIR}/${FLV}${SYST}/${FILE_BASE}${FLV}${SYST}.*.i3.zst
		INDEX=0
		for FILE in $INPUTFILES; do
			echo 'JOB '${FLV}${SYST}'index'$INDEX submit.sub >> $job_file
			echo 'VARS '${FLV}${SYST}'index'$INDEX 'FLV="'$FLV'"' 'SYST="'$SYST'"' 'FILE="'$FILE'"' >> $job_file
			echo '' >> $job_file
			INDEX=$((INDEX+1))
		done
	done
done



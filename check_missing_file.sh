SIM=/mnt/research/IceCube/jmicallef/simulation/level6/149999/*.zst
CHECK_DIR=/mnt/scratch/micall12/training_files/single_file/140000/
for FILE in $SIM;
    do
        BASEFILE=$(basename $FILE)
        CHECK_FILE=${CHECK_DIR}/${BASEFILE}_cleanedpulses_transformed_IC19.hdf5
        if [ ! -f "$CHECK_FILE" ]; then
            echo "$CHECK_FILE does not exist"
        fi
    done

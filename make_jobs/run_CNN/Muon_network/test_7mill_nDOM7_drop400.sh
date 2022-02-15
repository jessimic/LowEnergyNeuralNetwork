#!/bin/bash --login

########## SBATCH Lines for Resource Request ##########

#SBATCH --time=02:59:00             # limit of wall clock time - how long the job will run (same as -t)
#SBATCH --gres=gpu:1                # number of different nodes - could be an exact number or a range of nodes (same as -N)
#SBATCH --mem=100G            # memory required per allocated CPU (or core) - amount of memory (in bytes)
#SBATCH --job-name CNN_test_energy      # you can give your job a name for easier identification (same as -J)
#SBATCH --output /mnt/home/micall12/LowEnergyNeuralNetwork/make_jobs/run_CNN/CNN_Muon_7mill_nDOMcut7_drop400.out

########## Command Lines to Run ##########
TESTFILE=oscNext_level6_flercnn_pass2.120000_140000_130000.01h_01h_18-19k.cleanedpulses_transformed_IC19.no_cuts.hdf5
FILEDIR=/mnt/research/IceCube/jmicallef/DNN_files/testing/
OUTNAME=compare_muon_epochs_180_to_300
MUON2=MuonClassification_level4_7millevents_130000_10k-11k_nDOM7_LRdrop400

E_EPOCH=594
ENERGY=energy_numu_level6_cleanedpulses_IC19_E1to500_20000evtperbin_no8hitcut_LRe-3DROPe-.5EPOCHS200_nDOMcut7
PID=PID_level6_cleanedpulses_IC19_E5to200_30000evtperbin_sigmoid_binarycross_LRe-3DROPe-1EPOCHS50
PID_EPOCH=192
ZENITH=flat_zenith_IC_start_IC_end_flat_101bins_5_300_ABC_linear
ZEN_EPOCH=700
VERTEX=VertexTest
VER_EPOCH=232
MUON=MuonClassification_level4_pass2_1793kevents
MUON_EPOCH=108
#MUON_EPOCH2=180
#MUON_EPOCH3=192
MUON_EPOCH4=204
MUON_EPOCH5=216
MUON_EPOCH6=228
MUON_EPOCH7=240
MUON_EPOCH8=252
MUON_EPOCH9=264
MUON_EPOCH10=276
MUON_EPOCH11=288
MUON_EPOCH12=300

singularity exec -B /mnt/home/micall12:/mnt/home/micall12 -B /mnt/research/IceCube:/mnt/research/IceCube --nv /mnt/research/IceCube/Software/icetray_stable-tensorflow.sif python /mnt/home/micall12/LowEnergyNeuralNetwork/CNN_Test_hdf5.py -i ${TESTFILE} -d ${FILEDIR} -o /mnt/home/micall12/LowEnergyNeuralNetwork/output_plots/${MUON2}/ -n ${OUTNAME} --variable_list energy class zenith vertex muon nDOM muon muon muon muon muon muon muon muon muon --epoch_list ${E_EPOCH} ${PID_EPOCH} ${ZEN_EPOCH} ${VER_EPOCH} ${MUON_EPOCH} 1 ${MUON_EPOCH4} ${MUON_EPOCH5} ${MUON_EPOCH6} ${MUON_EPOCH7} ${MUON_EPOCH8} ${MUON_EPOCH9} ${MUON_EPOCH10} ${MUON_EPOCH11} ${MUON_EPOCH12} --modelname_list ${ENERGY} ${PID} ${ZENITH} ${VERTEX} ${MUON} nDOM ${MUON2} ${MUON2} ${MUON2} ${MUON2} ${MUON2} ${MUON2} ${MUON2} ${MUON2} ${MUON2} --factor_list 100 1 1 1 1 1 1 1 1 1 1 1 1 1 1

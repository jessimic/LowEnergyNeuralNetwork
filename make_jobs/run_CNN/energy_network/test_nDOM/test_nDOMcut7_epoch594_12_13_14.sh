#!/bin/bash --login

########## SBATCH Lines for Resource Request ##########

#SBATCH --time=02:59:00             # limit of wall clock time - how long the job will run (same as -t)
#SBATCH --gres=gpu:1                # number of different nodes - could be an exact number or a range of nodes (same as -N)
#SBATCH --mem=100G            # memory required per allocated CPU (or core) - amount of memory (in bytes)
#SBATCH --job-name CNN_test_energy      # you can give your job a name for easier identification (same as -J)
#SBATCH --output /mnt/home/micall12/LowEnergyNeuralNetwork/make_jobs/run_CNN/energy_network/test_nDOM/CNN_energy_20k_nDOMcut7_epoch594_12_13_14.out

########## Command Lines to Run ##########

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
ENDING=XYZ_ending
END_EPOCH=0

singularity exec -B /mnt/home/micall12:/mnt/home/micall12 -B /mnt/research/IceCube:/mnt/research/IceCube --nv /mnt/research/IceCube/Software/icetray_stable-tensorflow.sif python /mnt/home/micall12/LowEnergyNeuralNetwork/CNN_Test_hdf5.py -i oscNext_level6_flercnn_pass2.120000_140000_130000.01h_01h_18-19k.cleanedpulses_transformed_IC19.no_cuts.hdf5 -d /mnt/research/IceCube/jmicallef/DNN_files/ -o /mnt/home/micall12/LowEnergyNeuralNetwork/output_plots/test_energy_12jan22/ -n prediction_values_${E_EPOCH}epochs_12_14_13.01h_01h_18-19k --variable_list energy class zenith vertex muon nDOM ending --epoch_list ${E_EPOCH} ${PID_EPOCH} ${ZEN_EPOCH} ${VER_EPOCH} ${MUON_EPOCH} 1 ${END_EPOCH} --modelname_list ${ENERGY} ${PID} ${ZENITH} ${VERTEX} ${MUON} nDOM ${ENDING} --factor_list 100 1 1 1 1 1 1

#!/bin/bash --login

########## SBATCH Lines for Resource Request ##########

#SBATCH --time=00:59:00             # limit of wall clock time - how long the job will run (same as -t)
#SBATCH --gres=gpu:1                # number of different nodes - could be an exact number or a range of nodes (same as -N)
#SBATCH --mem=100G            # memory required per allocated CPU (or core) - amount of memory (in bytes)
#SBATCH --job-name CNN_test_energy      # you can give your job a name for easier identification (same as -J)
#SBATCH --output /mnt/home/micall12/LowEnergyNeuralNetwork/make_jobs/run_CNN/energy_network/test_nDOM/CNN_energy_20k_nDOMcut9_epoch504.out

########## Command Lines to Run ##########

EPOCH=504
MODEL=energy_numu_level6_cleanedpulses_IC19_E1to500_20000evtperbin_no8hitcut_LRe-3DROPe-.5EPOCHS200_nDOMcut9

singularity exec -B /mnt/home/micall12:/mnt/home/micall12 -B /mnt/research/IceCube:/mnt/research/IceCube --nv /mnt/research/IceCube/Software/icetray_stable-tensorflow.sif python /mnt/home/micall12/LowEnergyNeuralNetwork/CNN_Test_hdf5.py -i oscNext_genie_level6_flercnn_pass2.140000.00-03h.cleanedpulses_transformed_IC19.no_cuts.test_only.hdf5 -d /mnt/research/IceCube/jmicallef/DNN_files/testing/ -o /mnt/home/micall12/LowEnergyNeuralNetwork/output_plots/${MODEL}/ -n prediction_values_${EPOCH}epochs --variable_list energy nDOM --epoch_list ${EPOCH} 1 --modelname_list ${MODEL} nDOM

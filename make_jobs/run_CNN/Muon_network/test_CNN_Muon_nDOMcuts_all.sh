#!/bin/bash --login

########## SBATCH Lines for Resource Request ##########

#SBATCH --time=02:59:00             # limit of wall clock time - how long the job will run (same as -t)
#SBATCH --gres=gpu:1                # number of different nodes - could be an exact number or a range of nodes (same as -N)
#SBATCH --mem=100G            # memory required per allocated CPU (or core) - amount of memory (in bytes)
#SBATCH --job-name CNN_test_energy      # you can give your job a name for easier identification (same as -J)
#SBATCH --output /mnt/home/micall12/LowEnergyNeuralNetwork/make_jobs/run_CNN//CNN_Muon_nDOMcut4_cut7_trainset.out

########## Command Lines to Run ##########

FILE=oscNext_level6_flercnn_pass2.120000_140000_130000.01h_01h_18-19k.cleanedpulses_transformed_IC19.no_cuts.hdf5
INDIR=/mnt/research/IceCube/jmicallef/DNN_files/testing/
#FILE=oscNext_level6_flercnn_pass2.120000_00h_140000_12h-14h_130000_10k-11k.cleanedpulses_transformed_IC19_nocut8hit.testonly.no_cuts.hdf5
#INDIR=/mnt/scratch/micall12/training_files/batch_file/
OUTNAME=compare_muon_testset_nDOM4_72_76_nDOM7_76_80_84

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
MUON2=MuonClassification_level4_2millevents_130000_10.0-10.6k_nDOM4
MUON_EPOCH2=72
MUON_EPOCH3=76
MUON3=MuonClassification_level4_2millevents_130000_10.0-10.6k_nDOM7
MUON_EPOCH4=76
MUON_EPOCH5=80
MUON_EPOCH6=84

singularity exec -B /mnt/home/micall12:/mnt/home/micall12 -B /mnt/research/IceCube:/mnt/research/IceCube -B /mnt/scratch/micall12:/mnt/scratch/micall12 --nv /mnt/research/IceCube/Software/icetray_stable-tensorflow.sif python /mnt/home/micall12/LowEnergyNeuralNetwork/CNN_Test_hdf5.py -i ${FILE} -d ${INDIR} -o /mnt/home/micall12/LowEnergyNeuralNetwork/output_plots/${MUON2}/ -n ${OUTNAME} --variable_list energy class zenith vertex muon nDOM muon muon muon muon muon --epoch_list ${E_EPOCH} ${PID_EPOCH} ${ZEN_EPOCH} ${VER_EPOCH} ${MUON_EPOCH} 1 ${MUON_EPOCH2} ${MUON_EPOCH3} ${MUON_EPOCH4} ${MUON_EPOCH5} ${MUON_EPOCH6} --modelname_list ${ENERGY} ${PID} ${ZENITH} ${VERTEX} ${MUON} nDOM ${MUON2} ${MUON2} ${MUON3} ${MUON3} ${MUON3} --factor_list 100 1 1 1 1 1 1 1 1 1 1 

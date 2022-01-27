FILE=$1
FLV=$2
SYST=$3

echo singularity exec -B /data/:/data/ --nv /data/user/jmicallef/icetray_stable-tensorflow.sif sh -c '/usr/local/icetray/env-shell.sh python /data/user/jmicallef/LowEnergyNeuralNetwork/CNN_Test_i3.py -i '${FILE}' -o /data/user/jmicallef/FLERCNN_i3/'${FLV}${SYST}'/ --model_dir /data/user/jmicallef/LowEnergyNeuralNetwork/output_plots/ --model_name energy_numu_flat_5_150_CC_26659evtperbin_IC19_oldstartDC_lrEpochs50 -e 152'

singularity exec -B /data/:/data/ --nv /data/user/jmicallef/icetray_stable-tensorflow.sif sh -c '/usr/local/icetray/env-shell.sh python /data/user/jmicallef/LowEnergyNeuralNetwork/CNN_Test_i3.py -i '${FILE}' -o /data/user/jmicallef/FLERCNN_i3/'${FLV}${SYST}'/ --model_dir /data/user/jmicallef/LowEnergyNeuralNetwork/output_plots/ --model_name energy_numu_flat_5_150_CC_26659evtperbin_IC19_oldstartDC_lrEpochs50 -e 152'

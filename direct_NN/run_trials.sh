#Bash file to run NN with various settings

python neural_network_LE.py -n 16 16 16
python neural_network_LE.py -n 16 8
python neural_network_LE.py -n 32 16
python neural_network_LE.py -n 128 64 16
python neural_network_LE.py -n 128 64
python neural_network_LE.py -n 128 128 128
python neural_network_LE.py -n 128 128 128 -w True
python neural_network_LE.py -n 128 64 -w True
python neural_network_LE.py -n 32 16 -w True
python neural_network_LE.py -f sigmoid
python neural_network_LE.py -i mytestfile_direct.hdf5 --filename direct
python neural_network_LE.py -n 128 64 -i mytestfile_direct.hdf5 --filename direct
python neural_network_LE.py -n 128 64 16 -i mytestfile_direct.hdf5 --filename direct
python neural_network_LE.py -n 128 64 -i Level5_IC86.2013_genie_numu.014640.000XXX.hdf5  --filename large
python neural_network_LE.py -n 128 64 16 -i Level5_IC86.2013_genie_numu.014640.000XXX.hdf5  --filename large

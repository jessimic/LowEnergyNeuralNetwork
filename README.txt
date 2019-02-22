######################################
# README for Neural Network Project
#       Jessie Micallef
#       Updated: 22 Feb 2019
######################################

### ORDER TO RUN CONVOLUTIONAL NEURAL NETWORK ###
create_training_single_file_perDOM.py
concat_files.py
transform_plot_training_files.py (only if you want quartiles & plots of data)
CNN.py

## CNN.py ##
Convolutional neural network code
    - Reads in hdf5 training file (one)
    - Splits into training & testing sets
    - Transforms data to contained range
    - Runs convolutional network
    - Outputs plots on results

## concat_files.py ##
Concatonates hdf5 files into bigger hdf5 files
    - Used for training data sets
    - Takes in specified files and concats arrays
    - Outputs one large hdf5 file with all data
python concat_files.py -i /mnt/scrach/micall12/training_files/Level5_IC86.2013_genie_numu.014640.000000.i3.bz2.hdf5 -o Level5_IC86.2013_genie_numu.014640.0.hdf5

## create_training_single_file_perDOM.py ##
Creates training files using pulse series per DOM
    - Outputs hdf5 file with two 4D arrays (for DC and IC near DC) and one 2D array (labels)
    - Meant to take in one i3 file and output hdf5 with 4D array information
    - 5 variables per DOM per event
    - Use concatonate after to put all the hdf5 files together
    - Needs IceCube library!
python create_training_single_file_perDOM.py  -i /mnt/research/IceCube/jpandre/Matt/level5/numu/14640/Level5_IC86.2013_genie_numu.014640.000000.i3.bz2 -n Level5_IC86.2013_genie_numu.014640.000000.i3.bz2

## transform_plot_training_files.py ##
Plays with scalers to tranform data, plots results, strips zeros

## make jobs ##
Directory with bash scripts and job templates
    ## create_job_files_single_training.sh ##
    Takes in file names to edit job template to submit create_training_single_file_perDOM.py jobs
    ## job_concat_files.sb ##
    Job that sends in all the files in to concatonate
    ## job_create_files.sb ##
    Runs DNN code
    ## job_template_single_file.sb ##
    job template for create_job_files_single_training.sh

## plot_training_files.py ##
Sloppy code to plot true values of labels and some pulse information from training files


## direct_NN ##
Directory with DNN code
    ## ##
    ## ##
    ## ##

## best_network_test ##
DNN network configuation test

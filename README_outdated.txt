######################################
# README for Neural Network Project
#       Jessie Micallef
#       Updated: 30 May 2019
######################################

### ORDER TO RUN CONVOLUTIONAL NEURAL NETWORK ###
create_training_single_file_perDOM.py
cut_concat_files.py
~ Check_initial_stats.py (optional) ~
transform_training_files.py
CNN_CreateRunModel.py
~ CNN_syst_check.ipynb (optional) ~

## create_training_single_file_perDOM.py ##
Creates training files using pulse series per DOM
    - Outputs hdf5 file with three 4D arrays (data for DC, IC near DC, number pulses per DOM) and three 2D array (labels, reco_labels, initial_stats)
    - Meant to take in one i3 file and output hdf5 with 4D array information
    - 5 variables per DOM per event
    - Use concatonate after to put all the hdf5 files together
    - Needs IceCube library! Load this first before running

    Outputs specifically:
    - features contains: [event number, string index, dom index, [sum charge, time of first hit, time of last hit, charge weighted average of time, charge weighted std of time] ]
    - labels contains: [event number, [ nu energy, nu zenith, nu azimuth, nu time, nu x, nu y, nu z, track length (0 for cascade), isTrack, flavor, type (anti = 1), isCC]]
    - reco_labels contains: [event number, [ nu energy, nu zenith, nu azimuth, nu time, nu x, nu y, nu z]]
    - initial_states contains: [event number, [ count outside used strings, charge outside, count inside, charge inside] ]
    - number_pulses_per_dom contains: [event number, [string index, dom index, # hits] ]
Generic example:
    python create_training_single_file_perDOM.py  -i [path + input file name] -n [base name for output file]
Specific example:
    python create_training_single_file_perDOM.py  -i /mnt/research/IceCube/jpandre/Matt/level5/numu/14640/Level5_IC86.2013_genie_numu.014640.000000.i3.bz2 -n Level5_IC86.2013_genie_numu.014640.000000.i3.bz2

## CNN_CreateRunModel.py ##
Convolutional neural network code
    - Reads in hdf5 training file (one)
    - Splits into training & testing sets
    - Transforms data to contained range
    - Runs convolutional network
    - Outputs plots on results
Generic example:
    python CNN_CreateRunModel.py --input_file [input file name, MUST BE TRANSFORMED ALREADY] --path [path to input file] --name [base name for plot folder and model file name, will add drop, lr, and batch to end]  --epochs [number of epochs] --drop [drop rate, in decimal (< 1.0)]  --lr [learning rate] --batch [batch size] --load_weights [bool, True means it will load weights from old model file] --old_model [hdf5 file to model weights]
Specific example:
    python CNN_CreateRunModel.py --input_file 'Level5_IC86.2013_genie_nue.012640.100.cascade.lt60_vertexDC.transformed.hdf5' --path '/mnt/scratch/micall12/training_files/' --name 'nue_cascade_allfiles' --epochs 30 --drop 0.2 --lr 1e-3 --batch 256 --load_weights False --old_model 'current_model_while_running.hdf5'

## cut_concat_files.py ##
Concatonates hdf5 files into bigger hdf5 files
    - Used for training data sets
    - Takes in specified files and concats arrays
    - Can use cuts: track, cascasde, CC, NC, track CC, track NC, cascade CC, cascade NC, track CC cascade CC, track NC cascade NC
    - Shuffles events randomly (mixes all events)
    - Outputs one large hdf5 file with all data
Generic example:
    python cut_concat_files.py -i [input files] -d [path to input files] -o [base name for output file, end with . or _] -c [cuts, pick from predefined strings] -r [bool if files have old reco, like pegleg, to compare with]
Specific example:
    python concat_files.py -i /mnt/scrach/micall12/training_files/Level5_IC86.2013_genie_numu.014640.000000.i3.bz2.hdf5 -o Level5_IC86.2013_genie_numu.014640.0.hdf5


## transform_training_files.py ##
Applies Robust Scaler to input features
    - Takes in concatted file
    - Transforms input features to set range
Generic example:
    python transform_training_files.py -i [input file name] -d [path for input file] -r [bool, if file has old reco to compare with (i.e. pegleg)]
Specific example:
    python transform_training_files.py -i 'Level5_IC86.2013_genie_nue.012640.100.cascade.lt60_vertexDC.hdf5' -d '/mnt/scratch/micall12/training_files/' -r True


## make jobs ##
Directory with bash scripts and job templates
    ## create_job_files_single_training.sh ##
    Takes in file names to edit job template to submit create_training_single_file_perDOM.py jobs
    - EDIT INPUTFILES to point to correct path
    - EDIT SYST_SET to pick systematic set
    - EDIT FILEPATH for output of job scripts
    ## job_concat_files.sb ##
    Job that sends in all the files in to concatonate
    - EDIT path of IceCube library to source and location of concat_files.py
    ## job_template_single_file.sb ##
    job template for create_job_files_single_training.sh
    - EDIT path of IceCube library to source and location of create_training_single_file_perDOM.py

## plot_training_files.py ##
Sloppy code to plot true values of labels and some pulse information from training files


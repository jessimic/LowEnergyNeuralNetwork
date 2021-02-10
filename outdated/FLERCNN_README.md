# Welcome to FLERCNN: Fast Low Energy Reconstruction using a Convolutional Neural Network

                                                                                                      .y                
                                                                                                      .y                
                                                                                                      .y                
                                                                                                      .y                
                                                                                         -:/:-`   `/sydNhs/.            
                                                                                       .dMMMMMNd+sNMMMMMMMMNy.          
                                                                                      .mMMdo/ohNMMMMMMMMMMMMMN-         
                                                              `-:::-.`                yMMs`    sMMMMMMMMMMMMMMy         
                                                            /hNNNdddmNmy+.           -MMd      oMMMMMMMMMMMMMMy         
                                                           sMMy:`````-sNMNy.         hMN.    /syNMMMMMMMMMMMMN.         
                                                          .MMy -ohh+   -mMMm-      -hMN/     ```.sNMMMMMMMMNmNd/        
                                                          .NMo  ``mM/   .mMMNhssydmNNh-           `:oyhmNMy``.sMy`      
                                                           +MNs::oMM/    oMMMMMMMMMMdso/:--.`         .y:hh/`  hM+      
                                                            :hNMMMNs`   oNMMMdyMMNhsohMMMNNNmmhs+-    .y ```   sMy      
                                                             `..-..    /MMyMMMMMy.`-odMMNMMMdhhdNNdo-`-y`     `mM+      
                                                                       dMN`hMMMy./dNMms:.omMm/``.:dMNmNNmds:`:dMy`      
                                                                      `NMN`oMMMdmMMd+`    .+mNy--dNMMMMMMMMMmNh:   yo   
                                                                       mMM+oMMMMMm/`        `+dNNMMMMMMMMMMMMM:    /M+  
                                                             +/.   `h- oMMNmMMMMd`     .-:///::hMMMMMMMMMMMMMMh    -Mh  
                                                            `MMms-`+Md::dMMMMMMMy  `:sdmNMMMMMNNMMMMMMMMMMMMMMs    +My  
                                                             mMMMNdNMMMNNMMMMMMMy-smMMmyo/:::/ohmMMMMMMMMMMMMh`   -NM/  
                                                             oMMMMMMMMMMMMMMMMMMmNMMMMmddddddyo/:/hNMMMMMMNd/`  `+NMs   
                                                             .MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMNdy++dMMd/-`  ./dMNo`   
                                                              dMMMMMMMMMMMMMMMMMMMmymMMMNdssymMMmNMMmsyMN-`.:ymMNy-     
                                                              dMMMMMMMMMMMMMMMMMMMdhNMMM/` so/MM/:dMMMNMMmdmMMms-       
                                                             `MMMMMMMMMMMMMMMMMNNNMMMMMMmhhMMMMMNmNMMMMMMMMds:`         
                                                             -MMMMMMMMMMMMMMMMM+---/yMMMNmmmmmmmdNMMMMMMMMNo.`          
                                                             :MMMMMMMMMMMMM+yNMm:   `+mMNy/-.``.+NMMMMMMMMMMNdhs:`      
                                                        `.-+smMMMMMMMMMMMMM+ :dMN+`   .odNNmdhhhMMMMMMMMMMMMMMmNMh.     
                                                    `./sdmMMMMMMMMMMMMMMMMMy  `oNMh:`    .:+ossmMMMMMMMMMMMMMMy/MMd     
                                                 .:sdNMMMMMMMMMMMMMMMMMMMMMy    -yNMdo:......:+dMMMMMMMMMMMMMMysMMN`    
                                              ./ymMMMMMMMMMMMMMMMMMMMMMMMMMo      .+ymmmmmmmmmdysNMMMMMMMMMMMMMMNm/     
                                           `:ymMMMMMMMMMMMMMMMMMMMMMMMMMMMM-         `.-:::--.   +NMMMMMMNmy/++/:`      
                                         `/dMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMy                   `:yNNy::+d:-`             
                                        :dMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM.                 `/dMms-   .y                
           `::`                       `sMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM+                /dMmo.     .y                
          :mMN.                      .dMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMN-              sMNs.    `.:+d:-`             
         :NMM+                      `dMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMNh+-.`        /Mm/    -ymMMMMMMNy:           
        `NMMs                       yMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMNdyo:.`   os.    sMMMMMMMMMMMMy`         
        oMMm`                      -MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMd:/ohmNNMMMMMMMMMMh:        /MMMMMMMMMMMMMMo         
        dMMs                       yMMMMMMMMMMMMMMMMMMMMMMMNMMMMMMN.     .-:/osydmNMMMMs`      sMMMMMMMMMMMMMMh         
        NMMo                       NMMMMMMMMMMMMMMMMMMMMMMh-/MMMMM+               `/NMMMy      :MMMMMMMMMMMMMM/         
        mMMd                      .MMMMMMMMMMMMMMMMMMMMMMh`  mMMMM/                 oMMMM-      /mMMMMMMMMMMN+          
        oMMM+                     .MMMMMMMMMMMMMMMMMMMMMMm   sMMMMd`                `dNNd`       `/hNNMMNNh+`           
        `dMMMs`                   -MMMMMMMMMMMMMMMMMMMMMMs   -MMMMMs                  .`             `:h`               
         .dMMMm+.                 -MMMMMMMMMMMMMMMMMMMMMm`    oMMMMMo                                 .y                
           +NMMMMdy+:-.``      `:omMMMMMMMMMMMMMMMMMMMMN-`     sMMMMMy`                               .y                
            `/hNMMMMMMMMNNNNNNNMMMMMMMMMMMMMMMMMMMMMMMMMMMmo-.-.yMMMMMh`                              .y                
               `:oyhmNMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMh                              .y                
                       ``..-----------------:::::::::::::::::::::::::::-                               .                

## Brief Description
FLERCNN is optimized to reconstruct the energy and cosine zenith of events with energies mainly in the 10s of GeV. It uses DeepCore and the inner most IceCube strings to summarize the data per event with summary variables per DOM (charge and times of pulses). 


## Running on the HPCC

### Environments
To create the processing scripts, you will need IceTry access in the IceCube software to convert the i3 formatted files. The rest of the components use hdf5 files, so only a python environment is needed for remaining parts of processing. To train the network, Keras with a Tensorflow backend is needed. Tensorflow suggests using anaconda to install, though anaconda does not play nice with the IceCube metaprojects. Since the processing steps are separated, you can load difference environment for different steps.

- Create Training scripts (i3 --> hdf5)
	- Need IceCube software!
	- Option 1: cvmfs
		- `source /mnt/home/micall12/setup_combo_stable.sh`. Does the following steps:
			- `eval /cvmfs/icecube.opensciencegrid.org/py3-v4.1.0/setup.sh`
			- `module purge`
			- `/cvmfs/icecube.opensciencegrid.org/py3-v4.1.0/RHEL_7_x86_64/metaprojects/combo/stable/env-shell.sh`
	- Option 2: singularity container
		- `singularity exec -B /mnt/home/micall12:/mnt/home/micall12 -B /mnt/research/IceCube:/mnt/research/IceCube --nv /mnt/research/IceCube/Software/icetray_stable-tensorflow.sif python ...`
		- Must replace your home netID for `micall12`
		- Can also start a singularity shell, then run scripts interactively inside
- CNN Training and Testing:
	- Need tensorflow, keras, and python!
	- Option 1: singularity container
		- `singularity exec -B /mnt/home/micall12:/mnt/home/micall12 -B /mnt/research/IceCube:/mnt/research/IceCube --nv /mnt/research/IceCube/Software/icetray_stable-tensorflow.sif python ...`
		- Must replace your home netID for `micall12`
		- Can also start a singularity shell, then run scripts interactively inside
		- Advantage of this option: can send container to any cluster with code and should be able to run
		- Disadvantage of option: container is static, difficult to update software
	- Option 2: anaconda
		- Install anaconda
		- Create a virtual env `conda create -n tfgpu`
		- Go into virtual env `conda activate tfgpu`
		- Add necessary libraries:
			- `pip install tensorflow`
			- `pip install keras`
			- `pip install matplotlib`
		- Advantage of this option: easier to update, not a "static container"
		- Disadvantage of option: Tensorflow's GPU interaction has been known to stop working suddenly on HPCC, and the only solution found so far is to reinstall anaconda and then recreate virtual env

### Submitting Jobs
- Example job submission scripts in `make_jobs`
- Create HDF5: Making the training files (i3-->hdf5)
	- Most efficient to run in parallel
		- Can glob, but concat step takes a while
		- Each file takes a few minutes only
	- `create_job_files_single_training.sh` makes a job script for every file in the specified folder
	- `job_template...` should have all the flags/args you want for `create_training` code 
	- You can submit all these as jobs or run them locally with bash
- Run CNN: Training the CNN
	- Use `singularity` container to run CNN
	- Kill and recall tensorflow script every handful of epochs (memory leak that adds ~2min per epoch otherwise)
		- STEPS should correspond to the number of files in your data set
	- Assumes there is a folder called `output_plots` in your main directory
	- Should request GPU and about 27G

### Testing Jobs before Submission
- Option 1: interactive job
	- Request interactive job `salloc --time=03:00:00 --mem=27G --gres=gpu:1`
	- If not using singularity, load tensorflow environment as usual
	- To access the GPU, need `srun` before calling the code: `srun singularity ...` or `srun python`
- Option 2: run on development node
	- If not using singularity, load tensorflow environment as usual
	- GPU dev nodes (k80 or k20) need `srun` before calling the code `srun singularity ...`
	- Can run without the GPU (training will take hours per epoch), just to check that all the paths and setup for the network are working!

## Description of major code components

### Order to run code from creating new training data --> testing CNN
1. Create hdf5 files from i3 files (`create_training_file_perDOM_nocut.py`)
2. Flatten energy distribution (optional, for training) (`flatten_energy_distribution.py`)
3. Cut, concatenate, and transform data for easier training (`cut_concat_transform_separate_files.py`)
4. Train the CNN (`CNN_LoadMultipleFiles.py`)
5. Make testonly file (`make_test_file.py`)
5. Test the CNN (`CNN_TestOnly.py`)


### Training and Testing Scripts

#### Typical CNN Training/Testing Procedure
- Submit `CNN_LoadMultipleFiles.py` to train for many epochs (100s)
- Check progress using `plot_loss_from_column.py` during training
- Use `CNN_TestOnly.py` to check results at any time (using oscnext test)
	- When to check results: if validation curve leveling off or want to check results at specific epoch
	- Save PegLeg or Retro test once you have settled on final model

#### Description of Scripts	
- `CNN_LoadMultipleFiles.py` - used for training the CNN
	- Takes in multiple training files (of certain file pattern), loads one and trains for an epoch before loading the next file for the next epoch
		- Makes sure not too much data is stored at once (~30G)
		- Shuffles within the file between full file pass sets
		- Expects a train, test, validate set to load data in
	- Learning rate adjustable with parserr args
	- Batch size and dropout currently constant
	- Loss functions:
		- Energy = mean_absolute_percentage_error
		- Zenith = mean_squared_error
		- Track Length = mean_squared_error (NOT optimized)
	- Loads model architecture from `cnn_model.py`
	- Functionality:
		- Can train for energy or zenith alone
			- parser arg option --variables 1 and --first_variable "zenith" or "energy"
		- Can train for energy, zenith, and/or track at the same time
			- parser arg option --variables 2 or 3
			- Can only do order energy then zenith then track
		- Starts at the given epoch, runs for the number of epochs specified
			- Helps to continue training model if killed (loads weights from given model)
			- Helps to kill and reload tensorflow to avoid memory leak
		- Can plot "test", comparing to oscnext flat test sample
	- Appends loss to `saveloss_currentepoch.txt` file in output directory
	- Look at `make_jobs/run_CNN/` for slurm submission examples and `make_jobs_condor/run_CNN/` for HTCondor examples

- `plot_loss_from_column.py` - plot loss from column sorted saveloss txt file
	- `CNN_LoadMultipleFiles.py` output column sorted saveloss txt file
	- File also stores time to train per epoch and per loading data file + training per epoch
	- Order of loss, validation loss, etc. varies on number of variables training for (uses dict keys to pull correct values)
	- Functionality:
		- Can give ylim as ymin and ymax, parser args
		- Can specify which epoch to plot until, to shorten x axis (parser arg)
		- Manually can change number of files to average over and start at
			- Set to 7 files to average over
			- Set to start plotting avg plots at epoch 49
	- Outputs plots to outdir folder
		- `TrainingTimePerEpoch.png`
		- `loss_vs_epochs.png`
		- `AvgLossVsEpoch.png`
		- `AvgRangeVsEpoch.png`
	- Look at `make_jobs/plot_CNN/` for slurm submission examples and `make_jobs_condor/plot_CNN/` for HTCondor examples

-  `CNN_TestOnly.py` - used for testing the CNN
	- Takes in one file
		- Use `make_test_file.py` to make multiple files into one `testonly` set
		- See Processing Scripts section of README for more information
	- Evauluates network at given model:
		- Parser arg the directory name where model is stored
		- Parser arg the epoch number of the model to grab
	- Can compare to old reco
		- Parser arg boolean flag `--compare_reco`
		- Give test name `--test PegLeg` or "Retro". Use "oscnext for no comparison
	- Need to load in same model as training (`cnn_model.py`)
	- Creates many plots and outputs to model directory, with subfolder that has the test name and epoch number (gives ability to perform multiple test types on multiple epoch stages)
	- Look at `make_jobs/plot_CNN/` for slurm submission examples and `make_jobs_condor/plot_CNN/` for HTCondor examples



### Processing Scripts (Getting Data into Training/Testing Format)

#### Example Processing Flat Training Sample
- Generate hdf5 files with `create_training_file_perDOM_nocut.py`
	- Best performance is running in parallel (give script one infile)
	- Use setup in `/make_jobs/create_hdf5/` to generate many job scripts
	- `create_job_files_single_training.sh` makes a job script for every file in the specified folder
	- `job_template...` should have all the flags/args you want for `create_training` code 
	- You can submit all these as jobs or run them locally with bash
- To flatten energy distribution, use `flatten_energy_distribution.py`
	- Use `check_energybins.py` to apply cuts and find optimal input for "max_per_bins" arg
	- Concatenating is the time bottleneck, so suggest to run in segments
	- Don't forget to cut! Beware of the defaults
		- Energy max
		- Energy min
		- Event type (CC, NC, track, cascade, etc.)
		- Vertex start position
		- Containment cut (i.e. end position)
	- MAKE SURE TO SHUFFLE BEFORE SEPARATING INTO VARIOUS OUTFILES
	- Separating into outfiles will speed up the transform processing (next step), so it is suggested to do here (AFTER SHUFFLING)
	- Example scripts in `/make_jobs/make_even/`
	- Example for batches of submission scripts (one for each sim number): 
```python $INDIR/flatten_energy_distribution.py -i NuMu_140000_000???_level2_sim2.zst_lt200_NOvertex_IC19.hdf5 -o NuMu_140000_level2_sim2_IC19 
	--emax 100 --emin 5 --max_per_bin 36034 --cuts CC --shuffle False 
	--num_out 1 --start "old_start_DC" --end "all_end"
	```
	- Example for final level submission script (put 7 intermediate files together):
```python $INDIR/flatten_energy_distribution.py -i NuMu_140000_level2_sim?_IC19lt150_CC_start_IC7_all_end_flat_145bins_46240evtperbin.hdf5 -o NuMu_140000_level2_IC19_ 
--emax 100 --emin 5 --max_per_bin 36034 --cuts CC --start old_start_DC --shuffle True 
--num_out 7
```
	- Scripts for HPCC in `make_jobs/
- Cut, concatenate, and transform data with `cut_concat_transform_separate_files.py`
	- If you already cut and concatenated with the `flatten_energy_distribution`, then you won't need to use that functionality here
	- You can use the cut and concat features if skipped flattening (like for a testing sample)
	-  Don't forget to cut! Not as important if already cut during flatten, but beware of the defaults
		- Energy max
		- Energy min
		- Event type (CC, NC, track, cascade, etc.)
		- Vertex start position
		- Containment cut (i.e. end position)
	- MAKE SURE TO SHUFFLE BEFORE SEPARATING INTO VARIOUS OUTFILES
		- You don't need to shuffle again, if you already did in the last step
	- Example scripts in `/make_jobs/make_even/transform` and `/make_jobs/concat`
	- Example for transforming one file
```python $INDIR/cut_concat_transform_separate_files.py -i NuMu_140000_level2_IC19_lt150_CC_start_IC7_all_end_flat_145bins_46240evtperbin_file00.hdf5 
	-o NuMu_140000_level2_IC19_lt150_CC_start_IC7_all_end_flat_145bins_46240evtperbin_file00
	-c CC --emax 100 --emin 5 --shuffle False --trans_output True --tmax 200.0 --num_out 1```
	

#### Description of Scripts
- `create_training_file_perDOM_nocut.py` - data from i3 to output hdf5
	- Takes in one or many i3 files
	- Extracts pulses, summarizes pulses per DOM, discards pulses not passing cuts
		- Only keep events with ONE DeepCore SMT3 trigger
		- Only keep pulses in [-500, 4000] ns of trigger
		- Event has SRTTWOfflinePulsesDC frame (cleaned pulses)
		- Event header has InIceSplit
		- First particle in the MCTree must be a neutrino
	- Shifts all pulses by trigger time (all null hits at -20000, moved to more reasonable position during transform)
	- Functionality:
		- Can do any number of IC strings (put numbers in IC_near_DC_strings list)
		- Cleaned (SRTTWOfflinePulsesDC) or uncleaned (SplitInIcePulses) pulses, parser arg
		- Output reco labels array from pegleg or retro reco, parser arg
		- Emax cut (cut any event above specified energy max)
	- REQUIRES THE ICECUBE SOFTWARE (see `make_jobs/create_hdf5/` for example on hpcc)
	- Outputs the input features [evt # x 8 x 60 x 5] for DC and [evt # x 19 x 60 x 5] for IC and the output features [evt # x 12] for truth labels
		- Output labels: [nu energy, nu zenith, nu azimuth, nu time, nu x, nu y, nu z, track length (0 for cascade), isTrack, flavor, type (anti = 1), isCC]
		- Input feature summary variables: [sum charges, time first pulse, Time of last pulse, Charge weighted mean time of pulses, Charge weighted standard deviation of pulse times]
Can also put out a [evt # x 7] labels for old reco
		- Reco labels: [nu energy, nu zenith, nu azimuth, nu time, nu x, nu y, nu z, track length]

- `create_training_single_file_perDOM.py` - data from i3 to output hdf5 WITH MORE CUTS
	- Same information as for `create_training_file_perDOM_nocut.py`
	- Additional cuts:
		- Always has a starting vertex cut, can be DC or IC7
		- Optional ending position cut, parser arg
		- Fraction drop of cascade or track (automatically off)
	- Downside: may need to rerun if the vertex or ending position isn't set, can make these cuts later if you use `create_training_file_perDOM_nocut.py`

- `flatten_energy_distribution.py` - reads in multiple files and saves events only if that current energy bin has not reached its maximum
	- Throws away any event that is "superfluous" after maximum reached per bin
	- User specifies maximum per bin (use `check_energybins.py` to determine) and GeV size of bins
	- Cuts can be applied to get accurate number of events in sample:
		- Vertex cut start, parser arg
		- Ending position cut, parser arg
		- Energy maximum, parser arg
		- Energy minimum, parser arg
	- Can output more than 1 file, to split events between (so files are not so large)
	- MUST shuffle if using NUM_OUT > 1
	- Outputs `features_DC`, `features_IC`, `labels`, `num_pulses_per_dom`, `trigger_time`, and optional `reco_labels` to use for statitics/comparisons later

- `cut_concat_transform_separate.py`

- `cut_concat_separate_files.py`

- `make_test_file.py` - Concatenates arrays in file(s) and outputs one "testonly" file
	- Use to create large, many file, flat datasets into one testing file
		- Only pulls from the testing sets in this case (unused during training)
		- Give multiple files to pull from, puts together the training sets from these files
	- Use to create large, single file with all the data for testing in the Old Reco sets
		- Old samples like DRAGON or newly generated OscNext are not flat so we use them ONLY for training
		- Instead of writing a different pipeline to create these sets, they go through the regular pipeline
		- This last step takes all "Training", "Validation" and "Testing" arrays and makes them into one large "Testing" array
		- USE THE FLAG `--old_reco` to have the script put all train, test, validate into one

	
### Scripts to Evaluate/Check Data

- `check_energybins.py` - quickly read in many files and make distribution of energy
	- Used to determine how to make flat sample
	- Prints the number of events in the smallest populated bin
	- Use that information for `flatten_energy_distribution.py`
	- Cuts can be applied to get accurate number of events in sample:
		- Vertex cut start, parser arg
		- Ending position cut, parser arg
		- Energy maximum, parser arg
		- Energy minimum, parser arg
	- Able to use for transformed files (`--tranformed` flag)
		- Undoes the tranformations before calculating the vertex positions
		- Assumes track and azimuth indices were switched during transformation
		- Must specifiy the tmax to multiply track length by (to undo MaxAbs)
		- Energy will be miltipled by emax (to undo MaxAbs)
	
- `check_inputoutput.py` - plots input and output features of training files
	- Assumes train, test, validate arrays in file
	- Tries to import reco array, if it exists
	- Plots histograms of all 5 input variables (TAKES A LONG TIME TO FLATTEN) and 8 output variables
	- Also prints out neutrino/anti fraction, Track fraction, CC fraction

- `check_containment.py` - plots the starting anding ending positions after cuts
	- Used to check that containment cut worked
	- Cuts can be applied:
		- Energy maximum, parser arg
		- Energy minimum, parser arg
		- Type cut (CC, NC, etc.)
	- Able to use for transformed files (`--tranformed` flag)
		- Undoes the tranformations before calculating the vertex positions
		- Assumes track and azimuth indices were switched during transformation
		- Must specifiy the tmax to multiply track length by (to undo MaxAbs)
		- Energy will be miltipled by emax (to undo MaxAbs)

### Other Tools

- `PlottingFunctions.py` - long script containing most of the plotting functions and stastics used for testing
	- `get_RMS`
	- `get_FWHM`
	- `find_countours_2D`
	- `plot_history`
	- `plot_history_from_list`
	- `plot_history_from_list_split`
	- `plot_distributions_CCNC`
	- `plot_distributions`
	- `plot_2D_prediction`
	- `plot_2D_prediction_fraction`
	- `plot_resolution_CCNC`
	- `plot_single_resolution`
	- `plot_compare_resolution`
	- `plot_systematic_slices`
	- `plot_bin_slices`

- `handle_data.py` - Functions to handle data processing
	- `Shuffler`
	- `CutMask`
	- `VertexCut`
	- `SplitTrainTest`

- `scaler_transformations.py` - Functions that apply specific transformations to prerpare data for quicker training

- `apply_containment.py` - Cuts out events that don't end in the containment region, should be applied to a file already gone through final level processing
	- Will make a "not flat" dataset
	- Will reduce the number of events in dataset
	- Can be applied to any final level dataset

- `get_statistics.py` - Functions that find statistics (max, min, quartiles) to apply transformations on training data
	- No longer used, typically use static near-maximum values now
######################################
# README for Neural Network Project
#       Jessie Micallef
#       Updated: 13 May 2019
######################################

### ORDER TO RUN CONVOLUTIONAL NEURAL NETWORK ###
CreateTrainingFiles.ipynb
ConcatonateFiles.ipynb (optional)
TransformTrainingFiles.ipynb
CNN_CreateModel.ipynb

Once you have files that work, 

## CreateTrainingFiles.ipynb ##
Creates training files using pulse series per DOM
    - Outputs hdf5 file with three 4D arrays (data for DC, IC near DC, number pulses per DOM) and three 2D array (labels, reco_labels, i\
nitial_stats)
    - Meant to take in N number of i3 files and output one hdf5 with 4D array information
    - 5 variables per DOM per event
    - Needs IceCube library to run!

    Outputs specifically:
    - features contains: [event number, string index, dom index, [sum charge, time of first hit, time of last hit, charge weighted average of time, charge weighted std of time] ]
    - labels contains: [event number, [ nu energy, nu zenith, nu azimuth, nu time, nu x, nu y, nu z, track length (0 for cascade), isTrack, flavor, type (anti = 1), isCC]]
    - reco_labels contains: [event number, [ nu energy, nu zenith, nu azimuth, nu time, nu x, nu y, nu z]]
    - initial_states contains: [event number, [ count outside used strings, charge outside, count inside, charge inside] ]
    - number_pulses_per_dom contains: [event number, [string index, dom index, # hits] ]

## ConcatonateFiles.ipynb ##
    - Concatonates all given files into one, applies cuts if specified
    - Specify cuts to apply (see file, example = cascade, track, track NC, etc.)
    - Outputs the exact same arrays as CreateTrainingFiles, just adjusts the event numbers
    - Will add the name '.lt60_vertexDC.hdf5' to end of file along with the cuts specified


## TransformTrainingFiles.ipynb ##
   - Outputs same as CreateTrainingFiles but with everything split into training and testing sets, and with the features (X_train and X_test) transformed
   - Uses Robust Scaler transformation to shift mean to 0 and normalize the range
   - Can handle the reco labels, and num pulses and initials stats (just ignores these)
   - Outputs specicially:
       - Y_train = labels for training events
       - Y_test = labels for testing events
       - X_train_DC = features for DC strings only, for training events
       - X_test_DC  = features for DC strings only, for testing events
       - X_train_IC = features for IC near DC strings only, for training events
       - X_test_IC = features for IC near DC strings only, for testing events
       - reco_train = reco labels for training events (optional)
       - reco_test = reco labels for testing events (optional)


## CNN_CreateModel.ipynb ##
    - Takes in transformed data file to use for training
    - Neural Network model configuration created here!!
    - Saves trained network to file to use later
    - Plots:
        - Network history (loss vs epoch)
        - Energy distribution for truth and for network reco
        - Resolution plot for NN reco - Truth
        - 2D prediction plot (True vs Reco Energy)
        
## PlottingFunctions.py ##
    - Contains all the plotting functions, to call `from PlottingFunctions import [name of function]`
    - Contains the functions: 
        - get_RMS - used to calculate RMS for plotting statistics
        - get_FWHM - FWHM calculation method, not currently plugged in
        - plot_history - scatter line plot of loss vs epochs
        - plot_distributions_CCNC - plot energy distribution for truth, and for NN reco
        - plot_resolutions_CCNC - plot energy resoltuion for (NN reco - truth)
        - plot_2D_prediction - 2D plot of True vs Reco
        - plot_single_resolution - Resolution histogram, (NN reco - true) and can compare (old reco - true)
        - plot_compare_resolution - Histograms of resolutions for systematic sets, overlaid
        - plot_systematic_slices - "Scatter plot" with systematic sets on x axis and 68% resolution on y axis
        - plot_energy_slices - Scatter plot energy cut vs resolution

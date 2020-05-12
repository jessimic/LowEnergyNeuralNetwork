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

## Order to run code
1. Create hdf5 files from i3 files (`create_training_file_perDOM_nocut.py`)
2. Flatten energy distribution (optional, for training) (`flatten_energy_distribution.py`)
3. Cut, concatonate, and transform data for easier training (`cut_concat_transform_files.py`)
4. Train the CNN (`CNN_LoadMultipleFiles.py`)
5. Test the CNN (`CNN_TestOnly.py`)

## Description of major code components
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
		- Can do any number of IC strings (put numbers in IC_near_DC_strings list
		- Cleaned (SRTTWOfflinePulsesDC) or uncleaned (SplitInIcePulses) pulses, parse arg
		- Output reco labels array from pegleg or retro reco, parse arg
		- Emax cut (cut any event above specified energy max)
	- Outputs the input features [evt # x 8 x 60 x 5] for DC and [19 x 60 x 5] for IC and the output features [evt # x 12] for truth labels
		- Output labels: [nu energy, nu zenith, nu azimuth, nu time, nu x, nu y, nu z, track length (0 for cascade), isTrack, flavor, type (anti = 1), isCC]
		- Input feature summary variables: [sum charges, time first pulse, Time of last pulse, Charge weighted mean time of pulses, Charge weighted standard deviation of pulse times]
Can also put out a [evt # x 7] labels for old reco
		- Reco labels: [nu energy, nu zenith, nu azimuth, nu time, nu x, nu y, nu z, track length]

- `create_training_single_file_perDOM.py` - data from i3 to output hdf5 WITH MORE CUTS
	- Everything above is true
	- Additional cuts:
		- Always has a starting vertex cut, can be DC, IC, etc.
		- Optional ending position cut, parse arg
		- Fraction drop of cascade or track (automatically off)
	- Downside: may need to rerun if the vertex or ending position isn't set, can make these cuts later if you use `create_training_file_perDOM_nocut.py`

	
- `check_energybins.py` - quickly read in many files and make distribution of energy
	- Used to determine how to make flat sample
	- Prints the number of events in the smallest populated bin
	- Use that information for `flatten_energy_distribution.py`
	- Cuts can be applied to get accurate number of events in sample:
		- Vertex cut start, parse arg
		- Ending position cut, parse arg
		- Energy maximum, parse arg
		- Energy minimum, parse arg
	- Able to use for transformed files (`--tranformed` flag)
		- Undoes the tranformations before calculating the vertex positions
		- Assumes track and azimuth indices were switched during transformation
		- Must specifiy the tmax to multiply track length by (to undo MaxAbs)
		- Energy will be miltipled by emax (to undo MaxAbs)

- `flatten_energy_distribution.py` - reads in multiple files and saves events only if that current energy bin has not reached its maximum
	- Throws away any event that is "superfluous" after maximum reached per bin
	- User specifies maximum per bin (use `check_energybins.py` to determine) and GeV size of bins
	- Cuts can be applied to get accurate number of events in sample:
		- Vertex cut start, parse arg
		- Ending position cut, parse arg
		- Energy maximum, parse arg
		- Energy minimum, parse arg
	- Can output more than 1 file, to split events between (so files are not so large)
	- MUST shuffle if using NUM_OUT > 1
	- Outputs features_DC, features_IC, labels, num_pulses_per_dom, trigger_time, and optional reco_labels to use for statitics/comparisons later

	
- `check_inputoutput.py` - plots input and output features of training files
	- Assumes train, test, validate arrays in file
	- Tries to import reco array, if it exists
	- Plots histograms of all 5 input variables (TAKES A LONG TIME TO FLATTEN) and 8 output variables
	- Also prints out neutrino/anti fraction, Track fraction, CC fraction

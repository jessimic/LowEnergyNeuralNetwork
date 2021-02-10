'''
oscNext global variables

Tom Stuttard
'''

# Define data types
#TODO Avoid duplication with `run_database.py`
SIM_DATA_TYPES = ["genie","muongun","corsika","noise","nugen"]
DATA_TYPES = ["data"] + SIM_DATA_TYPES

# Define sub-event stream
SUB_EVENT_STREAM = "InIceSplit"

# Define pulse series
UNCLEANED_PULSES = "SplitInIcePulses" #TODO Use icetop,filterscripts.filter_globals.UncleanedInIcePulses instead?
CLEANED_PULSES = "SRTTWOfflinePulsesDC"

# Define filter
DC_FILTER_YEAR = 13

# Location of classifier models
CLASSIFIER_MODEL_DIR = "$I3_SRC/oscNext/resources/models"

# Cut bools
L3_CUT_BOOL_KEY = "L3_oscNext_bool" 
L4_CUT_BOOL_KEY = "L4_oscNext_bool" 
L5_CUT_BOOL_KEY = "L5_oscNext_bool" 
L6_CUT_BOOL_KEY = "L6_oscNext_bool" 
L7_CUT_BOOL_KEY = "L7_oscNext_bool" 

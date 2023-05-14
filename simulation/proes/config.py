import os.path as path

#############################################
#                GLOBAL VARIABLES           #
#############################################
ENABLE_LOGGING = False          # set to True for further logging in .log files

VERBOSE = False                 # set to True for verbose print statements

PROTON_PHYSICS = True           # whether or not we are interested in proton acceleration in the burst


#############################################
#                    PATHS                  #
#############################################
BASE_PATH = path.dirname(path.abspath(__file__))
DATA_PATH = path.join(path.split(BASE_PATH)[0], 'da')
IM_PATH = path.join(DATA_PATH, 'im')
LOG_PATH = path.join(DATA_PATH, 'log')
CSV_PATH = path.join(DATA_PATH, 'csv')

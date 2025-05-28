# use from ra_utils.utils.verbosity_enums import *

from enum import Flag, auto
class VerboseLevel(Flag):
    QUIET = auto()
    PRINT_PARAMS = auto()
    CHATTY = auto()

QUIET = VerboseLevel.QUIET
PRINT_PARAMS = VerboseLevel.PRINT_PARAMS   
CHATTY = VerboseLevel.CHATTY 

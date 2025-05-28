# use from ra_utils.utils.verbosity_enums import *



from enum import Flag, auto
from functools import total_ordering
@total_ordering
class VerboseLevel(Flag):
    MAFIA = auto()
    QUIET = auto()
    PRINT_PARAMS = auto()
    CHATTY = auto()
    VERYCHATTY = auto()
    SUPERCHATTY = auto()
    def __lt__(self, other):
        if self.__class__ is other.__class__:
          return self.value < other.value
        return NotImplemented

MAFIA = VerboseLevel.MAFIA
QUIET = VerboseLevel.QUIET
PRINT_PARAMS = VerboseLevel.PRINT_PARAMS   
CHATTY = VerboseLevel.CHATTY 
VERYCHATTY = VerboseLevel.VERYCHATTY
SUPERCHATTY = VerboseLevel.SUPERCHATTY


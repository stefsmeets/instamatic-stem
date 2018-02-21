from instamatic.gui.io_frame import *
from .stem_frame import *
from .beam_test_frame import *

from collections import namedtuple

Module = namedtuple('Module', ['name', 'display_name', 'tabbed', 'tk_frame'])

MODULES = (
Module("io", "i/o", False, IOFrame),
Module("scanning", "scanning", True, STEMFrame),
Module("beam", "beam", True, BeamTestFrame) )
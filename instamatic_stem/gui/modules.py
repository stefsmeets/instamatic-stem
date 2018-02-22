from instamatic.gui.io_frame import *
from . import stem_frame
from . import beam_test_frame

from .base_module import BaseModule

MODULES = (
BaseModule("io", "i/o", False, IOFrame, {}),
stem_frame.module,
beam_test_frame.module )


import os, sys
from instamatic.formats import *

import time
import logging

import datetime

from .modules import MODULES


def main():
    from instamatic.utils import high_precision_timers
    high_precision_timers.enable()  # sleep timers with 1 ms resolution
    
    # enable faster switching between threads
    sys.setswitchinterval(0.001)  # seconds

    import psutil
    p = psutil.Process(os.getpid())
    p.nice(psutil.REALTIME_PRIORITY_CLASS)  # set python process as high priority

    from instamatic import config

    date = datetime.datetime.now().strftime("%Y-%m-%d")
    logfile = config.logs_drc / f"instamatic_{date}.log"

    logging.basicConfig(format="%(asctime)s | %(module)s:%(lineno)s | %(levelname)s | %(message)s", 
                        filename=logfile, 
                        level=logging.DEBUG)

    logging.captureWarnings(True)
    log = logging.getLogger(__name__)
    log.info("Instamatic-stem.gui started")

    # Work-around for race condition (errors) that occurs when 
    # DataCollectionController tries to access them

    from ..settings import default as settings
    from ..beam_control import BeamCtrl
    beam_ctrl = BeamCtrl(**settings)

    from instamatic.gui.gui import DataCollectionController, DataCollectionGUI

    stream = DataCollectionGUI(cam=config.cfg.camera, modules=MODULES)

    while not stream._modules_have_loaded:
        time.sleep(0.1)

    gui = DataCollectionController(tem_ctrl=None, stream=stream, beam_ctrl=beam_ctrl, log=log)


if __name__ == '__main__':
    main()
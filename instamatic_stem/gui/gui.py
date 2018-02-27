import os, sys
from instamatic.formats import *

import time
import logging

import datetime

from .modules import MODULES


def main():
    import psutil, os
    p = psutil.Process(os.getpid())
    p.nice(psutil.REALTIME_PRIORITY_CLASS)  # set python process as high priority

    from os.path import dirname as up
    
    logging_dir = up(up(up(__file__)))

    date = datetime.datetime.now().strftime("%Y-%m-%d")
    logfile = os.path.join(logging_dir, "logs", "scanning_{}.log".format(date))

    logging.basicConfig(format="%(asctime)s | %(module)s:%(lineno)s | %(levelname)s | %(message)s", 
                        filename=logfile, 
                        level=logging.DEBUG)

    logging.captureWarnings(True)
    log = logging.getLogger(__name__)
    log.info("Scanning.gui started")

    # Work-around for race condition (errors) that occurs when 
    # DataCollectionController tries to access them

    from ..settings import default as settings
    from ..beam_control import BeamCtrl
    beam_ctrl = BeamCtrl(**settings)

    from instamatic.gui.gui import DataCollectionController, DataCollectionGUI

    from instamatic import config
    stream = DataCollectionGUI(cam=config.cfg.camera, modules=MODULES)

    while not stream._modules_have_loaded:
        time.sleep(0.1)

    gui = DataCollectionController(tem_ctrl=None, stream=stream, beam_ctrl=beam_ctrl, log=log)


if __name__ == '__main__':
    main()
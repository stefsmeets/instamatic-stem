from tkinter import *
from tkinter.ttk import *

import os, sys
import traceback
from instamatic.formats import *

import time
import logging

import threading
import queue

import datetime
from .experiment import get_coords

from instamatic.camera.videostream import VideoStream

from collections import namedtuple
from .interface import BeamCtrlFrame
Module = namedtuple('Module', ['name', 'display_name', 'tabbed', 'tk_frame'])

MODULES = Module("scanning", "scanning", False, BeamCtrlFrame),


class DataCollectionController(object):
    """docstring for DataCollectionController"""
    def __init__(self, stream, beam_ctrl, log=None):
        super(DataCollectionController, self).__init__()
        self.stream = stream
        self.camera = stream.cam.name
        self.beam_ctrl = beam_ctrl
        self.log = log

        self.q = queue.LifoQueue(maxsize=1)
        self.triggerEvent = threading.Event()
        
        self.module_scanning = self.stream.get_module("scanning")

        self.module_scanning.set_trigger(trigger=self.triggerEvent, q=self.q)
        self.module_scanning.beam_ctrl = beam_ctrl

        self.exitEvent = threading.Event()
        self.stream._atexit_funcs.append(self.exitEvent.set)
        self.stream._atexit_funcs.append(self.triggerEvent.set)

        self.wait_for_event()

    def wait_for_event(self):
        while True:
            self.triggerEvent.wait()
            self.triggerEvent.clear()

            if self.exitEvent.is_set():
                self.stream.close()
                sys.exit()

            job, kwargs = self.q.get()
            try:
                if job == "scanning":
                    self.acquire_data_scanning(**kwargs)
                elif job == "plot_scan_grid":
                    self.plot_scan_grid(**kwargs)
                else:
                    print("Unknown job: {}".format(job))
                    print("Kwargs:\n{}".format(kwargs))
            except Exception as e:
                traceback.print_exc()
                self.log.debug("Error caught -> {} while running '{}' with {}".format(repr(e), job, kwargs))
                self.log.exception(e)

    def acquire_data_scanning(self, **kwargs):
        from .experiment import do_experiment
        do_experiment(self.stream, **kwargs)
        
        # from experiment import do_experiment_continuous
        # do_experiment_continuous(self.stream, **kwargs)

    def plot_scan_grid(self, **kwargs):
        import matplotlib.pyplot as plt

        coords = get_coords(**kwargs)

        plt.scatter(*coords.T)
        plt.title("Coordinates for grid scan")
        plt.xlabel("X axis")
        plt.ylabel("Y axis")
        plt.axis('equal')
        plt.show()

 
class DataCollectionGUI(VideoStream):
    """docstring for DataCollectionGUI"""
    def __init__(self, *args, **kwargs):
        super(DataCollectionGUI, self).__init__(*args, **kwargs)
        self.modules = {}
        self._modules_have_loaded = False

    def buttonbox(self, master):
        frame = Frame(master)
        frame.pack(side="right", fill="both", expand="yes")

        # self.nb = Notebook(frame, padding=10)

        for module in MODULES:
            if module.tabbed:
                page = Frame(self.nb)
                module_frame = module.tk_frame(page)
                module_frame.pack(side="top", fill="both", expand="yes", padx=10, pady=10)
                self.modules[module.name] = module_frame
                self.nb.add(page, text=module.display_name)
            else:
                module_frame = module.tk_frame(frame)
                module_frame.pack(side="top", fill="both", expand="yes", padx=10, pady=10)
                self.modules[module.name] = module_frame

        # self.nb.pack(fill="both", expand="yes")

        btn = Button(master, text="Save image",
            command=self.saveImage)
        btn.pack(side="bottom", fill="both", padx=10, pady=10)

        self._modules_have_loaded = True

    def get_module(self, module):
        return self.modules[module]

    def saveImage(self):
        # module_io = self.get_module("io")

        # drc = module_io.get_experiment_directory()
        # if not os.path.exists(drc):
            # os.makedirs(drc)
        outfile = datetime.datetime.now().strftime("%Y%m%d-%H%M%S.%f") + ".tiff"
        # outfile = os.path.join(drc, outfile)

        try:
            from instamatic.processing.flatfield import apply_flatfield_correction
            flatfield, h = read_tiff(module_io.get_flatfield())
            frame = apply_flatfield_correction(self.frame, flatfield)
        except:
            frame = self.frame
        write_tiff(outfile, frame)
        print(" >> Wrote file:", outfile)


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

    from .settings import default as settings
    from .beam_control import BeamCtrl
    beam_ctrl = BeamCtrl(**settings)

    from instamatic import config
    stream = DataCollectionGUI(cam=config.cfg.camera)

    while not stream._modules_have_loaded:
        time.sleep(0.1)

    gui = DataCollectionController(stream, beam_ctrl=beam_ctrl, log=log)


if __name__ == '__main__':
    main()
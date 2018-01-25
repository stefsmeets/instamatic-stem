from Tkinter import *
from ttk import *
import threading
#import pyaudio
#from pyaudio import PyAudio

import sounddevice as sd

import time
from math import ceil
import numpy as np

#import matplotlib.pyplot as plt

devices = {
    'ch12': 'ADAT 1 (1+2) (Fireface ADAT 1 (1+2))',
    'ch34': 'ADAT 1 (3+4) (Fireface ADAT 1 (3+4))',
    'ch56': 'ADAT 1 (5+6) (Fireface ADAT 1 (5+6))',
    'ch78': 'ADAT 1 (7+8) (Fireface ADAT 1 (7+8))',
    'all':  'ASIO Fireface USB'
}


class BeamCtrl(object):
    """docstring for BeamCtrl"""
    def __init__(self, 
        device=None,          # sounddevice to use, set to `None` to use default
        fs=44100,             # sampling rate, Hz, must be integer
        duration=1.0,         # in seconds, may be float
        global_volume=1.0,    # range [0.0, 1.0]
        n_channels=2,         # number of channels
        mapping=(1,2),        # mapping of the channels
        chunksize=1024,       # number of sampling points per channel
        **kwargs
        ): 
        super(BeamCtrl, self).__init__()
        
        sd.default.device  = device

        self.fs            = fs           
        self.duration      = duration     
        self.global_volume = global_volume
        self.n_channels    = n_channels   
        self.mapping       = mapping
        self.chunksize     = chunksize    
 
        length             = ceil(fs*duration*n_channels)
        self.signal_data   = np.empty(int(length), dtype=np.float32).reshape(-1, n_channels)
        self.x             = np.arange(fs*duration) / float(fs)  # x-axis

    def get_signal(self, channel_data):
        n_channels = self.n_channels
        global_volume = self.global_volume

        signal_data = self.signal_data
        x = self.x
        
        # multichannel should be interweaved like this: L1R1L2R2L3R3L4R4 (8 channels)
        # https://zach.se/generate-audio-with-python/
        
        for channel in range(n_channels):
            # generate sine
            #freq = channel_data[channel]
            #signal_data.reshape(-1)[channel::n_channels] = (np.sin(2*np.pi*freq*x)).astype(np.float32) * global_volume

            # flat signal
            volume = channel_data[channel]
            signal_data.reshape(-1)[channel::n_channels] = volume * global_volume

        return signal_data

    def update(self, channel_data):
        self.signal = self.get_signal(channel_data)

        # print "signal updated"

    def start_stream(self):
        print "start stream"
        sd.play(self.signal_data, self.fs, mapping=self.mapping, loop=True)

    def stop_stream(self):
        print "stop stream"
        sd.stop()


class BeamCtrlFrame(object, LabelFrame):
    """docstring for BeamCtrlFrame"""
    def __init__(self, parent, settings):
        LabelFrame.__init__(self, parent, text="Fine beam control")
        self.parent = parent

        self.channels = settings["channels"]
        self.init_vars()
        self.beam_ctrl = BeamCtrl(**settings)

        frame = Frame(self)

        self.c_toggle = Checkbutton(frame, text="Enable", variable=self.var_toggle_beam, command=self.toggle_beam)
        self.c_toggle.grid(row=5, column=0, sticky="W")

        self.b_reset = Button(frame, text="reset", command=self.reset_channels)
        self.b_reset.grid(row=5, column=1, sticky="W")

        for i, channel in enumerate(self.channels):
            row = 10+i
            Label(frame, text=channel["name"], width=20).grid(row=row, column=0, sticky="W")
            e = Entry(frame, width=10, textvariable=channel["var"])
            e.grid(row=row, column=1, sticky="W", padx=5)
    
            slider = Scale(frame, variable=channel["var"], from_=-100, to=100, orient=HORIZONTAL, command=self.update_channels)
            slider.grid(row=row, column=2, columnspan=2, sticky="EW")

        frame.pack(side="top", fill="x", padx=10, pady=10)
        frame.columnconfigure(2, weight=1)

    def init_vars(self):
        for channel in self.channels:
            channel["var"] = channel["var"](channel["default"])

        self.var_toggle_beam = BooleanVar(value=False)

    def reset_channels(self):
        for channel in self.channels:
            channel["var"].set(channel["default"])
        self.update_channels()

    def toggle_beam(self):
        toggle = self.var_toggle_beam.get()
        if toggle:
            self.update_channels()
            self.beam_ctrl.start_stream()
        else:
            self.beam_ctrl.stop_stream()

    def set_trigger(self, trigger=None, q=None):
        self.triggerEvent = trigger
        self.q = q

    def update_channels(self, event=None):
        channel_data = []
        for channel in self.channels:
            var = channel["var"]
            val = var.get()
            var.set(val)
            channel_data.append(val / 100.0)
        # print "\nupdated", channel_data
        self.beam_ctrl.update(channel_data)


settings_fireface = {
'device': 'ASIO Fireface USB',
'global_volume': 1.0,
'fs': 44100,
'duration': 0.01,
'n_channels': 8,
'chunksize': 1024,
'mapping': (15,16,17,18,19,20,21,22),
'channels': (
    {"name": "BeamShift X?", "var": IntVar, "default": 0},
    {"name": "BeamShift Y?", "var": IntVar, "default": 0},
    {"name": "BeamTilt X?", "var": IntVar, "default": 0},
    {"name": "BeamTilt Y?", "var": IntVar, "default": 0},
    {"name": "ImageShift? X", "var": IntVar, "default": 0},
    {"name": "ImageShift? Y", "var": IntVar, "default": 0},
    {"name": "ImageTilt? X?", "var": IntVar, "default": 0},
    {"name": "ImageTilt? Y?", "var": IntVar, "default": 0}
)}

settings_testing = {
'device': None,
'global_volume': 1.0,
'fs': 44100,
'duration': 1.0,
'n_channels': 2,
'chunksize': 1024,
'mapping': (1,2),
'channels': (
    {"name": "Channel 1", "var": IntVar, "default": 0},
    {"name": "Channel 2", "var": IntVar, "default": 0}
)}


if __name__ == '__main__':
    root = Tk()
    BeamCtrlFrame(root, settings=settings_fireface).pack(side="top", fill="both", expand=True)
    #BeamCtrlFrame(root, settings=settings_testing).pack(side="top", fill="both", expand=True)
    root.mainloop()


import sounddevice as sd
import numpy as np
from math import ceil
import time


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

        self.channel_data   = None

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
        self.channel_data = channel_data
        self.signal = self.get_signal(channel_data)

        # print "signal updated", time.clock()

    def start_stream(self):
        print "start stream"
        sd.play(self.signal_data, self.fs, mapping=self.mapping, loop=True)

    def stop_stream(self):
        print "stop stream"
        sd.stop()

    def do_scan(self, o, x, y, duration=1.0, slices=20):
        fs = self.fs
        mapping = self.mapping
        ramps = []
        
        for i in range(slices):
            start = o + (x-o)*i/float(slices)
            end = start + (y-o)
            arr = ramp(start, end, fs, duration)
            ramps.append(arr)

        ramps = np.vstack(ramps)

        print ramps.shape
        sd.play(ramps, fs, mapping=mapping, blocking=False)
        print "Scanning started!"


def ramp(start, end, fs, duration):
    """
    start: (1, n_channels) vector
    end:   (1, n_channels) vector
    fs: frequency
    duration: time in s
    """

    n_channels = start.size
    n_samples = ceil(fs*duration)
    length = ceil(n_samples*n_channels)
    signal = np.empty(int(length), dtype=np.float32).reshape(-1, n_channels)

    for channel in range(n_channels):
        signal.reshape(-1)[channel::n_channels] = np.linspace(start[channel], end[channel], n_samples)

    return signal


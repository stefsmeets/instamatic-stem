from Tkinter import *
from ttk import *
import threading

from beam_control import BeamCtrl
import time
import numpy as np

from IPython import embed


class BeamCtrlFrame(object, LabelFrame):
    """docstring for BeamCtrlFrame"""
    def __init__(self, parent, settings, beam_ctrl):
        LabelFrame.__init__(self, parent, text="Fine beam control")
        self.parent = parent

        self.beam_ctrl = beam_ctrl
        self.channels = settings["channels"]
        self.init_vars()

        frame = Frame(self)

        self.c_toggle = Checkbutton(frame, text="Enable", variable=self.var_toggle_beam, command=self.toggle_beam)
        self.c_toggle.grid(row=5, column=0, sticky="W")

        self.b_reset = Button(frame, text="reset", command=self.reset_channels)
        self.b_reset.grid(row=5, column=1, sticky="W")

        for i, channel in enumerate(self.channels):
            self.make_slider(frame, channel["var"], channel["name"], 10+i, self.update_channels, -100, 100)

        Separator(frame, orient=HORIZONTAL).grid(row=100, columnspan=3, sticky="ew", pady=10)

        self.make_slider(frame, self.var_damping, "Damping factor", 101, self.update_channels, 0, 100)

        frame.pack(side="top", fill="x", padx=10, pady=10)
        frame.columnconfigure(2, weight=1)

    def make_slider(self, frame, var, label, row, command, minval, maxval):
         Label(frame, text=label, width=20).grid(row=row, column=0, sticky="W")
         e = Spinbox(frame, width=10, textvariable=var, from_=minval, to=maxval)
         e.grid(row=row, column=1, sticky="W", padx=5)
    
         slider = Scale(frame, variable=var, from_=minval, to=maxval, orient=HORIZONTAL, command=command)
         slider.grid(row=row, column=2, columnspan=2, sticky="EW")

    def init_vars(self):
        for channel in self.channels:
            channel["var"] = IntVar(value=channel["default"])
            channel["var"].trace("w", self.update_channels)

        self.var_damping = IntVar(value=100)
        self.var_damping.trace("w", self.update_channels)

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

    def update_channels(self, *args):
        # print "\nupdated", time.clock()
        try:
            channel_data = self.get_channel_data()
        except ValueError as e:
            return
        self.beam_ctrl.update(channel_data)

    def get_channel_data(self):
        channel_data = []
        damping = self.var_damping.get() / 100.0

        for channel in self.channels:
            val = channel["var"].get()
            channel_data.append(damping*val / 100.0)

        return channel_data


settings_fireface = {
'device': 'ASIO Fireface USB',
'global_volume': 1.0,
'fs': 44100,
'duration': 0.01,
'n_channels': 8,
'chunksize': 1024,
'mapping': (15,16,17,18,19,20,21,22),
'channels': (
    {"name": "BeamShift X", "var": None, "default": 0},
    {"name": "BeamShift Y", "var": None, "default": 0},
    {"name": "BeamTilt X?", "var": None, "default": 0},
    {"name": "BeamTilt Y?", "var": None, "default": 0},
    {"name": "ImageShift? X", "var": None, "default": 0},
    {"name": "ImageShift? Y", "var": None, "default": 0},
    {"name": "ImageTilt? X?", "var": None, "default": 0},
    {"name": "ImageTilt? Y?", "var": None, "default": 0}
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
    {"name": "Channel 1", "var": None, "default": 0},
    {"name": "Channel 2", "var": None, "default": 0}
)}


class App(threading.Thread):
    """docstring for App"""
    def __init__(self, settings):
        super(App, self).__init__()
        #BeamCtrlFrame(root, settings=settings_testing).pack(side="top", fill="both", expand=True)
        self.settings = settings
        self.beam_ctrl = BeamCtrl(**settings)

        self.start()

    def run(self):
        self.root = Tk()
        BeamCtrlFrame(self.root, settings=self.settings, beam_ctrl=self.beam_ctrl).pack(side="top", fill="both", expand=True)
        self.root.mainloop()

    def close(self):
        self.root.quit()

if __name__ == '__main__':
    app = App(settings=settings_testing)

    embed(banner1="")

    beam = app.beam_ctrl

    # origin (0, 0)
    raw_input("\n\nPress enter at top left -> origin (0, 0)")
    o = np.array(beam.channel_data)
    
    # topright (x, 0)
    raw_input("\n\nPress enter at top right -> topright (x, 0)")
    x = np.array(beam.channel_data) - o

    # botleft (0, y)
    raw_input("\n\nPress enter at bottom left -> botleft (0, y)")
    y = np.array(beam.channel_data) - o

    print
    print
    print o
    print x
    print y

    print
    for i in range(10):
        for j in range(10):
            coord = o + i*x + j*y
            print "{}: {} -> {}".format(i,j, coord)

            beam.update(coord)
            time.sleep(0.1)

    app.close()


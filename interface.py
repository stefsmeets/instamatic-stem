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

        Separator(frame, orient=HORIZONTAL).grid(row=100, columnspan=4, sticky="ew", pady=10)

        self.make_slider(frame, self.var_damping, "Damping factor", 101, self.update_channels, 0, 100)

        Separator(frame, orient=HORIZONTAL).grid(row=199, columnspan=4, sticky="ew", pady=10)

        frame.pack(side="top", fill="x", padx=10, pady=10)
        frame.columnconfigure(2, weight=1)

        frame = Frame(self)

        # duration
        Label(frame, text="Duration").grid(row=200, column=0, sticky="EW")
        e = Spinbox(frame, width=10, textvariable=self.var_scan_duration, from_=0.1, to=10.0, increment=0.1)
        e.grid(row=200, column=1, sticky="EW", padx=5)
    
        # slices
        Label(frame, text="Slices").grid(row=200, column=2, sticky="EW")
        e = Spinbox(frame, width=10, textvariable=self.var_scan_slices, from_=1, to=100)
        e.grid(row=200, column=3, sticky="EW", padx=5)

        self.bo = Button(frame, text="Set origin (0, 0)", command=self.set_scan_origin)
        self.bo.grid(row=201, column=0, sticky="EW")

        self.bx = Button(frame, text="Set X-axis (0, x)", command=self.set_scan_x_axis)
        self.bx.grid(row=201, column=1, sticky="EW")

        self.by = Button(frame, text="Set Y-axis (y, 0)", command=self.set_scan_y_axis)
        self.by.grid(row=201, column=2, sticky="EW")

        self.bgo = Button(frame, text="Scan!", command=self.do_scan)
        self.bgo.grid(row=201, column=3, sticky="EW")

        frame.pack(side="top", fill="x", padx=10, pady=10)


    def make_slider(self, frame, var, label, row, command, minval, maxval):
         Label(frame, text=label, width=20).grid(row=row, column=0, sticky="W")
         e = Spinbox(frame, width=10, textvariable=var, from_=minval, to=maxval, increment=0.1)
         e.grid(row=row, column=1, sticky="W", padx=5)
    
         slider = Scale(frame, variable=var, from_=minval, to=maxval, orient=HORIZONTAL, command=command)
         slider.grid(row=row, column=2, columnspan=2, sticky="EW")

    def init_vars(self):
        for channel in self.channels:
            channel["var"] = DoubleVar(value=channel["default"])
            channel["var"].trace("w", self.update_channels)

        self.var_damping = DoubleVar(value=100)
        self.var_damping.trace("w", self.update_channels)

        self.var_scan_slices = IntVar(value=20)
        self.var_scan_duration = DoubleVar(value=1.0)

        self.var_toggle_beam = BooleanVar(value=False)

        self.scan_variables = {"origin": None, "x_axis": None, "y_axis": None}

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

    def set_scan_origin(self):
        self.scan_variables["origin"] = np.array(self.get_channel_data())
        for key, value in self.scan_variables.items():
            print key, value
        print

    def set_scan_x_axis(self):
        self.scan_variables["x_axis"] = np.array(self.get_channel_data())
        for key, value in self.scan_variables.items():
            print key, value
        print

    def set_scan_y_axis(self):
        self.scan_variables["y_axis"] = np.array(self.get_channel_data())
        for key, value in self.scan_variables.items():
            print key, value
        print

    def do_scan(self):
        assert self.scan_variables["origin"] is not None, "Missing origin!"
        assert self.scan_variables["x_axis"] is not None, "Missing x_axis!"
        assert self.scan_variables["y_axis"] is not None, "Missing y_axis!"
        duration = self.var_scan_duration.get()
        slices = self.var_scan_slices.get()

        self.var_toggle_beam.set(False)
        self.beam_ctrl.do_scan(self.scan_variables["origin"],
                               self.scan_variables["x_axis"],
                               self.scan_variables["y_axis"],
                               duration,
                               slices)


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
        self.settings = settings
        self.beam_ctrl = BeamCtrl(**settings)

        self.start()

    def run(self):
        self.root = Tk()
        BeamCtrlFrame(self.root, settings=self.settings, beam_ctrl=self.beam_ctrl).pack(side="top", fill="both", expand=True)
        self.root.mainloop()

    def close(self):
        self.root.quit()


def run_gui(settings, beam_ctrl):
    root = Tk()
    BeamCtrlFrame(root, settings=settings, beam_ctrl=beam_ctrl).pack(side="top", fill="both", expand=True)
    root.mainloop()


def do_scan_lab6(beam):
    damp = 0.5
    origin   = np.array([ 17,  35], dtype=np.float) * damp * 0.01
    botleft  = np.array([-40,  26], dtype=np.float) * damp * 0.01
    topright = np.array([ 31, -21], dtype=np.float) * damp * 0.01
    
    beam.do_scan(origin, botleft, topright, duration=1.0, slices=20)


if __name__ == '__main__':
    settings = settings_testing
    # settings = settings_fireface

    beam = BeamCtrl(**settings)



    run_gui(settings, beam)



    # do_scan_lab6(beam)



    # run_gui(settings, beam)
    


    # app = App(settings=settings)
    # beam = app.beam_ctrl

    # embed(banner1="")

    # app.close()


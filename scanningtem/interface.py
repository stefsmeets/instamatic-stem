from Tkinter import *
from ttk import *

from beam_control import BeamCtrl

from IPython import embed

import numpy as np

from settings import DEFAULT_SETTINGS
from experiment import get_coords


class BeamCtrlFrame(object, LabelFrame):
    """docstring for BeamCtrlFrame"""
    def __init__(self, parent, settings=DEFAULT_SETTINGS):
        LabelFrame.__init__(self, parent, text="Fine beam control")
        self.parent = parent

        self.beam_ctrl = None

        self.channels = settings["channels"]
        self.init_vars()

        frame = Frame(self)

        Checkbutton(frame, text="Enable", variable=self.var_toggle_beam, command=self.toggle_beam).grid(row=5, column=0, sticky="W")
        Button(frame, text="reset", command=self.reset_channels).grid(row=5, column=1, sticky="W")

        for i, channel in enumerate(self.channels):
            self.make_slider(frame, channel["var"], channel["name"], 10+i, -100, 100, self.update_channels)

        Separator(frame, orient=HORIZONTAL).grid(row=100, columnspan=4, sticky="ew", pady=10)

        self.make_slider(frame, self.var_damping, "Damping factor", 101, 0, 100, self.update_channels)

        frame.pack(side="top", fill="x", padx=10, pady=10)
        frame.columnconfigure(2, weight=1)

        frame = Frame(self)

        Separator(frame, orient=HORIZONTAL).grid(row=1, columnspan=4, sticky="ew", pady=10)

        self.make_entry(frame, self.var_dwell_time, "Dwell time (s)", 5, 0, 0.01, 10.0, 0.01)
        self.make_entry(frame, self.var_exposure, "Exposure (s)", 6, 0, 0.01, 10.0, 0.01)
    
        self.make_slider(frame, self.var_strength, "Strength", 20, 0.0, 100.0, self.update_test_stream)
        self.make_slider(frame, self.var_rotation, "Rotation", 21, -180, 180, self.update_test_stream)

        self.make_entry(frame, self.var_grid_x, "Grid x", 30, 0, 2, 100, 1)
        self.make_entry(frame, self.var_grid_y, "Grid y", 30, 2, 2, 100, 1)

        Checkbutton(frame, text="Show perimeter", variable=self.var_toggle_test, command=self.toggle_test_scanning).grid(row=40, column=0, sticky="EW")
        Button(frame, text="Plot coords", command=self.show_grid_plot).grid(row=40, column=2, sticky="EW")
        Button(frame, text="Scan!", command=self.start_scanning).grid(row=40, column=3, sticky="EW")

        frame.pack(side="top", fill="x", padx=10, pady=10)

    def make_slider(self, frame, var, label, row, minval, maxval, command=None):
        Label(frame, text=label, width=20).grid(row=row, column=0, sticky="W")
        e = Spinbox(frame, width=10, textvariable=var, from_=minval, to=maxval, increment=0.1)
        e.grid(row=row, column=1, sticky="W", padx=5)
    
        slider = Scale(frame, variable=var, from_=minval, to=maxval, orient=HORIZONTAL, command=command)
        slider.grid(row=row, column=2, columnspan=2, sticky="EW")

    def make_entry(self, frame, var, label, row, column, minval, maxval, increment):
        Label(frame, text=label).grid(row=row, column=column, sticky="EW")
        e = Spinbox(frame, width=10, textvariable=var, from_=minval, to=maxval, increment=increment)
        e.grid(row=row, column=column+1, sticky="EW", padx=5)

    def init_vars(self):
        for channel in self.channels:
            channel["var"] = DoubleVar(value=channel["default"])
            channel["var"].trace("w", self.update_channels)

        self.var_damping = DoubleVar(value=100)
        self.var_damping.trace("w", self.update_channels)

        self.var_toggle_beam = BooleanVar(value=False)

        self.var_dwell_time  = DoubleVar(value=0.05)
        self.var_exposure    = DoubleVar(value=0.01)
        self.var_grid_x      = IntVar(value=10)
        self.var_grid_x.trace("w", self.update_test_stream)
        self.var_grid_y      = IntVar(value=10)
        self.var_grid_y.trace("w", self.update_test_stream)
        self.var_strength    = DoubleVar(value=50.0)
        self.var_rotation    = DoubleVar(value=0.0)
        self.var_toggle_test = BooleanVar(value=False)

    def reset_channels(self):
        for channel in self.channels:
            channel["var"].set(channel["default"])
        self.update_channels()

    def toggle_beam(self):
        toggle = self.var_toggle_beam.get()
        if toggle:
            self.update_channels()
            self.beam_ctrl.play()
        else:
            self.beam_ctrl.stop()

    def set_trigger(self, trigger=None, q=None):
        self.triggerEvent = trigger
        self.q = q

    def update_channels(self, *args):
        # print "\nupdated", time.clock()
        try:
            channel_data = self.get_channel_data()
        except ValueError as e:
            return
        if self.beam_ctrl:
            self.beam_ctrl.update(channel_data)

    def get_channel_data(self):
        channel_data = []
        damping = self.var_damping.get() / 100.0

        for channel in self.channels:
            val = channel["var"].get()
            channel_data.append(damping*val / 100.0)

        return channel_data

    def get_scanning_params(self):
        params = { "dwell_time": self.var_dwell_time.get(),
                   "grid_x": self.var_grid_x.get(),
                   "grid_y": self.var_grid_y.get(),
                   "strength": self.var_strength.get() / 100,
                   "rotation": self.var_rotation.get(),
                   "exposure": self.var_exposure.get(),
                   "beam_ctrl": self.beam_ctrl }
        return params

    def toggle_test_scanning(self, *args):
        toggle = self.var_toggle_test.get()
        if toggle:
            self.update_test_stream()
            self.beam_ctrl.play()
        else:
            self.beam_ctrl.stop()

    def update_test_stream(self, *args):
        if self.var_toggle_test.get():
            params = self.get_scanning_params()
            
            coords = get_coords(**params)
            grid_x = params["grid_x"]
            grid_y = params["grid_y"]

            idx = 0, grid_x-1, -1, -grid_y
            corners = coords.take(idx, 0)

            self.beam_ctrl.do_box_scan(corners)

    def start_scanning(self, *args):
        params = self.get_scanning_params()
        self.q.put(("scanning", params))
        self.triggerEvent.set()

    def show_grid_plot(self, *args):
        params = self.get_scanning_params()
        self.q.put(("plot_scan_grid", params))
        self.triggerEvent.set()


def run_gui(settings, beam_ctrl):
    root = Tk()
    BeamCtrlFrame(root, settings=settings, beam_ctrl=beam_ctrl).pack(side="top", fill="both", expand=True)
    root.mainloop()


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


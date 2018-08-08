from tkinter import *
from tkinter.ttk import *
from instamatic.utils.spinbox import Spinbox

from ..beam_control import BeamCtrl

from ..settings import default as default_settings
from ..experiment import get_coords


global_damping_factor = 100

class BeamTestFrame(LabelFrame):
    """docstring for BeamCtrlFrame"""
    def __init__(self, parent, settings=default_settings):
        LabelFrame.__init__(self, parent, text="Beam control")
        self.parent = parent

        self.channels = settings["channels"]
        self.init_vars()

        frame = Frame(self)

        Checkbutton(frame, text="Enable", variable=self.var_toggle_stream, command=self.toggle_stream).grid(row=5, column=0, sticky="W")
        Button(frame, text="Reset", command=self.reset_channels).grid(row=5, column=1, sticky="W")

        for i, channel in enumerate(self.channels):
            self.make_slider(frame, channel["var"], channel["name"], 10+i, -100, 100, self.update_stream)

        self.make_slider(frame, self.var_damping, "Strength", 101, 0, 100, self.update_stream)

        frame.pack(side="top", fill="x", padx=10, pady=10)
        frame.columnconfigure(2, weight=1)

    def make_slider(self, frame, var, label, row, minval, maxval, command=None):
        Label(frame, text=label, width=20).grid(row=row, column=0, sticky="W")
        e = Spinbox(frame, width=10, textvariable=var, from_=minval, to=maxval, increment=0.1)
        e.grid(row=row, column=1, sticky="EW", padx=5)
    
        slider = Scale(frame, variable=var, from_=minval, to=maxval, command=command)
        slider.grid(row=row, column=2, columnspan=2, sticky="EW")

    def make_entry(self, frame, var, label, row, column, minval, maxval, increment):
        Label(frame, text=label).grid(row=row, column=column, sticky="EW")
        e = Spinbox(frame, width=10, textvariable=var, from_=minval, to=maxval, increment=increment)
        e.grid(row=row, column=column+1, sticky="EW", padx=5, pady=2.5)

    def init_vars(self):
        for channel in self.channels:
            channel["var"] = DoubleVar(value=channel["default"])
            channel["var"].trace_add("write", self.update_stream)

        self.var_damping = DoubleVar(value=50.0)
        self.var_damping.trace_add("write", self.update_stream)

        self.var_toggle_stream   = BooleanVar(value=False)

    def reset_channels(self):
        for channel in self.channels:
            channel["var"].set(channel["default"])
        self.update_stream()

    def toggle_stream(self):
        toggle = self.var_toggle_stream.get()
        if toggle:
            self.update_stream(state="start")
        else:
            self.update_stream(state="stop")

    def set_trigger(self, trigger=None, q=None):
        self.triggerEvent = trigger
        self.q = q

    def update_stream(self, *args, state="continue"):
        params = self.get_params()
        params["state"] = state
        self.q.put(("beam_control", params))
        self.triggerEvent.set()

    def get_params(self):
        channel_data = []
        damping = self.var_damping.get() / (100.0 * global_damping_factor)

        for channel in self.channels:
            val = channel["var"].get()
            channel_data.append(damping*val / 100.0)

        return {"channel_data": channel_data}


def beam_control(controller, **kwargs):
    state = kwargs.get("state")
    if state == "stop":
        controller.beam_ctrl.stop()
        return

    channel_data = kwargs.get("channel_data")
    controller.beam_ctrl.update(channel_data)

    if state == "start":
        controller.beam_ctrl.play()


from instamatic.gui.base_module import BaseModule

module = BaseModule("beam", "beam", True, BeamTestFrame, commands={
    "beam_control": beam_control
} )


if __name__ == '__main__':
    from ..settings import default_settings as settings

    root = Tk()
    BeamTestFrame(root, settings=settings).pack(side="top", fill="both", expand=True)
    root.mainloop()

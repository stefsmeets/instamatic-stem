from tkinter import *
from tkinter.ttk import *
from instamatic.utils.spinbox import Spinbox

from ..beam_control import BeamCtrl

from ..settings import default as default_settings
from ..experiment import get_coords


global_damping_factor = 100

class STEMFrame(LabelFrame):
    """docstring for BeamCtrlFrame"""
    def __init__(self, parent, settings=default_settings):
        LabelFrame.__init__(self, parent, text="Set up a raster scan")
        self.parent = parent

        self.beam_ctrl = None

        self.channels = settings["channels"]
        self.init_vars()

        frame = Frame(self)

        self.make_entry(frame, self.var_dwell_time, "Dwell time (s)", 5, 0, 0.01, 10.0, 0.01)
        Label(frame, textvariable=self.var_actual_dwell_time).grid(row=5, column=2, columnspan=2, sticky="EW")
        
        blocksizes = [0, 16, 32, 64, 128, 256, 512, 1024, 2048]
        Label(frame, text="Blocksize").grid(row=6, column=0, sticky="EW")
        Combobox(frame, width=10, textvariable=self.var_blocksize, values=blocksizes).grid(row=6, column=1, sticky="EW", padx=5, pady=2.5)
        
        # latencies = ["low", "high", 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        # Label(frame, text="Stream latency (s)").grid(row=7, column=0, sticky="EW")
        # Combobox(frame, width=10, textvariable=self.var_stream_latency, values=latencies).grid(row=7, column=1, sticky="EW", padx=5, pady=2.5)
        
        # Label(frame, text="Hardware latency (s)").grid(row=8, column=0, sticky="EW")
        # Spinbox(frame, textvariable=self.var_hardware_latency, width=10, from_=-1.0, to=1.0, increment=0.05).grid(row=8, column=1, sticky="EW", padx=5, pady=2.5)

        self.make_entry(frame, self.var_exposure, "Exposure (s)", 9, 0, 0.01, 10.0, 0.01)
    
        self.make_slider(frame, self.var_strength, "Strength", 20, 0.0, 100.0, self.update_test_stream)
        self.make_slider(frame, self.var_rotation, "Rotation", 21, -180, 180, self.update_test_stream)

        self.make_entry(frame, self.var_grid_x, "Grid x", 30, 0, 2, 100, 1)
        self.make_entry(frame, self.var_grid_y, "Grid y", 30, 2, 2, 100, 1)

        Checkbutton(frame, text="Show perimeter", variable=self.var_toggle_test, command=self.toggle_test_scanning).grid(row=40, column=0, sticky="EW")
        Button(frame, text="Plot coords", command=self.show_grid_plot).grid(row=40, column=1, sticky="EW", padx=5, pady=2.5)
        Button(frame, text="Save variables", command=self.to_yaml).grid(row=40, column=2, sticky="EW", padx=5, pady=2.5)
        Button(frame, text="Scan!", command=self.start_scanning).grid(row=40, column=3, sticky="EW", padx=5, pady=2.5)

        frame.pack(side="top", fill="x", padx=10, pady=10)

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
        self.var_dwell_time    = DoubleVar(value=0.05)
        self.var_dwell_time.trace_add("write", self.update_dwell_time)
        self.var_blocksize     = IntVar(value=1024)
        self.var_blocksize.trace_add("write", self.update_dwell_time)
        self.var_actual_dwell_time = StringVar(value="")
        
        self.var_stream_latency   = StringVar(value="low")
        self.var_hardware_latency = DoubleVar(value=0.0)

        self.var_exposure      = DoubleVar(value=0.01)
        self.var_grid_x        = IntVar(value=10)
        self.var_grid_x.trace_add("write", self.update_test_stream)
        self.var_grid_y        = IntVar(value=10)
        self.var_grid_y.trace_add("write", self.update_test_stream)
        self.var_strength      = DoubleVar(value=50.0)
        self.var_strength.trace_add("write", self.update_test_stream)
        self.var_rotation      = DoubleVar(value=0.0)
        self.var_toggle_test   = BooleanVar(value=False)

    def set_trigger(self, trigger=None, q=None):
        self.triggerEvent = trigger
        self.q = q

    def update_dwell_time(self, *args):
        try:
            dwell_time = self.var_dwell_time.get()
            blocksize = self.var_blocksize.get()

            fs = 44100
            ft = 1/fs
            
            if blocksize == 0:
                 blocksize = int(dwell_time / ft)

            nblocks = int((dwell_time / ft) // blocksize)
            dwell_time = ft*blocksize*nblocks
            self.var_actual_dwell_time.set(f" -> {dwell_time:.3f} s @ {blocksize} (n={nblocks})")
        except:
            pass

    def get_scanning_params(self):
        params = { "dwell_time": self.var_dwell_time.get(),
                   "blocksize": self.var_blocksize.get(),
                   "stream_latency": self.var_stream_latency.get(),
                   "hardware_latency": self.var_hardware_latency.get(),
                   "grid_x": self.var_grid_x.get(),
                   "grid_y": self.var_grid_y.get(),
                   "strength": self.var_strength.get() / (100.0 * global_damping_factor),
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

    def stop(self):
        if self.var_toggle_test.get():
            self.var_toggle_test.set(False)
        self.beam_ctrl.stop()

    def update_test_stream(self, *args):
        if self.var_toggle_test.get():
            params = self.get_scanning_params()
            
            grid_x = params["grid_x"]
            grid_y = params["grid_y"]
            coords = get_coords(**params).reshape(grid_y, grid_x, 2)

            corners = [
            coords[ 0,  0],
            coords[ 0, -1],
            coords[-1, -1],
            coords[-1,  0],
            ]

            self.beam_ctrl.do_box_scan(corners)

    def start_scanning(self, *args):
        self.stop()
        params = self.get_scanning_params()
        self.q.put(("scanning", params))
        self.triggerEvent.set()

    def show_grid_plot(self, *args):
        params = self.get_scanning_params()
        self.q.put(("plot_scan_grid", params))
        self.triggerEvent.set()

    def to_yaml(self, *args):
        import yaml
        from pathlib import Path
        params = self.get_scanning_params()
        params.pop("beam_ctrl")
        outfile = "input.yaml"
        yaml.dump(params, stream=open(outfile, "w"), default_flow_style=False)
        print(f"Wrote file: {Path(outfile).absolute()}")


if __name__ == '__main__':
    from ..settings import default_settings as settings

    beam = BeamCtrl(**settings)

    root = Tk()
    BeamCtrlFrame(root, settings=settings, beam_ctrl=beam_ctrl).pack(side="top", fill="both", expand=True)
    root.mainloop()

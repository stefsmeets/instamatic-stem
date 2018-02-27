from tkinter import *
from tkinter.ttk import *
from instamatic.utils.spinbox import Spinbox

import numpy as np
from PIL import Image, ImageTk

from instamatic.formats import read_tiff
from scipy import ndimage

from instamatic.tools import autoscale


class NavigationFrame(LabelFrame):
    """docstring for BeamCtrlFrame"""
    def __init__(self, parent):
        LabelFrame.__init__(self, parent, text="Navigation")
        self.parent = parent

        self.init_vars()

        frame = Frame(self)

        Checkbutton(frame, text="Enable", variable=self.var_toggle_stream, command=self.toggle_stream).grid(row=5, column=1, sticky="W")
        Button(frame, text="Load navigation image", command=self.load_image).grid(row=5, column=0, sticky="W")

        frame.pack(side="top", fill="x", padx=10, pady=10)

        frame = Frame(self)

        self.initialize_image(frame)
        # self.load_image("C:\instamatic\work_2018-02-27\experiment_8\image.tiff")
        
        frame.pack(side="top", fill="both", padx=10, pady=10)

    def init_vars(self):
        self.maxdim   = 256
        self.scale    = 1.0
        self.strength = 0.5
        self.channel_data = [0, 0]

        self.var_toggle_stream   = BooleanVar(value=False)

    def set_trigger(self, trigger=None, q=None):
        self.triggerEvent = trigger
        self.q = q

    def toggle_stream(self):
        toggle = self.var_toggle_stream.get()
        if toggle:
            self.update_stream(state="start")
        else:
            self.update_stream(state="stop")

    def update_stream(self, *args, state="continue"):
        params = {"channel_data": self.channel_data}
        params["state"] = state
        self.q.put(("beam_control", params))
        self.triggerEvent.set()

    def get_params(self):
        return {}

    def initialize_image(self, master):
        im = np.ones((self.maxdim, self.maxdim))
        image = Image.fromarray(im)
        image = ImageTk.PhotoImage(image)
        self.panel = Label(master, borderwidth=0, relief=GROOVE)
        self.panel.configure(image=image)
        self.panel.image = image
        self.panel.pack(fill="none", expand=False)
        self.panel.bind('<Button-1>', self.callback)

    def load_image(self, fn=None):
        from tkinter import filedialog
        if not fn:
            fn = filedialog.askopenfilename(parent=self.parent, initialdir=".", title="Select navigation image")
        if not fn:
            return

        im, h = read_tiff(fn)
        image, scale = autoscale(im, self.maxdim)
        self.scale = scale

        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        self.panel.configure(image=image)
        self.panel.image = image

        strength = h["scan_strength"]
        grid_x = h["scan_grid_x"]
        grid_y = h["scan_grid_y"]
        rotation = h["scan_rotation"]

        assert (grid_x, grid_y) == im.shape, (im.shape, grid_x, grid_y)

        from ..experiment import get_coords
        self.coords = get_coords(grid_x=grid_x, grid_y=grid_y, strength=strength, rotation=rotation).reshape(grid_x, grid_y, 2)

        print(self.coords.shape)


    def callback(self, event):
        x = min(event.x - 2, self.maxdim)
        y = min(event.y - 2, self.maxdim)

        x = int(x/self.scale)
        y = int(y/self.scale)

        self.channel_data = self.coords[x, y]
        self.update_stream()


from instamatic.gui.base_module import BaseModule

module = BaseModule("nav", "Navigation", True, NavigationFrame, commands={
} )


if __name__ == '__main__':
    from ..settings import default_settings as settings

    root = Tk()
    NavigationFrame(root, settings=settings).pack(side="top", fill="both", expand=True)
    root.mainloop()

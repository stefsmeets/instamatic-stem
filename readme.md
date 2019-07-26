# Instamatic-STEM

STEM diffraction module for [instamatic](https://github.com/stefsmeets/instamatic). Interfaces with the Nanomegas Digiscan and synchronizes the beam position with the Timepix camera at a maximum framerate of ~70 fps (on a JEOL JEM-2100).

Communcation with the deflector coils is done using a [RME Fireface 802](https://www.rme-audio.de/en/products/fireface_802.php) sound card that plugs into the Nanomegas daughterboard via an optical cable (ADAT). The soundcard is driven by the excellent [sounddevice](https://python-sounddevice.readthedocs.io/) python library, which offers high-level bindings for [PortAudio](http://www.portaudio.com/)). The deflectors are best driven over the ASIO API, which offers the most consistent low latency.

The signal itself is constructed as a 16-bit waveform, essentially a stream of float32 numbers from -1.0 to +1.0 as a C-contiguous array. It describes the modulation of the beam deflectors from the values set by the TEM for each of the channels (i.e. deflectors). Note that the modulations are applied _after_ the TEM settings, and is therefore subject to the current lens and deflector alignment. This means that any signal can be constructed and sent to the TEM. The regular alignment is maintained after the signal is turned off.

### Mapping

- Channels 1/2: Beamshift X/Y (CLA1)
- Channels 3/4: Beamtilt X/Y (CLA2)
- Channels 5/6: ImageShift 1 X/Y (IS1)
- Channels 7/8: ImageShift 2 X/Y (IS2)

### Maintainance

This was a small project to better understand how the beam deflectors work. I no longer have access to the equipment and therefore no longer able to maintain/test this code.

## Usage

Start the gui by typing `instamatic.stem` in the command line.

### Rasterscan

The STEM module for instamatic can set up a simple raster scan for doing STEM diffraction. It works surprisingly well and can achieve framerates of up to 70 fps. Some options of the GUI are explained here.

- Dwell_time: The time the beam stays at every position. This fines the window in which it is possible to capture an image.
- Exposure: Exposure time of the camera. The acquisition time (exposure time + overhead) must fit within the window defined by the dwell_time
- Strenght, grid_x, grid_y, rotation: defines the grid of coordinates for the beam.
- Shuffle coordinates: Randomize the raster coordinates as to minimize beam damage
- Show perimeter: Move the beam over the outlines of the scan

The raw data are saved as `data.h5`, the virtual image (sum of all individual patterns) is saved as `image.tiff`, and the scan metadata to `scan_log.txt`.

### Blocksize & latency

- Block size/Buffer size: The number of frames that are written to the audio buffer during every callback. There are 44100 frames in a second (44100 hz). A smaller blocksize will mean a lower latency can be achieved, but the audio feed may not be able to keep up.
- Latency: The time it takes between a signal being written to the audio buffer and beam being moved (i.e. sound coming out of the speaker or the beam being deflected). This value should be known accurately so that the capture window can be predicted. This is needed to synchronize the camera with the beam position.

The key factor here is the latency. Blocksize does not matter if the latency is too low. A too low latency will cause jitter. The internal buffer size for the Fireface 802 can be changed in the Fireface USB Settings app, and is independent from the one used in PortAudio.

### Settings

Various settings can be changed in [instamatic_stem/settings.py](instamatic_stem/settings.py)

## Requirements

 - Python3.6
 - sounddevice
 - matplotlib
 - numpy

`pip install -r requirements.txt`

## Installation

The latest development version can always be installed via:
    
    pip install https://github.com/stefsmeets/instamatic-stem/archive/master.zip

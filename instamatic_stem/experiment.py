import numpy as np
import sounddevice as sd
import threading
import time
from collections import deque, defaultdict
import os, sys
from pathlib import Path
from instamatic.formats import write_tiff


import h5py as h5


def printer(data):
    """Print things to stdout on one line dynamically"""
    sys.stdout.write("\r\x1b[K"+data.__str__())
    sys.stdout.flush()


def get_coords(grid_x=10, grid_y=10, strength=0.5, rotation=0.0, **kwargs):
    mx = float(grid_x) / max(grid_x, grid_y)
    my = float(grid_y) / max(grid_x, grid_y)

    x = np.linspace(-strength*mx, strength*mx, grid_x)
    y = np.linspace(-strength*my, strength*my, grid_y)

    xx, yy = np.meshgrid(x, y)
    coords = np.vstack((xx.reshape(-1), yy.reshape(-1))).T

    theta = np.radians(rotation)
    r = np.array( [[ np.cos(theta), -np.sin(theta) ],
                   [ np.sin(theta),  np.cos(theta) ]] )

    return np.dot(coords, r)


def signal_generator(coords, pre_fill=2, post_fill=5, repeat=10):
    """Convert the coordinate signal array into a generator
    pre_fill and post_fill add some blocks before and after the signal

    yield (collect[bool], coordinate[x, y])"""

    for _ in range(pre_fill*repeat):
        yield False, [0, 0]
    for row in coords:
        yield True, row
        for n in range(repeat-1):
            yield False, row
    for _ in range(post_fill*repeat):
        yield False, [0, 0]


def do_experiment(cam, beam_ctrl,
    strength,
    grid_x,
    grid_y,
    dwell_time,
    rotation,
    exposure,
    shuffle_coords=False,
    blocksize=1024,
    stream_latency=None,
    hardware_latency=0.0,
    expdir=None,
    plot=False,
    gui=True):

    channels = beam_ctrl.n_channels
    fs = beam_ctrl.fs

    print("starting experiment")
    print()
    print(f"    Dwell time: {dwell_time}")
    print(f"    Exposure:   {exposure}")
    print(f"    Latency:    {stream_latency} + {hardware_latency}")
    print(f"    Exposure:   {exposure}")
    print(f"    Grid x:     {grid_x}")
    print(f"    Grid y:     {grid_y}")
    print(f"    Rotation:   {rotation}")
    print(f"    Strength:   {strength}")
    print()

    ft = 1/fs

    if blocksize == 0:
        blocksize = int(dwell_time / ft)

    nblocks = int((dwell_time / ft) // blocksize)
    dwell_time = ft*blocksize*nblocks

    print(f"Block duration: {blocksize*ft:.3f} ms")
    print(f"N blocks: {nblocks}")
    print(f"Adjusted dwell time: {dwell_time:.3f} s @ {blocksize} frames/block")
    print()

    assert dwell_time > exposure, f"Dwell time ({dwell_time} s) must be larger than exposure ({exposure} s)"

    i = 0

    def callback(outdata, frames, streamtime, status):
        assert frames == blocksize

        if status.output_underflow:
            print('Output underflow: reduce exposure?')
            raise sd.CallbackAbort
        elif status.priming_output:
            pass
        else:
            assert not status, str(status)

        try:
            collect, coord = next(gen_coords)
            data.reshape(-1)[0::channels] = coord[0]
            data.reshape(-1)[1::channels] = coord[1]
        except StopIteration:
            raise sd.CallbackStop

        if collect:
            # print(f"{streamtime.currentTime-start_time:.3f} - {streamtime.outputBufferDacTime-start_time:.3f}, {streamtime.outputBufferDacTime-streamtime.currentTime:.3f} ")
            queue.append(streamtime.outputBufferDacTime)
        # else:
            # print(f"Collect: False @ {streamtime.outputBufferDacTime:6.3f}")

        outdata[:] = data

    data = np.zeros(blocksize*channels, dtype=np.float32).reshape(-1, channels)

    event = threading.Event()

    try:
        stream_latency = float(stream_latency)
    except TypeError:
        if stream_latency is None:
            stream_latency = beam_ctrl.latency
    except ValueError:
        # must be 'low' or 'high'
        pass

    # dither: Add random noise to make signal less determininistic, I assume we do not want this
    stream = sd.OutputStream(
            samplerate=fs, blocksize=blocksize, latency=stream_latency,
            device=beam_ctrl.device, channels=beam_ctrl.n_channels, dtype=beam_ctrl.dtype,
            callback=callback, finished_callback=event.set, 
            extra_settings=beam_ctrl.extra_settings,
            dither_off=True,
            prime_output_buffers_using_stream_callback=True)

    coords = get_coords(grid_x, grid_y, strength, rotation)

    if shuffle_coords:
        permutation = np.random.permutation(len(coords))
        coords = coords[permutation]

    gen_coords = signal_generator(coords, repeat=nblocks)

    queue = deque()
    buffer = []

    waiting_times = []
    frame_start_times = []
    frame_times = []
    missed = []
    empty = np.zeros((516, 516), dtype=np.uint16)

    i = 0

    time.sleep(0.25)

    if gui:
        cam.block()

    with stream:
        start_time = stream.time
        window_start = start_time

        print(f"Reported stream latency: {stream.latency:.3f}")
        print(f"Stream start time: {start_time:.3f}")
        print()

        while stream.active or len(queue):
            previous_window_start = window_start

            try:
                next_frame_time = queue.popleft()
            except IndexError:
                # print(f" -> No frames @ {stream.time-start_time:6.3f}, continuing!")
                time.sleep(dwell_time / 2)
                continue
            else:
                i += 1

            s = f"\r {i:4d} "

            current_time = stream.time

            window_start = next_frame_time + hardware_latency
            window_end   = window_start + dwell_time

            s += f"[ c: {current_time - start_time:.3f} | n: {window_start - start_time:.3f} | d: {window_start - previous_window_start:.3f} ] -> "

            if current_time < window_start:
                diff = window_start - current_time
                s += f"Waiting: {diff:.3f} s"
                time.sleep(window_start - current_time)
            elif current_time > window_end - exposure:
                s += f"        +{current_time - window_start:.3f} s -> Miss :( "
                missed.append(i)
                buffer.append(empty)  # insert empty to maintain correct data shape
                print(s, end='')
                continue
            else:
                s += f"        +{current_time - window_start:.3f} s"

            arr = cam.getImage(exposure).astype(np.uint16, copy=False)  # copy=False ensures that no copy is made if dtype is already satisfied
            buffer.append(arr)
            s += f" -> OK!     [ Miss: {len(missed)} ]         "
            print(s, end='')

            frame_time = window_start - previous_window_start

            frame_times.append(frame_time)
            frame_start_times.append(window_start)
            waiting_times.append(window_start - current_time)

        event.wait()  # Wait until playback is finished

    if gui:
        cam.unblock()

    ntot = len(coords)
    nmissed = len(missed)
    print()
    print("Scanning done!")
    print(f"Stream latency:     {1000*stream.latency:.2f} ms")
    print(f"Average wait:       {1000*np.mean(waiting_times[1:]):.2f} +- {1000*np.std(waiting_times[1:]):.2f} ms")
    print(f"Average frame time: {1000*np.mean(frame_times[1:]):.2f} +- {1000*np.std(frame_times[1:]):.2f} ms")
    
    dt = max(frame_start_times) - min(frame_start_times)
    print(f"Time taken:         {dt:.1f} s ({ntot} frames)")
    print(f"Frametime:          {1000*(dt)/ntot:.1f} ms ({ntot/(dt):.1f} fps)")
    print(f"Missed frames:      {nmissed} ({nmissed/ntot:.1%})")
    
    if not expdir:
        expdir = Path.cwd()
        
    write_output = True
    write_to_hdf5 = True
    write_to_tiff = False

    if shuffle_coords:
        idx = np.argsort(permutation)
    else:
        idx = np.arange(len(buffer))

    if write_output:
        if write_to_tiff:
            for i in idx:
                arr = buffer[i]
                write_tiff(expdir / f"{i:04d}.tiff", arr)
                printer(f"{i} / {len(buffer)}")
        
        if write_to_hdf5:
            x, y = buffer[0].shape
            dtype = buffer[0].dtype
            fn = expdir / f"data.h5"
            f = h5.File(fn)
            d = f.create_dataset("STEM-diffraction", shape=(len(buffer), x, y), dtype=dtype)
            for i in idx:
                arr = buffer[i]
                printer(f"{i} / {len(buffer)}")
                d[i] = arr
            f.close()
    
            printer(f"Wrote buffer to {fn}\n")

        im = np.zeros((grid_x, grid_y))
        im_view = im.ravel()
        for i in idx:
            arr = buffer[i]
            im_view[i] = arr.sum()

        h = {
            "scan_strength": strength,
            "scan_grid_x": grid_x,
            "scan_grid_y": grid_y,
            "scan_dwell_time": dwell_time,
            "scan_rotation": rotation,
            "scan_exposure": exposure,
        }

        image_fn = expdir / "image.tiff"
        write_tiff(expdir / image_fn, im, header=h)
        print(f"Wrote image to {image_fn}")

        with open(expdir / "scan_log.txt", "w") as f:
            print(time.ctime(), file=f)
            print(file=f)
            print("dwell_time:", dwell_time, file=f)
            print("exposure:", exposure, file=f)
            print("strength:", strength, file=f)
            print("grid_x:", grid_x, file=f)
            print("grid_y:", grid_y, file=f)
            print("rotation:", rotation, file=f)
            print(file=f)
            print(beam_ctrl.info(), file=f)
            print(file=f)
            print("Missed", file=f)
            print(missed, file=f)
            print(file=f)
            print("Coords", file=f)
            print(str(coords), file=f)
            print(file=f)

            print("Wrote info to", f.name)


def main():
    import psutil, os
    p = psutil.Process(os.getpid())
    p.nice(psutil.REALTIME_PRIORITY_CLASS)  # set python process as high priority

    from .settings import default
    from instamatic.camera.videostream import VideoStream
    from instamatic.camera import Camera
    from .beam_control import BeamCtrl
    from instamatic import config
    import yaml

    beam_ctrl = BeamCtrl(**default)
    cam = Camera(config.cfg.camera)

    args = sys.argv[1:]
    if not args:
        print("Usage: instamatic.stem.nogui input.yaml")
        sys.exit()

    kwargs = yaml.load(open(args[0], "r"))

    do_experiment(cam               = cam,
                  beam_ctrl         = beam_ctrl,
                  gui               = False,
                  **kwargs)

if __name__ == '__main__':
    main()

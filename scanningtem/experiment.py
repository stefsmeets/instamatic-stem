import numpy as np
import sounddevice as sd
import threading
import Queue
import time
from collections import deque, defaultdict
from monotonic import monotonic

from IPython import embed


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


def signal_generator(coords, pre_fill=5, post_fill=1):
    """Convert the coordinate signal array into a generator
    pre_fill and post_fill add some blocks before and after the signal

    yield (collect[bool], coordinate[x, y])"""

    for _ in range(pre_fill):
        yield False, [0, 0]
    for row in coords:
        yield True, row
    for _ in range(post_fill):
        yield False, [0, 0]


def do_experiment(cam, strength,
    grid_x,
    grid_y,
    dwell_time,
    rotation,
    exposure,
    beam_ctrl,
    plot=False):

    channels = beam_ctrl.n_channels

    print "starting experiment"
    print
    print "    Dwell time: {}".format(dwell_time)
    print "    Exposure:   {}".format(exposure)
    print "    Grid x:     {}".format(grid_x)
    print "    Grid y:     {}".format(grid_y)
    print "    Rotation:   {}".format(rotation)
    print "    Strength:   {}".format(strength)
    print
    blocksize = int(dwell_time * beam_ctrl.fs)
    block_duration = float(blocksize) / beam_ctrl.fs
    print "    Blocksize:  {}".format(blocksize)
    print

    assert dwell_time > exposure, "Dwell time ({} s) must be larger than exposure ({} s)".format(dwell_time, exposure)


    def callback(outdata, frames, streamtime, status):
        assert frames == blocksize

        if status.output_underflow:
            print 'Output underflow: reduce exposure?'
            raise sd.CallbackAbort
        assert not status

        try:
            collect, coord = gen_coords.next()
            data.reshape(-1)[0::channels] = coord[0]
            data.reshape(-1)[1::channels] = coord[1]
        except StopIteration:
            print "Stopping now!!"
            raise sd.CallbackStop

        if collect:
            queue.append(streamtime.outputBufferDacTime)

        outdata[:] = data

    data = np.zeros(blocksize*channels, dtype=np.float32).reshape(-1, channels)

    event = threading.Event()

    stream = sd.OutputStream(
            samplerate=beam_ctrl.fs, blocksize=blocksize, latency=beam_ctrl.latency,
            device=beam_ctrl.device, channels=beam_ctrl.n_channels, dtype=beam_ctrl.dtype,
            callback=callback, finished_callback=event.set, dither_off=True)

    # stream = beam_ctrl.get_output_stream(callback=callback, finished_callback=event.set, blocksize=blocksize)

    coords = get_coords(grid_x, grid_y, strength, rotation)
    gen_coords = signal_generator(coords)

    starttime = monotonic()
    queue = deque()
    buffer = []

    waiting_times = []
    missed = []

    i = 0

    t0 = time.clock()
    cam.block()

    with stream:

        while stream.active or len(queue):
            try:
                next_frame = queue.popleft()
            except IndexError:
                print " -> No frames, continuing!"
                time.sleep(dwell_time / 4)
                continue
            else:
                i += 1

            diff = next_frame - stream.time
            print "Waiting {:-6.1f} ms (current: {:-6.3f}, next: {:-6.3f})".format(1000*diff, stream.time - starttime, next_frame - starttime),
            if diff < 0:
                if diff + dwell_time > exposure*1.5:
                    print " -> second chance",
                else:
                    print " -> Missed the window, continuing! (latency: {})".format(stream.latency)
                    missed.append(i)
                    continue
            else:
                time.sleep(diff)
            
            cam.getImage(exposure)
            arr = cam.getImage(exposure)
            buffer.append(arr)
            print " -> Image captured!"

            waiting_times.append(diff)

        # time.sleep(1)
        # event.wait()  # Wait until playback is finished
        print
        print "Scanning done!"
        print "Stream latency: {} s".format(stream.latency)
        print "Average wait: {:.2f} +- {:.2f} ms".format(1000*np.mean(waiting_times[1:]), 1000*np.std(waiting_times[1:]))

    cam.unblock()
    t1 = time.clock()

    print "Time taken: {:.1f} s".format(t1-t0)
    print "Frametime: {:.1f} ms ({:.1f} fps)".format(1000*(t1-t0)/len(coords), len(coords)/(t1-t0))
    print "Missed frames: {} ({:.1%})".format(len(missed), float(len(missed))/len(coords))
    buffer = np.stack(buffer)
    t = time.time()
    fn = "scan_{}.npy".format(t)
    # np.save(fn, buffer)
    print "Wrote buffer to", fn

    with open("scan_{}.txt".format(t), "w") as f:
        print >> f, time.ctime()
        print >> f
        print >> f, "dwell_time:", dwell_time
        print >> f, "exposure:", exposure
        print >> f, "strength:", strength
        print >> f, "grid_x:", grid_x
        print >> f, "grid_y:", grid_y
        print >> f, "rotation:", rotation
        print >> f
        print >> f, beam_ctrl.info()
        print >> f
        print >> f, "Missed"
        print >> f, missed
        print >> f
        print >> f, "Coords"
        print >> f, str(coords)
        print >> f
        print "Wrote info to", f.name


def do_experiment_continuous(cam, strength,
    grid_x,
    grid_y,
    dwell_time,
    rotation,
    exposure,
    beam_ctrl,
    plot=False):

    channels = beam_ctrl.n_channels

    print "starting experiment"
    print
    print "    Dwell time: {}".format(dwell_time)
    print "    Exposure:   {}".format(exposure)
    print "    Grid x:     {}".format(grid_x)
    print "    Grid y:     {}".format(grid_y)
    print "    Rotation:   {}".format(rotation)
    print "    Strength:   {}".format(strength)
    print
    blocksize = int(dwell_time * beam_ctrl.fs)
    block_duration = float(blocksize) / beam_ctrl.fs
    print "    Blocksize:  {}".format(blocksize)
    print

    assert dwell_time > exposure, "Dwell time ({} s) must be larger than exposure ({} s)".format(dwell_time, exposure)

    grid_x_modified = 2  # for now, fix me later
    coords = get_coords(grid_x_modified, grid_y, strength, rotation)

    from beam_control import ramp

    buffer = []

    for i in range(len(coords))[::2]:
        coord1 = coords[i]
        coord2 = coords[i+1]

        signal = ramp(coord1, coord2, beam_ctrl.fs, 10.0)

        print signal, signal.shape

        # sd.play(signal, beam_ctrl.fs, mapping=beam_ctrl.mapping, loop=True, blocking=True)

        try:
            beam_ctrl.signal_data[:] = signal
        except ValueError:
            beam_ctrl.signal_data = np.zeros_like(signal)
            beam_ctrl.signal_data[:] = signal

        t0 = time.clock()
        beam_ctrl.play(loop=False, blocking=True)
        s = beam_ctrl.get_stream()

        while s.active:
            arr = cam.getImage(exposure)
            buffer.append(arr)

        t1 = time.clock()

        print t1 - t0

        print "sleep"
        beam_ctrl.stop()
        print "ramp", i

    return 

















    gen_coords = signal_generator(coords)

    starttime = monotonic()
    queue = deque()
    buffer = []

    waiting_times = []
    missed = []

    i = 0

    t0 = time.clock()
    cam.block()

    with stream:

        while stream.active or len(queue):
            try:
                next_frame = queue.popleft()
            except IndexError:
                print " -> No frames, continuing!"
                time.sleep(dwell_time / 4)
                continue
            else:
                i += 1

            diff = next_frame - stream.time
            print "Waiting {:-6.1f} ms (current: {:-6.3f}, next: {:-6.3f})".format(1000*diff, stream.time - starttime, next_frame - starttime),
            if diff < 0:
                if diff + dwell_time > exposure*1.5:
                    print " -> second chance",
                else:
                    print " -> Missed the window, continuing! (latency: {})".format(stream.latency)
                    missed.append(i)
                    continue
            else:
                time.sleep(diff)
            
            cam.getImage(exposure)
            arr = cam.getImage(exposure)
            buffer.append(arr)
            print " -> Image captured!"

            waiting_times.append(diff)

        event.wait()  # Wait until playback is finished
        print
        print "Scanning done!"
        print "Stream latency: {} s".format(stream.latency)
        print "Average wait: {:.2f} +- {:.2f} ms".format(1000*np.mean(waiting_times[1:]), 1000*np.std(waiting_times[1:]))

    cam.unblock()
    t1 = time.clock()

    print "Time taken: {:.1f} s".format(t1-t0)
    print "Frametime: {:.1f} ms ({:.1f} fps)".format(1000*(t1-t0)/len(coords), len(coords)/(t1-t0))
    print "Missed frames: {} ({:.1%})".format(len(missed), float(len(missed))/len(coords))
    buffer = np.stack(buffer)
    t = time.time()
    fn = "scan_{}.npy".format(t)
    # np.save(fn, buffer)
    print "Wrote buffer to", fn

    with open("scan_{}.txt".format(t), "w") as f:
        print >> f, time.ctime()
        print >> f
        print >> f, "dwell_time:", dwell_time
        print >> f, "exposure:", exposure
        print >> f, "strength:", strength
        print >> f, "grid_x:", grid_x
        print >> f, "grid_y:", grid_y
        print >> f, "rotation:", rotation
        print >> f
        print >> f, beam_ctrl.info()
        print >> f
        print >> f, "Missed"
        print >> f, missed
        print >> f
        print >> f, "Coords"
        print >> f, str(coords)
        print >> f
        print "Wrote info to", f.name


if __name__ == '__main__':
    import psutil, os
    p = psutil.Process(os.getpid())
    p.nice(psutil.REALTIME_PRIORITY_CLASS)  # set python process as high priority

    from settings import DEFAULT_SETTINGS
    from instamatic.camera.videostream import VideoStream
    from beam_control import BeamCtrl

    beam_ctrl = BeamCtrl(**DEFAULT_SETTINGS)
    cam = VideoStream("simulate")

    do_experiment(cam               = cam,
                  dwell_time        = 0.035,
                  exposure          = 0.010,
                  strength          = 50.0,
                  grid_x            = 20,
                  grid_y            = 20,
                  rotation          = 0.0,
                  beam_ctrl         = beam_ctrl)

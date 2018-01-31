import numpy as np
import sounddevice as sd
import threading
import Queue
import time
from collections import deque, defaultdict
from monotonic import monotonic

from IPython import embed


def do_experiment(cam, strength,
    grid_x,
    grid_y,
    dwell_time,
    rotation,
    exposure,
    beam_ctrl):

    print "starting experiment"
    print
    print "    Dwell time: {}".format(dwell_time)
    print "    Exposure:   {}".format(exposure)
    print "    Grid x:     {}".format(grid_x)
    print "    Grid y:     {}".format(grid_y)
    print "    Rotation:   {}".format(rotation)
    print "    Strength:   {}".format(strength)

    # assert dwell_time > exposure, "Dwell time ({} s) must be larger than exposure ({} s)".format(dwell_time, exposure)

    x = np.linspace(-strength, strength, grid_x)
    y = np.linspace(-strength, strength, grid_y)

    xx, yy = np.meshgrid(x, y)
    coords = np.vstack((xx.reshape(-1), yy.reshape(-1))).T

    def callback(outdata, frames, streamtime, status):
        assert frames == blocksize

        if status.output_underflow:
            print 'Output underflow: reduce exposure?'
            raise sd.CallbackAbort
        assert not status

        # outputBufferDacTime should correspond to when the data are played
        #  and thus when the image should be taken
        # print streamtime.currentTime - starttime, streamtime.outputBufferDacTime - starttime
        deq.append(streamtime.outputBufferDacTime)

        try:
            data = q.get_nowait()
        except Queue.Empty:
            # print 'Buffer is empty: increase buffersize?'
            raise sd.CallbackAbort

        if len(data) < len(outdata):
            outdata[:len(data)] = data
            outdata[len(data):] = 0
            print "stop"
            raise sd.CallbackStop
        else:
            outdata[:] = data

    fs = beam_ctrl.fs
    device = beam_ctrl.device
    channels = beam_ctrl.n_channels
    dtype = beam_ctrl.dtype
    buffersize = 10000
    latency = "high"

    print
    print "    fs:", fs
    print "    device:", device
    print "    channels:", channels
    print "    dtype:", dtype
    print "    buffersize:", buffersize
    print "    latency:", latency

    # blocksize = 1024
    # block_duration = float(blocksize) / fs
    # print "block duration (s):", block_duration
    
    blocksize = int(dwell_time * fs)
    block_duration = float(blocksize) / fs
    print "    Blocksize:", blocksize
    print

    event = threading.Event()
    q = Queue.Queue(maxsize=buffersize)

    data = np.zeros(blocksize*channels, dtype=np.float32).reshape(-1, channels)

    stream = sd.OutputStream(
        samplerate=fs, blocksize=blocksize, latency=latency,
        device=device, channels=channels, dtype=dtype,
        callback=callback, finished_callback=event.set, dither_off=True)
    
    theta = np.radians(rotation)
    r = np.array( [[ np.cos(theta), -np.sin(theta) ],
                   [ np.sin(theta),  np.cos(theta) ]] )

    coords = np.dot(coords, r)

    for coord in coords:
        # print q.qsize(), coord, i, imax, timeout
        data.reshape(-1)[0::channels] = coord[0]
        data.reshape(-1)[1::channels] = coord[1]

        q.put(data.copy())

    starttime = monotonic()
    deq = deque()
    buffer = []

    waiting_times = []
    missed = []

    i = 0

    cam.block()
    with stream:
        timeout = blocksize * buffersize / float(fs)

        while stream.active or len(deq):
            try:
                next_frame = deq.popleft()
            except IndexError:
                print " -> No frames, continuing!"
                time.sleep(dwell_time / 5)
                continue

            diff = next_frame - stream.time
            print "Waiting {:-6.1f} ms (current: {:-6.3f}, next: {:-6.3f})".format(1000*diff, stream.time - starttime, next_frame - starttime),
            if diff < 0:
                if diff + dwell_time > exposure:
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
            i += 1

        # print "timeout (s):", timeout ## not used
        event.wait()  # Wait until playback is finished
        print "Scanning done!"
        print "Average wait: {:.2f} +- {:.2f} ms".format(1000*np.mean(waiting_times[1:]), 1000*np.std(waiting_times[1:]))

    cam.unblock()
    print "Missed frames: {}".format(len(missed))
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
        print >> f, "fs:", fs
        print >> f, "device:", device
        print >> f, "channels:", channels
        print >> f, "dtype:", dtype
        print >> f, "buffersize:", buffersize
        print >> f, "latency:", latency
        print >> f
        print >> f, "Missed"
        print >> f, missed
        print >> f
        print >> f, "Coords"
        print >> f, str(coords)
        print >> f
        print "Wrote info to", f.name


if __name__ == '__main__':
    from settings import DEFAULT_SETTINGS
    from instamatic.camera.videostream import VideoStream
    from beam_control import BeamCtrl

    beam_ctrl = BeamCtrl(**DEFAULT_SETTINGS)
    cam = VideoStream("simulate")

    do_experiment(cam               = cam,
                  dwell_time        = 0.04,
                  exposure          = 0.01,
                  strength          = 50.0,
                  grid_x            = 20,
                  grid_y            = 20,
                  rotation          = 0.0,
                  beam_ctrl         = beam_ctrl)

import numpy as np
import sounddevice as sd
import threading
import Queue
import time
from collections import deque
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
    buffersize = 1000

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
        samplerate=fs, blocksize=blocksize,
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
    cam.block()
    with stream:
        timeout = blocksize * buffersize / float(fs)

        while stream.active or len(deq):
            try:
                next_frame = deq.popleft()
            except IndexError:
                print " -> No frames, continuing!"
                time.sleep(dwell_time / 2)
                continue

            diff = next_frame - stream.time
            print "Waiting {:.3f} s (current: {:.3f}, next: {:.3f})".format(diff, stream.time - starttime, next_frame - starttime), 
            if diff < 0:
                print " -> Missed the window, continuing!"
                continue
            
            time.sleep(diff)
            cam.getImage(exposure)
            arr = cam.getImage(exposure)
            buffer.append(arr)
            print " -> Image captured!"

        # print "timeout (s):", timeout ## not used
        event.wait()  # Wait until playback is finished
        print "Scanning done!"
    cam.unblock()

    buffer = np.stack(buffer)
    fn = "scan_{}.npy".format(time.time())
    # np.save(fn, buffer)
    print "Wrote buffer to", fn

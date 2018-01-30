import numpy as np
import sounddevice as sd
import threading
import Queue
import time


def do_experiment(cam, strength,
    grid_x,
    grid_y,
    dwell_time,
    rotation,
    beam_ctrl):

    print "starting experiment"
    print
    print "    Dwell time: {}".format(dwell_time)
    print "    Grid x:     {}".format(grid_x)
    print "    Grid y:     {}".format(grid_y)
    print "    Rotation:   {}".format(rotation)
    print "    Strength:   {}".format(strength)

    x = np.linspace(-strength, strength, grid_x)
    y = np.linspace(-strength, strength, grid_y)

    xx, yy = np.meshgrid(x, y)
    coords = np.vstack((xx.reshape(-1), yy.reshape(-1))).T

    def callback(outdata, frames, time, status):
        # print time.currentTime, time.inputBufferAdcTime, time.outputBufferDacTime

        assert frames == blocksize
        if status.output_underflow:
            print 'Output underflow: increase blocksize?'
            raise sd.CallbackAbort
        assert not status

        arr = cam.getImage(0.01)
        buffer.append(arr)

        try:
            data = q.get_nowait()
        except Queue.Empty:
            print 'Buffer is empty: increase buffersize?'
            raise sd.CallbackAbort

        if len(data) < len(outdata):
            outdata[:len(data)] = data
            outdata[len(data):] = np.zeros((len(outdata) - len(data), 2))
            print "stop"
            raise sd.CallbackStop
        else:
            outdata[:] = data

    fs = 44100
    device = None
    channels = 2
    dtype = 'float32'
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
        callback=callback, finished_callback=event.set)
    
    theta = np.radians(rotation)
    r = np.array( [[ np.cos(theta), -np.sin(theta) ],
                   [ np.sin(theta),  np.cos(theta) ]] )

    coords = np.dot(coords, r)

    for coord in coords:
        # print q.qsize(), coord, i, imax, timeout
        data.reshape(-1)[0::channels] = coord[0]
        data.reshape(-1)[1::channels] = coord[1]

        q.put(data.copy())

    buffer = []
    cam.block()
    with stream:
        timeout = blocksize * buffersize / float(fs)
        # print "timeout (s):", timeout ## not used

        event.wait()  # Wait until playback is finished
        time.sleep(2.0)
        print "Scanning done!"
    cam.unblock()

    buffer = np.stack(buffer)
    fn = "scan_{}.npy".format(time.time())
    np.save(fn, buffer)
    print "Wrote buffer to", fn

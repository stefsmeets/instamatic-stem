import numpy as np
import sounddevice as sd
import threading
import time
from collections import deque, defaultdict

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


def do_experiment(cam, strength,
    grid_x,
    grid_y,
    dwell_time,
    rotation,
    exposure,
    beam_ctrl,
    blocksize=512,
    latency=None,
	write_output=True,
    plot=False):

    channels = beam_ctrl.n_channels
    fs = beam_ctrl.fs

    print("starting experiment")
    print()
    print(f"    Dwell time: {dwell_time}")
    print(f"    Exposure:   {exposure}")
    print(f"    Grid x:     {grid_x}")
    print(f"    Grid y:     {grid_y}")
    print(f"    Rotation:   {rotation}")
    print(f"    Strength:   {strength}")
    print()

    ft = 1/fs
    nblocks = int((dwell_time / ft) // blocksize)
    dwell_time = ft*blocksize*nblocks

    print(f"Block duration: {blocksize*ft:.3f} ms")
    print(f"N blocks: {nblocks}")
    print(f"Adjusted dwell time: {dwell_time:.3f} s @ {blocksize} frames/block")

    assert dwell_time > exposure, f"Dwell time ({dwell_time} s) must be larger than exposure ({exposure} s)"

    i = 0

    def callback(outdata, frames, streamtime, status):
        assert frames == blocksize

        if status.output_underflow:
            print('Output underflow: reduce exposure?')
            raise sd.CallbackAbort
        assert not status

        try:
            collect, coord = next(gen_coords)
            data.reshape(-1)[0::channels] = coord[0]
            data.reshape(-1)[1::channels] = coord[1]
        except StopIteration:
            print("\nStopping now!!")
            raise sd.CallbackStop

        if collect:
            # print(f"{streamtime.currentTime-starttime:.3f} - {streamtime.outputBufferDacTime-starttime:.3f}, {streamtime.outputBufferDacTime-streamtime.currentTime:.3f} ")
            queue.append(streamtime.outputBufferDacTime)
        # else:
            # print(f"Collect: False @ {streamtime.outputBufferDacTime:6.3f}")

        outdata[:] = data

    data = np.zeros(blocksize*channels, dtype=np.float32).reshape(-1, channels)

    event = threading.Event()

    if not latency:
        latency = beam_ctrl.latency

    # dither: Add random noise to make signal less determininistic, I assume we do not want this
    stream = sd.OutputStream(
            samplerate=fs, blocksize=blocksize, latency=latency,
            device=beam_ctrl.device, channels=beam_ctrl.n_channels, dtype=beam_ctrl.dtype,
            callback=callback, finished_callback=event.set, 
            extra_settings=beam_ctrl.extra_settings,
            dither_off=True,
            prime_output_buffers_using_stream_callback=True)

    coords = get_coords(grid_x, grid_y, strength, rotation)
    gen_coords = signal_generator(coords, repeat=nblocks)

    queue = deque()
    buffer = []

    waiting_times = []
    dts = []
    frame_times = []
    missed = []
    empty = np.zeros((516, 516))

    i = 0

    time.sleep(0.25)

    t0 = time.clock()
    cam.block()

    with stream:
        starttime = stream.time
        previous_frame_time = starttime

        print(f"Reported stream latency: {stream.latency:.3f}")
        print(f"Stream start time: {starttime:.3f}")

        time.sleep(stream.latency)

        while stream.active or len(queue):
            try:
                next_frame = queue.popleft()
            except IndexError:
                print(f" -> No frames @ {stream.time-starttime:6.3f}, continuing!")
                time.sleep(stream.latency)
                continue
            else:
                i += 1

            stream_time = stream.time
            frame_time = next_frame + stream.latency

            diff = frame_time - stream_time
            print(f"Waiting {1000*diff:-6.1f} ms (current: {stream.time-starttime:-6.3f} s, next: {frame_time-starttime:-6.3f}, dt: {frame_time-previous_frame_time:-6.3f})", end=" ")
            dt = frame_time - previous_frame_time
            previous_frame_time = frame_time
            
            if diff < 0:
                if diff + dwell_time > exposure*1.5:
                    pass
                    print(" -> ...", end='')
                else:
                    print(" -> Missed the window")
                    missed.append(i)
                    buffer.append(empty)  # insert empty to maintain correct data shape
                    continue
            else:
                time.sleep(diff)
            
            arr = cam.getImage(exposure)
            buffer.append(arr)
            print(" -> OK!")

            dts.append(dt)
            frame_times.append(frame_time)
            waiting_times.append(diff)

        # time.sleep(1)
        # event.wait()  # Wait until playback is finished

    cam.unblock()
    t1 = time.clock()

    ntot = len(coords)
    nmissed = len(missed)
    print()
    print("Scanning done!")
    print(f"Stream latency:     {1000*stream.latency:.2f} ms")
    print(f"Average wait:       {1000*np.mean(waiting_times[1:]):.2f} +- {1000*np.std(waiting_times[1:]):.2f} ms")
    print(f"Average frame time: {1000*np.mean(dts[1:]):.2f} +- {1000*np.std(dts[1:]):.2f} ms")
    
    dt = max(frame_times) - min(frame_times)
    print(f"Time taken:         {dt:.1f} s ({ntot} frames)")
    print(f"Frametime:          {1000*(dt)/ntot:.1f} ms ({ntot/(dt):.1f} fps)")
    print(f"Missed frames:      {nmissed} ({nmissed/ntot:.1%})")
    
    if write_output:
        t = time.time()
        buffer = np.stack(buffer)
        fn = f"scan_{t}.npy"
        np.save(fn, buffer)
        print(f"Wrote buffer to {fn}")
    
        with open("scan_{}.txt".format(t), "w") as f:
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
    from .beam_control import BeamCtrl

    beam_ctrl = BeamCtrl(**default)
    cam = VideoStream("simulate")

    do_experiment(cam               = cam,
                  dwell_time        = 0.035,
                  exposure          = 0.01,
                  strength          = 1.0 / 100,
                  grid_x            = 10,
                  grid_y            = 10,
                  rotation          = 0.0,
                  blocksize         = 32,
                  latency           = 0.0,
                  write_output      = False,
                  beam_ctrl         = beam_ctrl)
    cam.close()

if __name__ == '__main__':
    main()

from __future__ import print_function
from ctypes import wintypes
import ctypes
import atexit

winmm = ctypes.WinDLL('winmm')


class TIMECAPS(ctypes.Structure):
    _fields_ = (('wPeriodMin', wintypes.UINT),
                ('wPeriodMax', wintypes.UINT))


def enable(milliseconds=1):
    caps = TIMECAPS()
    winmm.timeGetDevCaps(ctypes.byref(caps), ctypes.sizeof(caps))
    milliseconds = min(max(milliseconds, caps.wPeriodMin), caps.wPeriodMax)

    winmm.timeBeginPeriod(milliseconds)

    print("Change time period to {} ms".format(milliseconds))

    def reset_time_period():
        print("Reset time period from {} ms".format(milliseconds))
        winmm.timeEndPeriod(milliseconds)

    atexit.register(reset_time_period)


if __name__ == '__main__':
    import timeit

    setup = 'import time'
    stmt = 'time.sleep(0.001)'
    print(timeit.timeit(stmt, setup, number=1000))

    print("change time period")
    enable(1)

    setup = 'import time'
    stmt = 'time.sleep(0.001)'
    print(timeit.timeit(stmt, setup, number=1000))

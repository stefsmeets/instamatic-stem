from __future__ import print_function
import ctypes

# https://github.com/atdt/monotonic

kernel32 = ctypes.cdll.kernel32

GetTickCount64 = getattr(kernel32, 'GetTickCount64', None)

GetTickCount64.restype = ctypes.c_ulonglong

def monotonic():
    """Monotonic clock, cannot go backward."""
    return GetTickCount64() / 1000.0

if __name__ == '__main__':
    print("Monotonic time:", monotonic())

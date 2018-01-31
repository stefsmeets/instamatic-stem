settings_fireface_all = {
'device': 'ASIO Fireface USB',
'global_volume': 1.0,
'fs': 44100,
'duration': 0.01,
'n_channels': 8,
'chunksize': 1024,
'mapping': (15,16,17,18,19,20,21,22),
'channels': (
    {"name": "BeamShift X", "var": None, "default": 0},
    {"name": "BeamShift Y", "var": None, "default": 0},
    {"name": "BeamTilt X?", "var": None, "default": 0},
    {"name": "BeamTilt Y?", "var": None, "default": 0},
    {"name": "ImageShift? X", "var": None, "default": 0},
    {"name": "ImageShift? Y", "var": None, "default": 0},
    {"name": "ImageTilt? X?", "var": None, "default": 0},
    {"name": "ImageTilt? Y?", "var": None, "default": 0}
)}

settings_fireface = {
'device': 'ADAT 1 (1+2) (RME Fireface 802)',
'global_volume': 1.0,
'fs': 44100,
# 'hostapi': 0,  # 'Windows MME'           # works well with 5/1
# 'hostapi': 1,  # 'Windows DirectSound'   # Always missing the window, strange timing issues
'hostapi': 3,  # 'Windows WASAPI'        # works well with 5/1, but misses sporadically
'duration': 0.01,
'n_channels': 2,
'chunksize': 1024,
'mapping': (1, 2),
'channels': (
    {"name": "BeamShift X", "var": None, "default": 0},
    {"name": "BeamShift Y", "var": None, "default": 0},
)}

# Do not use, clock is unreliable?
# I get different values from stream.time and time.getBufferDacTime
# maybe a different clock is used?
settings_fireface_WDM_K5 = {
'device': 'ADAT 1 (1+2) (Fireface ADAT 1 (1+2))',
'global_volume': 1.0,
'fs': 44100,
'hostapi': 4,  # 'Windows WDM-KS'
'duration': 0.01,
'n_channels': 2,
'chunksize': 1024,
'mapping': (1, 2),
'channels': (
    {"name": "BeamShift X", "var": None, "default": 0},
    {"name": "BeamShift Y", "var": None, "default": 0},
)}

settings_testing = {
'device': None,
'global_volume': 1.0,
'fs': 44100,
'duration': 1.0,
'n_channels': 2,
'chunksize': 1024,
'mapping': (1,2),
'channels': (
    {"name": "Channel 1", "var": None, "default": 0},
    {"name": "Channel 2", "var": None, "default": 0}
)}

DEFAULT_SETTINGS = settings_fireface
# DEFAULT_SETTINGS = settings_testing
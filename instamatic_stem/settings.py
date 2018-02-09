asio_base = {
'device': 'ASIO Fireface USB',
'hostapi': 2,  # asio
}

cla1_asio = {**asio_base,
**{'mapping': (14,15),
   'channels': (
    {"name": "BeamShift X", "var": None, "default": 0},
    {"name": "BeamShift Y", "var": None, "default": 0})
}}

all_asio = {**asio_base,
**{'mapping': (15,16,17,18,19,20,21,22),
   'channels': (
      {"name": "BeamShift X", "var": None, "default": 0},
      {"name": "BeamShift Y", "var": None, "default": 0},
      {"name": "BeamTilt X?", "var": None, "default": 0},
      {"name": "BeamTilt Y?", "var": None, "default": 0},
      {"name": "ImageShift? X", "var": None, "default": 0},
      {"name": "ImageShift? Y", "var": None, "default": 0},
      {"name": "ImageTilt? X?", "var": None, "default": 0},
      {"name": "ImageTilt? Y?", "var": None, "default": 0})
}}

cla1_adat_base = {
'device': 'ADAT 1 (1+2) (RME Fireface 802)',
'mapping': (1, 2),
'channels': (
    {"name": "BeamShift X", "var": None, "default": 0},
    {"name": "BeamShift Y", "var": None, "default": 0})
}

# seems to work well
cla1_mme = {**cla1_adat_base,
**{'hostapi': 0} }

   # strange timing issues, misses a lot
cla1_ds = {**cla1_adat_base,
**{'hostapi': 1} }

# WASAPI was designed for low latency and, works well
# supposedly the exclusive mode bypasses the windows mixer, which reduced latency
cla1_wasapi = {**cla1_adat_base,
**{'hostapi': 3,
   'exclusive': False} }

# Do not use, clock is unreliable?
# I get different values from stream.time and time.getBufferDacTime
# maybe a different clock is used?
cla1_wdm_ks = {
'device': 'ADAT 1 (1+2) (Fireface ADAT 1 (1+2))',
'hostapi': 4,  # 'Windows WDM-KS', Windows Driver Model - Kernel Streaming
'mapping': (1, 2),
'channels': (
    {"name": "BeamShift X", "var": None, "default": 0},
    {"name": "BeamShift Y", "var": None, "default": 0})
}

testing = {
'device': None,
'mapping': (1,2),
'channels': (
    {"name": "Channel 1", "var": None, "default": 0},
    {"name": "Channel 2", "var": None, "default": 0})
}

# default = testing
default = all_asio
# default = cla1_asio
# default = cla1_mme
# default = cla1_ds
default = cla1_wasapi
# default = cla1_wdm_ks

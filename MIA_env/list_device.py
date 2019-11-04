from tensorflow.python.client import device_lib
# from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
device_lib.list_local_devices()
print(get_available_gpus())

# from __future__ import absolute_import, division, print_function, unicode_literals

# import tensorflow    as tf
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
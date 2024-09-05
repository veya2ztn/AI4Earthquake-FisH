import h5py
import numpy as np
import sys

index_part = sys.argv[1]

index = np.load(f'datasets/STEAD/stead.accelerate/split/{index_part}.index.npy')
waveform = np.load(f'datasets/STEAD/stead.accelerate/split/{index_part}.npy')


### lets build the hdf5 file 
#### for each name in index, we will get the waveform vie method like handle.get('data/{name}')
#### now, build the handle
with h5py.File(f'datasets/STEAD/stead.accelerate/split/{index_part}.hdf5', 'w') as handle:
    data = handle.create_group('data')
    for name,wave in zip(index, waveform):
        data.create_dataset(name, data=wave)

# ### now, lets read the hdf5 file
# with h5py.File(f'datasets/STEAD/stead.accelerate/split/{index_part}.hdf5', 'r') as handle:
#     data = handle.get('data')
#     for name in data:
#         print(name, data[name][:5])
#         break
import os
import sys
import h5py
from tqdm import tqdm
import numpy as np


def read_event_data(event_name_sub_list):
    if len(event_name_sub_list) == 0:return []
    g_event = np.zeros((len(event_name_sub_list), 6000, 3))
    for i, event_name in enumerate(event_name_sub_list):
        g_event[i] = np.array(f.get(f'data/{event_name}'))
    return g_event


event_name_train = np.load('datasets/STEAD/BDLEELSSO/az_trace_name_train2.npy')
event_name_eval  = np.load('datasets/STEAD/BDLEELSSO/az_trace_name_test2.npy')
event_name = np.concatenate([event_name_train,event_name_eval])

print(f"totally event:{len(event_name)}")


dtfl = h5py.File("datasets/STEAD/stead.hdf5", 'r')
total_chunk = 100
index_range = np.linspace(0, len(event_name), total_chunk+1).astype('int')

if __name__ == '__main__':
    split = int(sys.argv[1])
    start = index_range[split]
    end   = index_range[split+1]
    target_path = f"datasets/STEAD/BDLEELSSO/Split/{start:08d}_{end:08d}.npy"
    if os.path.exists(target_path):exit()
    index_list = []
    waves_list = []
    for i in tqdm(range(start, end)):
        trace_name = event_name[i]
        waveform = np.array(dtfl.get(f'data/{trace_name}'))
        if len(waveform.shape) == 0:continue
        #print(waveform.shape)
        waves_list.append(waveform)
        index_list.append(trace_name)
    if len(index_list) > 0:
        waves_list = np.stack(waves_list)
        index_list = np.array(index_list)
        np.save(target_path, waves_list)
        np.save(target_path.replace('.npy', '.index.npy'), index_list)

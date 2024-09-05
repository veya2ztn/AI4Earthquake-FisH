"""
This is usually get failed by the calling h5py handle

"""
import obspy
import h5py
from obspy import UTCDateTime
import numpy as np
from obspy.clients.fdsn.client import Client
import matplotlib.pyplot as plt
from obspy import read_inventory
import pandas as pd
import h5py
from tqdm import tqdm
import multiprocessing as mp

def make_stream(dataset):
    '''
    input: hdf5 dataset
    output: obspy stream

    '''
    data = np.array(dataset)

    tr_E = obspy.Trace(data=data[:, 0])
    tr_E.stats.starttime = UTCDateTime(dataset.attrs['trace_start_time'])
    tr_E.stats.delta = 0.01
    tr_E.stats.channel = dataset.attrs['receiver_type']+'E'
    tr_E.stats.station = dataset.attrs['receiver_code']
    tr_E.stats.network = dataset.attrs['network_code']

    tr_N = obspy.Trace(data=data[:, 1])
    tr_N.stats.starttime = UTCDateTime(dataset.attrs['trace_start_time'])
    tr_N.stats.delta = 0.01
    tr_N.stats.channel = dataset.attrs['receiver_type']+'N'
    tr_N.stats.station = dataset.attrs['receiver_code']
    tr_N.stats.network = dataset.attrs['network_code']

    tr_Z = obspy.Trace(data=data[:, 2])
    tr_Z.stats.starttime = UTCDateTime(dataset.attrs['trace_start_time'])
    tr_Z.stats.delta = 0.01
    tr_Z.stats.channel = dataset.attrs['receiver_type']+'Z'
    tr_Z.stats.station = dataset.attrs['receiver_code']
    tr_Z.stats.network = dataset.attrs['network_code']

    stream = obspy.Stream([tr_E, tr_N, tr_Z])

    return stream




dtfl = h5py.File("datasets/STEAD/stead.hdf5", 'r')
def process_row(row):
    if not row['has_inventory']:
        return None
    
    dataset = dtfl.get(f'data/{row.trace_name}')
    st = make_stream(dataset)
    inventory = read_inventory(row['inventory_path'])
    try:
        st = st.remove_response(inventory=inventory, output="ACC", plot=False)
    except:
        return None
    return (np.stack([st[0].data, st[1].data, st[2].data], axis=-1), row.trace_name)


# read your metadata
metadata = pd.read_csv("datasets/STEAD/station.info.csv")
#metadata = metadata.iloc[0:1000]
# create a pool of workers
with mp.Pool(mp.cpu_count()) as pool:
    results = list(tqdm(pool.imap(process_row, [row for _, row in metadata.iterrows()]), total=len(metadata)))

# filter out None results and unpack waves and index
results = [result for result in results if result is not None]
waves_list, index_list = zip(*results)
waves_list = np.stack(waves_list)
index_list = np.array(index_list)

np.save("datasets/STEAD/waveform_accelerate.npy", waves_list)
np.save("datasets/STEAD/waveform_accelerate.index.npy", index_list)

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

def make_stream(dataset,starttime=None, receiver_type=None, receiver_code=None,network_code=None):
    '''
    input: hdf5 dataset
    output: obspy stream

    '''
    data = np.array(dataset).T
    frequency     = 0.01 # s
    starttime     = UTCDateTime(dataset.attrs.get('trace_start_time', starttime))
    receiver_type = dataset.attrs.get('receiver_type', receiver_type)
    receiver_code = dataset.attrs.get('receiver_code', receiver_code)
    network_code  = dataset.attrs.get('network_code', network_code)
    tr_E = obspy.Trace(data=data[:, 0])
    tr_E.stats.starttime = starttime
    tr_E.stats.delta   = frequency
    tr_E.stats.channel = receiver_type  +'E'
    tr_E.stats.station = receiver_code  
    tr_E.stats.network = network_code   

    tr_N = obspy.Trace(data=data[:, 1])
    tr_N.stats.starttime = starttime
    tr_N.stats.delta   = 0.01
    tr_N.stats.channel = receiver_type  +'N'
    tr_N.stats.station = receiver_code  
    tr_N.stats.network = network_code   

    tr_Z = obspy.Trace(data=data[:, 2])
    tr_Z.stats.starttime = starttime
    tr_Z.stats.delta   = 0.01
    tr_Z.stats.channel = receiver_type  +'Z'
    tr_Z.stats.station = receiver_code  
    tr_Z.stats.network = network_code   

    stream = obspy.Stream([tr_E, tr_N, tr_Z])

    return stream


import sys,os
def processing_one_trace_name(trace_name, trace_start_time):
    _,network_code, receiver_code, _ ,receiver_type = trace_name.split('.')
    inventory_path = f'/nvme/zhangtianning.di/datasets/Instance/Responses/{network_code}_{receiver_code}.xml'
    inventory = read_inventory(inventory_path)
    dataset = dtfl.get(f'data/{trace_name}') 
    #st = make_stream(dataset,starttime=trace_start_time,receiver_type=receiver_type,  receiver_code=receiver_code,network_code=network_code)
    # st= make_stream(dataset, starttime=trace_start_time,
    #             receiver_type=receiver_type,  
    #             receiver_code=inventory.networks[0].stations[0].code,
    #             network_code=inventory.networks[0].code )
    st = make_stream(dataset,starttime=trace_start_time,receiver_type=receiver_type,  receiver_code=receiver_code,network_code=network_code)
    _type = 'ACC'
    st = st.remove_response(inventory=inventory, output=_type, plot=False) 
    return 

if __name__ == '__main__':
    metadata = pd.read_csv("instance.clean.csv")
    
    index_part= int(sys.argv[1])
    num_parts = int(sys.argv[2])
    
    totally_paper_num = len(metadata)
    if totally_paper_num > 1:
        divided_nums = np.linspace(0, totally_paper_num , num_parts+1)
        divided_nums = [int(s) for s in divided_nums]
        start_index = divided_nums[index_part]
        end_index   = divided_nums[index_part + 1]
    else:
        start_index = 0
        end_index   = 1

    #metadata = metadata.iloc[start_index:end_index]
    
    dtfl        = h5py.File("Instance_events_counts.hdf5", 'r')
    target_path = f"waveform_accelerate/{start_index:08d}_{end_index:08d}.npy"
    if os.path.exists(target_path):exit()
    index_list = []
    waves_list = []
    for i in tqdm(range(start_index,end_index)):
        trace_name, trace_start_time = metadata.iloc[i]
        _,network_code, receiver_code, _ ,receiver_type = trace_name.split('.')
        inventory_path = f'/nvme/zhangtianning.di/datasets/Instance/Responses/{network_code}_{receiver_code}.xml'
        inventory = read_inventory(inventory_path)
        dataset = dtfl.get(f'data/{trace_name}') 
        # st= make_stream(dataset, starttime=trace_start_time, receiver_type=receiver_type, receiver_code=inventory.networks[0].stations[0].code,network_code=inventory.networks[0].code )
        st = make_stream(dataset,starttime=trace_start_time,receiver_type=receiver_type,  receiver_code=receiver_code,network_code=network_code)
        _type = 'ACC'
        try:
            st = st.remove_response(inventory=inventory, output=_type, plot=False) 
        except:
            continue
        waves_list.append(np.stack([st[0].data,st[1].data,st[2].data],axis=-1))
        index_list.append(trace_name)
    if len(index_list)>0:
        waves_list = np.stack(waves_list)
        index_list = np.array(index_list)
        np.save(target_path, waves_list)
        np.save(target_path.replace('.npy','.index.npy'), index_list)


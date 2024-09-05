from dataset.resource.load_resource import load_resource
from dataset.resource.resource_arguements import ResourceConfig, ResourceDiTing
from tqdm.auto import tqdm
from scipy.signal import stft
import numpy as np
import os
if __name__ == '__main__':
    import sys
    config = ResourceDiTing(resource_source='diting.group.full.good.hdf5')
    df, dtfl, index_map, normer, noise_engine = load_resource(config)

    partdf   = df[df['split']=='TRAIN']


    assert len(sys.argv)==3, "Usage: python create_cluster_for_each_station.py split_num split_idx"
    split_num = int(sys.argv[1])
    split_idx = int(sys.argv[2])
    assert split_idx < split_num   
    index_range = np.linspace(0, len(partdf), split_num+1).astype('int')
    split = split_idx
    start = index_range[split]
    end   = index_range[split+1]
    target_path = f"datasets/DiTing330km/STFT/Split/{start:08d}_{end:08d}.npy"
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    if os.path.exists(target_path):exit()
    length   = end - start
    PartData = np.zeros((length,65, 48, 3, 2))
    fs = 50
    window = 'hann'
    n  = 128
    for i in tqdm(range(length)):
        metadata = df.iloc[i]
        linedata = dtfl[metadata['trace_name']]
        total_length = 3000
        ahead_length = 500
        if metadata.p_arrival_sample<ahead_length:
            start_padding = ahead_length - metadata.p_arrival_sample
            linedata = np.pad(linedata, ((start_padding,0),(0,0)))

        linedata = linedata[metadata.p_arrival_sample-ahead_length:metadata.p_arrival_sample+total_length-ahead_length]
        if len(linedata) < total_length:
            end_padding = total_length - len(linedata)
            linedata = np.pad(linedata, ((0,end_padding),(0,0)))
        _, _, Z1  = stft(linedata[:,0], fs=fs, window=window, nperseg=n)
        _, _, Z2  = stft(linedata[:,1], fs=fs, window=window, nperseg=n)
        _, _, Z3  = stft(linedata[:,2], fs=fs, window=window, nperseg=n)
        PartData[i,:,:,0,0] = Z1.real
        PartData[i,:,:,1,0] = Z2.real
        PartData[i,:,:,2,0] = Z3.real
        PartData[i,:,:,0,1] = Z1.imag
        PartData[i,:,:,1,1] = Z2.imag
        PartData[i,:,:,2,1] = Z3.imag
    

    np.save(target_path, PartData)
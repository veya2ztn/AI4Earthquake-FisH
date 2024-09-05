import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
import os 
import tabulate
from concurrent.futures import ProcessPoolExecutor
#from train_utils import show_df_in_table
from .resource_arguements import ResourceConfig
from typing import Optional, Union, List, Tuple

class ResourceLoader:
    """
        Load original data from the path.
        There is no more data modification in the follow programming.
        Numpy Version:
            1. load the numpy data from the path
            2. load the metadata from the path
            3. filter the metadata and waveform data
            4. process the metadata and waveform data
        HDF5 Version:
            TODO:
        Make share 
            - each processing has a check interface that skip when the data is load from a offline phase.
            - the offline data can be generate by the online processing code.
    """
    @staticmethod   
    def get_must_need_metadata_keys(config):
        return config.metadata_keys


    def check_metadata_item(self, df):
        assert 'trace_name' in df, "the trace_name is not in the metadata"
        for key in self.get_must_need_metadata_keys(self.config):
            assert key in df, f"the {key} is not in the metadata"
    waveform_order = None
    
    
    def __init__(self, config:ResourceConfig):
        self.config   = config
    
    def make_offline_csv(self, df):
        """
        make the offline csv file for the later processing
        """
        raise NotImplementedError
    
    def align_direction_of_waveform(self, waveformdata, target_direction='ENZ'):
        ### firstly align the waveform 
        assert self.waveform_order is not None, "the loader must have inner waveform order"
        aligned_order = [self.waveform_order.index(s) for s in target_direction]
        print(f"reshape channel order from {self.waveform_order} to {target_direction}")
        waveformdata = waveformdata[..., aligned_order]
        return waveformdata
    
    def get_the_numpy_waveform_name(self):
        if self.config.waveform_path is None:
            return f'{self.config.dataset_name}/{self.config.waveform_type}'
        else:
            return self.config.waveform_path


    def snap_the_file(self,data_root_dir, name, endwwith='.npy')->Union[str, List[str]]:
        """
        firstly, check the 
        """
        print(data_root_dir, name)
        data_path = os.path.join(data_root_dir, name)
        if os.path.isdir(data_path):
            assert endwwith == '.npy'
            return [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.npy') and '.index' not in f]
        else:
            data_path = self.config.waveform_path
            assert os.path.isfile(data_path), f"the file {data_path} is not exist"
            return data_path
            # data_path = data_path+endwwith
            # assert os.path.isfile(data_path), f"the file {data_path} is not exist"
            # return data_path

    def load_waveform_from_numpy(self) -> (np.ndarray, np.ndarray):
        #assert wave_file is not None
        wave_file = self.snap_the_file(self.config.data_root_dir, self.get_the_numpy_waveform_name(), endwwith='.npy')
        print(f'start loading data {wave_file} ................')
        waveform_name, waveform_data = load_numpy_data(wave_file)
        return waveform_name, waveform_data
    
    def filter_metadata(self, df):
        raise NotImplementedError("you must implement this function")   
        
    
    def load_metadata_from_csv(self) -> np.ndarray:
        if self.config.generate_offline_metadata is None:
            df = pd.read_csv(self.config.offline_metadata_path)
        else:
            df,output_file = self.make_offline_csv()
            print(f'totol {len(df)} trace read from csv file:{output_file} ')
        self.check_metadata_item(df)
        df = self.filter_metadata(df)
        print(f'filted metadata and get {len(df)} trace in later processing')
        return df 
    
    def filter_trace(self, waveform_data, waveform_name, metadata):
        """
        filter the waveform data and waveform name according to the metadata
        """
        old_len = len(metadata)
        metadata = metadata[metadata['trace_name'].isin(list(waveform_name))]
        new_len = len(metadata)
        waveform_data = waveform_data[np.intersect1d(waveform_name, metadata['trace_name'].values,True,True)[1]]
        old_wavelen = len(waveform_name)
        waveform_name = waveform_name[np.intersect1d(waveform_name, metadata['trace_name'].values,True,True)[1]]
        new_wavelen = len(waveform_name)
        if old_len != new_len:
            print(f"""
                  WARNING: the metadata and waveform_data is not match, we firstly filter the metadata from {old_len} to {new_len}.
                  There are {old_len - new_len} trace is not used in the metadata.
                  Then we filte the waveform_data from {old_wavelen} to {new_wavelen}.
                  There are {old_wavelen - new_wavelen} trace is not used in the waveform.
                  """)
        print(f"we totally get {len(metadata)} trace in the dataset")
        return waveform_data, waveform_name, metadata
    
            
    def load_numpy_pair(self):
        metadata    = self.load_metadata_from_csv()
        waveform_name, waveform_data = self.load_waveform_from_numpy()
        waveform_data, waveform_name, metadata = self.filter_trace(waveform_data, waveform_name, metadata)
        data = {'waveforms'    : waveform_data,
                'waveform_name': waveform_name}
        event_set_list = self.get_event_list(metadata)
        inputs_metadata, source_metadata = self.process_metadata(metadata)
        inputs_waveform = self.process_waveform(data, metadata)
        return inputs_waveform, inputs_metadata, source_metadata, event_set_list

    def load_hdf5_pair(self):
        metadata    = self.load_metadata_from_csv()
        waveform_name, waveform_data = self.load_hdf5_data(metadata)
        data = {'waveforms'    : waveform_data,
                'waveform_name': waveform_name}
        event_set_list = self.get_event_list(metadata)
        inputs_metadata, source_metadata = self.process_metadata(metadata)
        inputs_waveform = self.process_waveform(data, metadata)
        return inputs_waveform, inputs_metadata, source_metadata, event_set_list
    
    def load_hdf5_data(self,metadata):
        raise NotImplementedError("you must implement this function")

    def get_event_list(self, metadata):
        metadata['index'] = np.arange(len(metadata))
        event_set_list = metadata.groupby('source_id')['index'].apply(lambda x: x.to_numpy()).tolist()
        return event_set_list 

    def process_metadata(self, metadata):
        raise NotImplementedError("you must implement this function")
    def process_waveform(self, waveform_data,metadata):
        raise NotImplementedError("you must implement this function")
        
def load_numpy_data(waveformdata_path,numpy_dtypye='float16'):
    if isinstance(waveformdata_path, str):
        waveform_name_path = waveformdata_path.replace(
            'f16.', '').replace('.npy', '.index.npy')
        waveform_name = np.load(waveform_name_path, allow_pickle=True)
        waveformdata = np.load(waveformdata_path)
    else:
        assert isinstance(waveformdata_path, list)
        waveform_name = []
        waveformdata = []
        if numpy_dtypye == 'float16':
            waveformdata_path = [f for f in waveformdata_path if '.f16' in f]
        else:
            waveformdata_path = [f for f in waveformdata_path if '.f16' not in f]
        print(waveformdata_path)
        for data_path in tqdm(waveformdata_path):
            name_path = data_path.replace(
                'f16.', '').replace('.npy', '.index.npy')
            waveform_name.append(np.load(name_path, allow_pickle=True))
            waveformdata.append(np.load(data_path))
        waveform_name = np.concatenate(waveform_name)
        waveformdata = np.concatenate(waveformdata)

    return waveform_name, waveformdata


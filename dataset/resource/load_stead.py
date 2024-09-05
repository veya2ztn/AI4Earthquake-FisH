from .resource_arguements import ResourceConfig
from .base import ResourceLoader, load_numpy_data
import numpy as np
from typing import Optional, Union, List, Tuple
import os
import pandas as pd
try:
    from pyproj import Geod, Transformer
except:
    pass
import json
import torch
import random
import tabulate
import h5py
from tqdm import tqdm
import multiprocessing as mp
from functools import partial





class STEAD_Loader(ResourceLoader):
    origin_waveform_order = 'ENZ'
    sampling_rate = 100
    
    def __init__(self, config:ResourceConfig):
        self.config   = config
        specified_dataset_path = self.config.data_root_dir+'STEAD/stead'
        specified_dataset_path = specified_dataset_path + '_'+self.config.trace_category
        specified_dataset_path = specified_dataset_path + '_'+self.config.numpy_dtypye
        specified_dataset_path = specified_dataset_path + '_'+self.config.waveform_type
        if self.config.delta_xyz:
            specified_dataset_path = specified_dataset_path+'_delta_xyz'
        if self.config.align_time:
            specified_dataset_path = specified_dataset_path + '_align/'
        self.specified_dataset_path = specified_dataset_path

        
    
    def make_dxdydz_csv(self, df):
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:4978")

        receiver_x, receiver_y, receiver_z = transformer.transform(df['receiver_latitude'],
                                                                df['receiver_longitude'],
                                                                df["receiver_elevation_m"])
        source_x,   source_y,  source_z   = transformer.transform( df['source_latitude'],
                                                                df['source_longitude'],
                                                                -df["source_depth_km"]*1000,
                                                                )
        receiver_x /= 1000
        receiver_y /= 1000
        receiver_z /= 1000
        source_x   /= 1000
        source_y   /= 1000
        source_z   /= 1000
        df['receiver_x_km'] = receiver_x
        df['receiver_y_km'] = receiver_y
        df['receiver_z_km'] = receiver_z
        df['source_x_km']   = source_x
        df['source_y_km']   = source_y
        df['source_z_km']   = source_z
        df['delta_vector_x']= source_x - receiver_x
        df['delta_vector_y']= source_y - receiver_y
        df['delta_vector_z']= source_z - receiver_z
        return df
    
    def make_align_csv(self, df):
        df['group_size'] = df.groupby('source_id')['source_id'].transform('count')
        df['trace_start_timestamp'] = pd.to_datetime(df['trace_start_time']).apply(lambda x: x.timestamp()) 
        df['p_arrival_s_before']  = df['p_arrival_sample']/100 - 2
        grouped_df = df.groupby("source_id")
        the_event_start_recorder_timestamp = grouped_df[['trace_start_timestamp','p_arrival_s_before']].apply(lambda x: (x['trace_start_timestamp'] + x['p_arrival_s_before']).min())

        source_id_2_event_recorder_start_timestamp = the_event_start_recorder_timestamp.to_dict()
        offset_for_event_recorder= []
        for trace_recorder_start_timestamp, source_id in df[['trace_start_timestamp','source_id']].values:
            event_recorder_start_timestamp = source_id_2_event_recorder_start_timestamp[source_id]
            delay_timestamp = event_recorder_start_timestamp - trace_recorder_start_timestamp
            delay_offset    = round(delay_timestamp*100) # the sample rate is 100HZ
            offset_for_event_recorder.append(delay_offset)

        df['offset_for_event_recorder']    = offset_for_event_recorder
        ## make sure that there are 20s record about the p wave
        #df = df[df['offset_for_event_recorder']+3000 > df['p_arrival_sample']+2000]

        df['event_start_record_timestamp'] = the_event_start_recorder_timestamp

        df['the_event_start_recorder_time'] = pd.to_datetime(the_event_start_recorder_timestamp,unit='s')
        df['trace_start_time'] = pd.to_datetime(pd.to_datetime(df['trace_start_time']),unit='s')
        #I think do no change the origin value is better, but now we change it
        #df['p_arrival_sample'] = df['p_arrival_sample'] - df['offset_for_event_recorder']
        df['p_arrival_sample_align'] = df['p_arrival_sample'] - df['offset_for_event_recorder']
        return df
    
    def make_offline_csv(self):
        """
        make the offline csv file for the later processing
        """
        #csv_file = self.snap_the_file(self.config.data_root_dir, self.get_the_numpy_waveform_name(), endwwith='.csv')
        csv_file = self.config.data_root_dir+'/STEAD/STEAD.csv'
        df = pd.read_csv(csv_file)
        print(f'total trace in csv file: {len(df)}')
        if self.config.delta_xyz:
            df = self.make_dxdydz_csv(df)
        df = df[df['source_latitude'].notnull() & df['source_longitude'].notnull() & df['source_depth_km'].notnull() & df['source_magnitude']>0]
        if self.config.align_time:
            df = self.make_align_csv(df)
        keys = df.keys()
        df = df[[
            "p_arrival_sample",
            "receiver_elevation_m",
            "receiver_latitude",
            "receiver_longitude",
            "receiver_code",
            "trace_name",
            "source_id",
            'source_magnitude',
            "source_origin_time",
            "source_latitude",
            "source_longitude",
            "source_depth_km",
            'delta_vector_x',
            'delta_vector_y',
            'delta_vector_z',
            'receiver_x_km',
            'receiver_y_km',
            'receiver_z_km',
            'source_x_km',
            'source_y_km',
            'source_z_km',
            'p_arrival_sample_align',
            'offset_for_event_recorder',
            'split'
        ]].copy()
        output_file = 'stead'
        output_file = output_file + '_'+self.config.trace_category
        output_file = output_file + '_'+self.config.numpy_dtypye
        output_file = output_file + '_'+self.config.waveform_type
        if self.config.delta_xyz:
            output_file = output_file+'_delta_xyz'
        if self.config.align_time:
            output_file = output_file + '_align'
        output_file = output_file + '.csv'
        os.mkdir(self.specified_dataset_path)
        output_file = os.path.join(self.specified_dataset_path, output_file)  
        self.output_file = output_file
        df.to_csv(output_file, index=True)
        return df, output_file
        
    def load_metadata_from_csv(self) -> np.ndarray:
            return super().load_metadata_from_csv()
    
    def filter_metadata(self, df):
        ## filter the dataframe along the given condition in config
        if 'onlyevent' in self.config.trace_category:
            if  'trace_category'  in df:
                df = df[df['trace_category'] == 'earthquake_local']
            else:
                print("the trace_category is not in the metadata, we will skip this filter")
        for flag, key in [['lat','source_latitude'],
                          ['lon','source_longitude'],
                          ['dep','source_depth_km']]:
            if flag in self.config.trace_category:
                df = df[df[key].notnull()]
            else:
                print("the source_latitude is not in the metadata, we will skip this filter")
        if self.config.station_channel is None:
            df = df
        else:
            print(f'we use channel {self.config.station_channel} to filter the metadata')
            df = df[df['receiver_type'].isin(self.config.station_channel)] 
        if self.config.split_file is None:
            df = df[df['split'] == self.config.split]
        else:
            trace_name_list = np.load(self.config.split_file, allow_pickle=True)
            df = df[df['trace_name'].isin(trace_name_list)]
        df['group_size'] = df.groupby('source_id')['source_id'].transform('count')
        if 'multigroup' in self.config.trace_category:
            df = df[df['group_size'] > 1]

        
        keys = df.keys()
        must_keys = self.get_must_need_metadata_keys(self.config)
        # for key in keys:
        #     if 'Unnamed' in key:df.drop(key, axis=1, inplace=True)
        #     if key not in must_keys:
        #         df.drop(key, axis=1, inplace=True)
        return df
    
    def load_hdf5_data(self,metadata):
        waveform_name = metadata['trace_name'].values
        waveform_data = h5py.File(self.config.data_root_dir+'STEAD/stead.hdf5', 'r')
        return waveform_name, waveform_data
        
    def process_metadata(self, metadata):
        target_type = self.config.target_type
        ### source metadata
        df = metadata
        grouped_df = df.groupby("source_id")
        print(f"there are {len(grouped_df)} events in the data")
        group_size_counts = df['group_size'].value_counts().sort_index()
        # Filter the counts for group sizes from 1 to 9
        filtered_counts = group_size_counts[group_size_counts.index.isin(range(1, 20))]

        # Convert the filtered counts to a pandas DataFrame
        group_size_table = pd.DataFrame(filtered_counts).reset_index()
        group_size_table.columns = ['group_size', 'count']
        print(tabulate.tabulate(group_size_table.transpose(),headers='', tablefmt='psql', showindex=True))

        print(f"===> Notification: the old trail use wrong location information as [lat, lat , deep] rather than [lat, lon, deep]. However, the performance is good. <===")
        source_magnitude = np.array(grouped_df['source_magnitude'].apply(lambda x: x.to_numpy().mean()).tolist())
        if target_type == 'all_in_one_source':
            source_latitude  = np.array(grouped_df['source_latitude'].apply(lambda x: x.to_numpy().mean()).tolist())
            source_longitude = np.array(grouped_df['source_longitude'].apply(lambda x: x.to_numpy().mean()).tolist())
            source_depth_km  = np.array(grouped_df['source_depth_km'].apply(lambda x: x.to_numpy().mean()).tolist())
            source_metadata  = {'location': np.stack([source_latitude, source_longitude, source_depth_km],-1),
                                'magnitude': source_magnitude}
            inputs_metadata = {
                            "receiver_latitude" : metadata['receiver_latitude'].values,
                            "receiver_longitude": metadata['receiver_longitude'].values,
                            "receiver_elevation_m": metadata['receiver_elevation_m'].values
                        }
        elif target_type == 'all_in_all_delta':
            if np.abs(metadata['coords_z'].values).max() > 1000:
                print("========== [ will divide coords_z by 1000 since we hope its unit is km rather than meter  ] ==========")
                metadata['coords_z'] = metadata['coords_z']/1000
            source_metadata = {'delta_latitude' : df["delta_latitude"].values,
                            'delta_longitude': df["delta_longitude"].values,
                            'delta_deepth'   : df['delta_deepth'].values,# (delta is define as deepth - receiver_elevation_m not deepth + receiver_elevation_m)
                            'magnitude': source_magnitude}
            inputs_metadata = {
                            "receiver_latitude" : metadata['receiver_latitude'].values,
                            "receiver_longitude": metadata['receiver_longitude'].values,
                            "receiver_elevation_m": metadata['receiver_elevation_m'].values
                        }
        elif target_type == 'all_in_all_xyz_delta':
            source_metadata = {'delta_vector_x': df['delta_vector_x'].values,
                            'delta_vector_y': df['delta_vector_y'].values,
                            'delta_vector_z': df['delta_vector_z'].values,# (delta is define as deepth - receiver_z_km )
                            'magnitude': source_magnitude}
            inputs_metadata = {
                    "receiver_vector_x": metadata['receiver_x_km'].values,
                    "receiver_vector_y": metadata['receiver_y_km'].values,
                    "receiver_vector_z": metadata['receiver_z_km'].values,# (receiver_z_km is define as -receiver_elevation_m/1000 )
                }
        else:
            raise NotImplementedError(f"target_type={target_type} is not support")
        inputs_metadata["p_arrival_sample_align"] = metadata['p_arrival_sample_align'].values
        if 'offset_for_event_recorder' in metadata:
            inputs_metadata["offset_for_event_recorder"] = metadata['offset_for_event_recorder'].values
            
        #source_metadata  = np.stack([source_latitude, source_longitude, source_depth_km, source_magnitude], -1)
        #inputs_metadata = one_event_metadata[['coords_x', 'coords_y', 'coords_z', 'p_picks']].values
        # <<---- waveform is 6000 and divide to 3000
        # print("========== [ notice, some dataset need let p_pick divide by 2, if the performance not good, check this ] ==========")
        # if inputs_metadata['p_picks'].max() > 3000 and 'STEAD' in data_path:
        #     print("========== [ will divide p_pick by 2, this mean we use sample rate == 50Hz] ==========")
        #     print("========== [ To provide cross-dataset performance, we disable this option after 20230725] ==========")
        #     inputs_metadata['p_picks'] = inputs_metadata['p_picks']//2
        #     raise NotImplementedError(f"should not make p_picks large than 3000") ## The STEAD dataset p_arrive never bigger than 3000
        metadata['index'] = np.arange(len(metadata))  
        return inputs_metadata, source_metadata
    
    def process_waveform(self, waveform_data, metadata):
        trace_name_list = metadata['trace_name'].values      
        if isinstance(waveform_data, h5py._hl.files.File):
            inputs_waveform = (waveform_data, trace_name_list)
        else:
            waveform_data['all_trace_name_list'] = trace_name_list
            inputs_waveform = waveform_data
        return inputs_waveform
    
    def save_config(self):
        config_path = self.config.data_root_dir+'STEAD/stead'
        config_path = 'stead'
        config_path = config_path + '_'+self.config.trace_category
        config_path = config_path + '_'+self.config.numpy_dtypye
        config_path= config_path + '_'+self.config.waveform_type
        if self.config.delta_xyz:
            config_path = config_path+'_delta_xyz'
        if self.config.align_time:
            config_path = config_path + '_align'
        config_path = config_path + '.json'
        csv_path = config_path.replace('.json','.csv')
        config_path = os.path.join(self.specified_dataset_path, config_path)  
        csv_path = os.path.join(self.specified_dataset_path, csv_path)
        newconfig = self.config
        newconfig.generate_offline_metadata = None
        newconfig.offline_metadata_path = csv_path
        newconfig.save_json(config_path)
        
    
    def get_data(self):
        if self.config.data_type == 'numpy':
            return self.load_numpy_pair()
        elif self.config.data_type == 'hdf5':
            return self.load_hdf5_pair()
        else:
            raise NotImplementedError("the data_type must be numpy or hdf5")


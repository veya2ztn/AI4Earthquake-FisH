#from mltool.dataaccelerate import sendall2gpu
import torch
import numpy as np
import pandas as pd
from functools import lru_cache
from scipy.stats import gaussian_kde
from scipy.optimize import root
#from mltool.visualization import *
from scipy.optimize import minimize
import tabulate
from concurrent.futures import ProcessPoolExecutor
import os
from .dataset_arguements import MultistationDatasetConfig

class Stationed_PreNormlized_Dataset(torch.utils.data.Dataset):

    
    dataloader_mode = False
    name_idx_map = None
    
    def __init__(self, inputs_waveform,
                 inputs_metadata,
                 source_metadata,
                 event_set_list,
                 config:MultistationDatasetConfig            
                 ):
        self.config = config
        self.inputs_waveform = inputs_waveform
        self.inputs_metadata = inputs_metadata
        self.source_metadata = source_metadata
        self.event_set_list = event_set_list
        self.data_augmentation_config = {
            'oversample': self.config.oversample,
            'magnitude_resampling': self.config.magnitude_resampling,
            'min_upsample_magnitude': self.config.min_upsample_magnitude,
            'max_upsample_magnitude': self.config.max_upsample_magnitude,
            'upsample_high_station_events': self.config.upsample_high_station_events,

        }
        base_indexes, indexes = self.initialize_indexes_by_data_augmentation(
            self.data_augmentation_config)
        indexes = self.shuffle_indexes()
        self.adjust_mean = self.config.adjust_mean
        self.trigger_based = self.config.trigger_based
        self.fake_borehole = self.config.fake_borehole
        self.shuffle = self.config.shuffle
        self.windowlen = self.config.windowlen
        self.max_stations = self.config.max_stations
        self.shard_size = self.config.shard_size
        self.length = int(np.ceil(self.indexes.shape[0] / self.shard_size))
        self.cutout = self.config.cutout
        

    @property
    def p_arrive_time(self):
        return self.inputs_metadata['p_arrival_sample_align']

    @property
    def offset_event_start_time(self):
        if 'offset_for_event_recorder' in self.inputs_metadata:
            return self.inputs_metadata['offset_for_event_recorder']
        else:
            return None
        
        
    #the following methods are not used in the current version
    #*********************************************************************************************************************
    #*********************************************************************************************************************
    #*********************************************************************************************************************
    #*********************************************************************************************************************
    #*********************************************************************************************************************
    #*********************************************************************************************************************
    def apply_cutting_off(self, waveforms):
        self.adjust_mean = self.config.adjust_mean
        windowlen = self.config.window_length
        trigger_based = self.config.windowlen
        if not self.cutout:
            return waveforms.shape[2], waveforms

        if self.sliding_window:
            window_end = np.random.randint(max(windowlen, self.cutout[0]), min(
                waveforms.shape[2], self.cutout[1]) + 1)
            waveforms = waveforms[:, :, window_end - windowlen: window_end]
            cutout = window_end
            if self.adjust_mean:
                waveforms -= np.mean(waveforms, axis=2, keepdims=True)
        else:
            cutout = np.random.randint(*self.cutout)  # apply random cutout off
            if self.adjust_mean:
                waveforms -= np.mean(waveforms[:, :,
                                     :cutout+1], axis=2, keepdims=True)
            waveforms[:, :, cutout:] = 0
        return cutout, waveforms
    # this method is called but do nothing
    def apply_integrate(self, waveforms):
        integrate = self.config.integrate
        if integrate:
            raise NotImplementedError("integrate is not allowed ")
            waveforms = np.cumsum(waveforms, axis=2) / \
                integrate['sampling_rate']
        return waveforms
    

    def apply_label_smoothing(self, magnitude):
        if self.config.label_smoothing:
            raise NotImplementedError("label smoothing is not allowed")
        else:
            return magnitude

    def apply_station_blinding(self):
        # , "station_blinding is not allowed, should be done at the data picking stage"
        raise NotImplementedError

    def filte_stations_without_P_arriving(self, cutout, p_arrival_sample_align, waveforms, location):
        if self.trigger_based:
            # Remove waveforms for all stations that did not trigger yet to avoid knowledge leakage
            # Ensure that stations without P picks do not show data
            # This proceduce will make the data much more sparse
            # This proceduce should be done for the waveform distillation, and should not be set here.
            p_arrival_sample_align[p_arrival_sample_align <= 0] = waveforms.shape[2]  # 3000
            waveforms[cutout < p_arrival_sample_align, :, :] = 0
            location[cutout < p_arrival_sample_align, :, :] = 0
        return waveforms, location


    def apply_fake_borehole(self, waveforms, metadata):
        if not (self.fake_borehole and waveforms.shape[3] == 3):
            return waveforms, metadata
        raise NotImplementedError("fake_borehole is not allowed")
    #*********************************************************************************************************************
    #*********************************************************************************************************************
    #*********************************************************************************************************************
    #*********************************************************************************************************************
    #*********************************************************************************************************************
    #*********************************************************************************************************************
    #the end of methodst that are not used in the current version
    
    def get_data(self, indexes):
        #self.timer('pickup_stations').start()
        rows_to_picks = [self.pickup_stations(
            self.event_set_list[idx]) for idx in indexes]
        # (B, 4)
        #self.timer('pickup_stations').stop()
        #self.timer('phase1').start()
        cutouts = [cutout for cutout, rows in rows_to_picks if len(rows) > 0]
        rows_to_pick = [rows for cutout,
                        rows in rows_to_picks if len(rows) > 0]
        if len(rows_to_pick) == 0:
            return None

        outputs = self.get_target_pool(indexes, rows_to_picks)
        #self.timer('mask_pick').start()
        picked_inputs_waveform, picked_inputs_metadata = self.get_inputs_from_slicing(
            rows_to_pick)  # get padded data

        # picked_inputs_waveform2 = np.where(index_array[..., None, None] >= 0, self.inputs_waveform[index_array], 0)  # (B,L,3000,3)
        # print(f"diff ===> {((picked_inputs_waveform2-picked_inputs_waveform)**2).mean()}")
        #picked_inputs_metadata = np.where(index_array[..., None] >= 0, self.inputs_metadata[index_array], 0)  # (B,L,4)
        #self.timer('mask_pick').stop()

        #print(picked_inputs_waveform.shape)
        #print(picked_inputs_metadata.shape)

        #self.timer('cutout').start()
        for k, v in picked_inputs_metadata.items():
            assert len(v) == len(picked_inputs_waveform), \
                f"the picked_inputs_metadata shape is {[(k,v.shape) for k,v in picked_inputs_metadata.items()]} it should be same as waveform size {len(picked_inputs_waveform)}"
        #wavemask = np.full((len(rows_to_pick), max_rows,self.inputs_waveform.shape[-2]), True)
        for i, cutout in enumerate(cutouts):
            picked_inputs_waveform[i, :, cutout:] = 0
        #picked_inputs_waveform[wavemask] = 0
        #self.timer('cutout').stop()

        #self.timer('rest').start()
        #picked_station_location = picked_inputs_metadata[..., 0:3]  # (B,L,3)
        #picked_station_p_arrive = picked_inputs_metadata[...,  3]  # (B,L)
        picked_station_location = picked_inputs_metadata

        ### random mask tail sequence (should make sure the P arriving information contain in waveform)
        #picked_inputs_waveform = self.apply_cutting_off(picked_inputs_waveform) #<---is moved into pickup_stations

        ### if cutout used, then need exclude those wave that wont have P-arrive information
        #picked_inputs_waveform, picked_station_location = self.filte_stations_without_P_arriving(cutout,picked_station_p_arrive,
        #                                                                                    picked_inputs_waveform,
        #                                                                                    picked_station_location)
        # #<---is moved into pickup_stations

        ### intergrate waveform
        picked_inputs_waveform = self.apply_integrate(picked_inputs_waveform)

        ### channel manupilate
        #picked_inputs_waveform, picked_station_location = self.apply_fake_borehole(picked_inputs_waveform, picked_station_location)

        ### data clear: the metadata shape like (B, station, 3) but as the coding shows, it is a two side padding,
        ### which is not good for pytorch
        #### lets convert the two-side padding into one side padding.

        ## ---------- clear totally zeros data
        #mask = (picked_station_location!=0).any(2).any(1) #(B, L , 3)
        #picked_inputs_waveform  = picked_inputs_waveform[mask]
        #picked_station_location = picked_station_location[mask]
        #magnitude               = magnitude[mask]
        #location                = location[mask]
        ### ---------- move station rows so that they are one side padding
        #good_mask    = (picked_station_location!=0).any(2)
        #set_length   = good_mask.sum(1)
        #index_array = np.full((len(set_length), picked_station_location.shape[1]),False)
        #for i, row_length in enumerate(set_length):index_array[i, :row_length] = True
        #new_picked_inputs_waveform = np.zeros_like(picked_inputs_waveform)
        #new_picked_inputs_waveform[index_array] = picked_inputs_waveform[good_mask]
        #picked_station_location = np.zeros_like(picked_station_location)
        #picked_station_location[index_array] = picked_station_location[good_mask]

        ### data collect
        waveforms = picked_inputs_waveform
        metadata = picked_station_location

        if self.shard_size == 1 or self.dataloader_mode:
            ## lets padding the data
            waveforms = np.pad(waveforms, ((
                0, 0), (0, self.max_stations - waveforms.shape[1]), (0, 0), (0, 0)))  # (B,L,3000,3)
            metadata = dict([(k, np.pad(v, ((0, 0), (0, self.max_stations - v.shape[1]))))
                            for k, v in metadata.items()])   # (B,L,1)
            for k in outputs.keys():
                if k not in ['location', 'magnitude']:
                    v = outputs[k]
                    outputs[k] = np.pad(
                        v, ((0, 0), (0, self.max_stations - v.shape[1])))
        waveforms = waveforms[0] if self.shard_size == 1 else waveforms
        metadata = dict([(k, v[0]) for k, v in metadata.items()]
                        ) if self.shard_size == 1 else metadata
        inputs = {"waveform_inp": waveforms, "metadata_inp": metadata}
        
        #self.timer('rest').stop()
        #self.timer.log(['pickup_stations', 'phase1','create_index_array','mask_pick','cutout','rest'], normalizer=10)
        
        return inputs, outputs
    
    def get_inputs_from_slicing(self, rows_to_pick):
        #self.timer('create_index_array').start()
        #self.timer('create_index_array').stop()
        # if isinstance(self.inputs_waveform, np.ndarray):
        #     assert self.offset_event_start_time is None, f"not finish for given offset_event_start_time"
        #     max_rows = max([len(row_set) for row_set in rows_to_pick])
        #     index_array = np.full((len(rows_to_pick), max_rows), -1)
        #     for i, row_set in enumerate(rows_to_pick):
        #         index_array[i, :len(row_set)] = row_set
        #     picked_inputs_waveform = self.inputs_waveform[index_array]
        #     picked_inputs_waveform[index_array < 0] = 0
        #     picked_inputs_metadata = {}
        #     for k, v in self.inputs_metadata.items():
        #         data = v[index_array]
        #         data[index_array < 0] = 0
        #         picked_inputs_metadata[k] = data
        #     return picked_inputs_waveform, picked_inputs_metadata
        if 'h5' in str(type(self.inputs_waveform['waveforms'])):
            assert self.offset_event_start_time is not None, f"not finish for given offset_event_start_time"
            inputs_waveform = self.inputs_waveform['waveforms']
            waveform_name   = self.inputs_waveform['waveform_name']
            trace_name_all  = self.inputs_waveform['all_trace_name_list']
            self.name_idx_map = dict([(name, idx) for idx, name in enumerate(waveform_name)]) if self.name_idx_map is None else self.name_idx_map
            max_rows        = max([len(row_set) for row_set in rows_to_pick])
            # rows_to_pick = [group1_row_set, group2_row_set, ..., groupn_row_set]
            index_array = None
            ### compute the ids to pickup waveform
            assert self.offset_event_start_time is not None, f"you must given offset_event_start_time"
            #     raise NotImplementedError("not finish for given offset_event_start_time")
            #     # index_array = np.full((len(rows_to_pick), max_rows), -1)
            #     # for i, row_set in enumerate(rows_to_pick):
            #     #     select_wave_id = np.array([self.name_idx_map[name] for name in trace_name_all[row_set]])
            #     #     index_array[i, :len(row_set)] = select_wave_id #(B, max_rows)
            #     # picked_inputs_waveform = inputs_waveform[index_array] #(B, max_rows, 3000, 3)
            #     # picked_inputs_waveform[index_array < 0] = 0
            # else:
            #raise
            # above picked_inputs_waveform is collected into an uniformed array
            # if self.offset_event_start_time is not None, this mean it is not an aligned array
            # this mean we need shift the wave (D=3000) by self.offset_event_start_time, 
            # each (B, max_rows) has different offset, which may lose the speed advantage
            # One possible way is createa a copy and do index picking again
            picked_inputs_waveform = []
            for i, row_set in enumerate(rows_to_pick):
                waveform_each_batch = []
                offsets     = self.offset_event_start_time[row_set]
                trace_names = trace_name_all[row_set]
                for name, offset in zip(trace_names,offsets):
                    waveform_now = inputs_waveform.get(f'data/{name}')
                    #(3000,3) --> (3000,3) # offset is calculated by (should_start - now_start) 
                    # if the offset is negitive, then means the waveform should start earlier then now recording, thus we use -offset for np.roll
                    
                    waveform_now = np.roll(waveform_now, -offset,axis=0) 
                    if offset < 0: waveform_now[:-offset] = 0
                    else: waveform_now[-offset:] = 0
                    waveform_each_batch.append(waveform_now) 
                waveform_each_batch = np.stack(waveform_each_batch) # (L1,3000,3)
                waveform_each_batch = np.pad(waveform_each_batch,((0,max_rows-len(waveform_each_batch)),(0,0),(0,0)))
                picked_inputs_waveform.append(waveform_each_batch)
            picked_inputs_waveform = np.stack(picked_inputs_waveform)

            index_array = np.full((len(rows_to_pick), max_rows), -1) if index_array is None else index_array
            for i, row_set in enumerate(rows_to_pick):
                index_array[i, :len(row_set)] = row_set # overwrite
            picked_inputs_metadata = {}
            for k, v in self.inputs_metadata.items():
                data = v[torch.LongTensor(index_array)]
                if isinstance(data,np.ndarray):data[index_array < 0] = 0
                picked_inputs_metadata[k] = data
            # picked_inputs_metadata = self.inputs_metadata[torch.LongTensor(index_array)]
            # picked_inputs_metadata[index_array < 0] = 0
            return picked_inputs_waveform, picked_inputs_metadata
            
            # all_trace_picked = np.concatenate(rows_to_pick)  # (BL,)
            # event_names = self.inputs_waveform[1][all_trace_picked]
            # #num_process = 20
            # #with ProcessPoolExecutor(max_workers=num_process) as executor:
            # picked_inputs_waveform = [np.array(self.inputs_waveform[0].get(
            #     f'data/{event_name}'))[..., :, :] for event_name in event_names]  # is a list of BL x (6000, 3)
            # #picked_inputs_waveform_flatten= np.concatenate(picked_inputs_waveform) # is a array (BL,)

            # max_rows = max([len(row_set) for row_set in rows_to_pick])
            # picked_inputs_waveform_pad = np.zeros(
            #     (len(rows_to_pick), max_rows, *picked_inputs_waveform[0].shape[-2:]))
            # picked_inputs_metadata_pad = np.zeros(
            #     (len(rows_to_pick), max_rows, 4))
            # # raise NotImplementedError(
            # #     "self.inputs_metadata is no longer a numpy but a dict not")
            # # is a array (BL,)
            # picked_inputs_metadata_flatten = self.inputs_metadata[all_trace_picked]
            # offset = 0
            # for i, row_set in enumerate(rows_to_pick):
            #     length = len(row_set)
            #     picked_inputs_waveform_pad[i,
            #                                :length] = picked_inputs_waveform[offset:offset+length]
            #     picked_inputs_metadata_pad[i,
            #                                :length] = picked_inputs_metadata_flatten[offset:offset+length]
            #     offset = offset + length
            # assert offset == len(picked_inputs_metadata_flatten)
            # return picked_inputs_waveform_pad, picked_inputs_metadata_pad
        elif 'numpy' in str(type(self.inputs_waveform['waveforms'])):
            inputs_waveform = self.inputs_waveform['waveforms']
            waveform_name   = self.inputs_waveform['waveform_name']
            trace_name_all  = self.inputs_waveform['all_trace_name_list']
            self.name_idx_map = dict([(name, idx) for idx, name in enumerate(waveform_name)]) if self.name_idx_map is None else self.name_idx_map
            max_rows        = max([len(row_set) for row_set in rows_to_pick])
            # rows_to_pick = [group1_row_set, group2_row_set, ..., groupn_row_set]
            index_array = None
            ### compute the ids to pickup waveform
            assert self.offset_event_start_time is not None, f"you must given offset_event_start_time"
            #     raise NotImplementedError("not finish for given offset_event_start_time")
            #     # index_array = np.full((len(rows_to_pick), max_rows), -1)
            #     # for i, row_set in enumerate(rows_to_pick):
            #     #     select_wave_id = np.array([self.name_idx_map[name] for name in trace_name_all[row_set]])
            #     #     index_array[i, :len(row_set)] = select_wave_id #(B, max_rows)
            #     # picked_inputs_waveform = inputs_waveform[index_array] #(B, max_rows, 3000, 3)
            #     # picked_inputs_waveform[index_array < 0] = 0
            # else:
            #raise
            # above picked_inputs_waveform is collected into an uniformed array
            # if self.offset_event_start_time is not None, this mean it is not an aligned array
            # this mean we need shift the wave (D=3000) by self.offset_event_start_time, 
            # each (B, max_rows) has different offset, which may lose the speed advantage
            # One possible way is createa a copy and do index picking again
            picked_inputs_waveform = []
            for i, row_set in enumerate(rows_to_pick):
                waveform_each_batch = []
                offsets     = self.offset_event_start_time[row_set]
                trace_names = trace_name_all[row_set]
                for name, offset in zip(trace_names,offsets):
                    waveform_now = inputs_waveform[self.name_idx_map[name]]
                    #(3000,3) --> (3000,3) # offset is calculated by (should_start - now_start) 
                    # if the offset is negitive, then means the waveform should start earlier then now recording, thus we use -offset for np.roll
                    
                    waveform_now = np.roll(waveform_now, -offset,axis=0) 
                    if offset < 0: waveform_now[:-offset] = 0
                    else: waveform_now[-offset:] = 0
                    waveform_each_batch.append(waveform_now) 
                waveform_each_batch = np.stack(waveform_each_batch) # (L1,3000,3)
                waveform_each_batch = np.pad(waveform_each_batch,((0,max_rows-len(waveform_each_batch)),(0,0),(0,0)))
                picked_inputs_waveform.append(waveform_each_batch)
            picked_inputs_waveform = np.stack(picked_inputs_waveform)

            index_array = np.full((len(rows_to_pick), max_rows), -1) if index_array is None else index_array
            for i, row_set in enumerate(rows_to_pick):
                index_array[i, :len(row_set)] = row_set # overwrite
            picked_inputs_metadata = {}
            for k, v in self.inputs_metadata.items():
                data = v[torch.LongTensor(index_array)]
                if isinstance(data,np.ndarray):data[index_array < 0] = 0
                picked_inputs_metadata[k] = data
            # picked_inputs_metadata = self.inputs_metadata[torch.LongTensor(index_array)]
            # picked_inputs_metadata[index_array < 0] = 0
            return picked_inputs_waveform, picked_inputs_metadata
        else:
            raise NotImplementedError(
                f"the type of self.inputs_waveform is {type(self.inputs_waveform)}")
            
    def pickup_stations(self, aset):

        if self.cutout:
            p_arrive_time = self.p_arrive_time[aset]  # (L,)
            if self.offset_event_start_time is not None:
                p_arrive_time -= self.offset_event_start_time[aset] # align to the event_start_time
            cutout        = np.random.randint(*self.cutout)
            select_aset   = aset[p_arrive_time < cutout]
        else:
            cutout = self.inputs_waveform_length
            select_aset = aset
        max_stations = self.config.max_stations
        if len(select_aset) <= max_stations:
            return cutout, select_aset
        else:
            return cutout, np.random.choice(select_aset, max_stations, replace=False)
        
    def get_target_pool(self, indexes, rows_to_picks):
        index_array = None
        filted_zero_i = [i for i, (_, rows) in enumerate(
            rows_to_picks) if len(rows) > 0]
        rows_to_pick = [rows for cutout,
                        rows in rows_to_picks if len(rows) > 0]
        max_rows = max([len(row_set) for row_set in rows_to_pick])
        outputs = {}
        for k, v in self.source_metadata.items():
            if len(v) == len(self.p_arrive_time):
                if index_array is None:
                    index_array = np.full((len(rows_to_pick), max_rows), -1)
                    for i, row_set in enumerate(rows_to_pick):
                        index_array[i, :len(row_set)] = row_set
                
                data = v[index_array]
                # if len(rows_to_pick) == 1 and len(rows_to_pick[0])==1: # this is due to the bad feature that 1,1 slice will get a number rather than a array
                #     data = np.array([[data]])
                if isinstance(data, np.ndarray):
                    data[index_array < 0] = 0
                
            else:
                data = v[indexes][filted_zero_i]
                if self.shard_size == 1:
                    data = data[0]
                if len(data.shape) == 2:
                    data = data[..., None]
            outputs[k] = data

        return outputs

    def shuffle_indexes(self):
        shuffle = self.config.shuffle
        if shuffle:
            np.random.shuffle(self.indexes)
        return self.indexes
    
    def __len__(self):
        return self.length
    
    def initialize_indexes_by_data_augmentation(self, data_augmentation_config, verbose=True):
        index_trace = []
        self.base_indexes = np.arange(len(self.event_set_list))
        index_trace.append(['initial', len(self.base_indexes)])

        magnitude_resampling = data_augmentation_config['magnitude_resampling']
        min_upsample_magnitude = data_augmentation_config['min_upsample_magnitude']
        max_upsample_magnitude = data_augmentation_config['max_upsample_magnitude']
        self.reverse_index = None
        if magnitude_resampling > 1:
            magnitude = self.quake_magnitude
            for i in np.arange(min_upsample_magnitude, max_upsample_magnitude):
                ind = np.where(np.logical_and(
                    i < magnitude, magnitude <= i + 1))[0]
                self.base_indexes = np.concatenate(
                    (self.base_indexes, np.repeat(ind, int(magnitude_resampling ** (i - 1) - 1))))
        index_trace.append(
            [f'upsample_on_level_{min_upsample_magnitude}-{max_upsample_magnitude}', len(self.base_indexes)])

        upsample_high_station_events = data_augmentation_config['upsample_high_station_events']
        if upsample_high_station_events is not None:
            new_indexes = []
            for ind in self.base_indexes:
                n_stations = len(self.event_set_list[ind])
                new_indexes += [ind for _ in range(
                    n_stations // upsample_high_station_events + 1)]
            self.base_indexes = np.array(new_indexes)
        index_trace.append(
            ['upsample_high_station_events', len(self.base_indexes)])

        oversample = data_augmentation_config['oversample']
        self.indexes = np.repeat(self.base_indexes, oversample, axis=0)
        index_trace.append(['runtime_indexes', len(self.indexes)])

        if verbose:
            names = [a[0] for a in index_trace]
            sizes = [a[1] for a in index_trace]
            df = pd.DataFrame([sizes], columns=names)
            print(tabulate.tabulate(df,  headers='keys',
                  tablefmt='psql', showindex=False))
        return self.base_indexes, self.indexes

    def __getitem__(self, index):

        data = None
        i = 0
        while data is None:
            i += 1
            indexes = self.indexes[index *
                                   self.shard_size:(index + 1) * self.shard_size]
            if i > 10:
                raise NotImplementedError("too many fails")
            data = self.get_data(indexes)
            index = np.random.randint(self.length)
        return data

    @staticmethod
    def normlized_data_base(*args, **kargs):
        return normlized_data_base(*args, **kargs)

    @property
    def quake_magnitude(self):
        return self.source_metadata["magnitude"]

    @property
    def quake_location(self):
        if 'location' in self.source_metadata:
            return self.source_metadata['location']
        else:
            return np.stack([self.source_metadata['latitude'], self.source_metadata['longitude'], self.source_metadata['deepth']], -1)

    @property
    def recorder_location(self):
        return self.inputs_metadata[..., :3]
        
    
    
def normlized_data_base(metadata,normlized_ways, mode='orgin2distribution'):
        if normlized_ways is None:return metadata
        return_array = False
        if not isinstance(metadata,dict):
            return_array = True
            metadata = {'data':metadata}
        new_metadata_pool = {}
        for key, full_data in metadata.items(): 
            if return_array:
                assert not isinstance(normlized_ways,dict),f"the normlized_ways is {normlized_ways}"
                normlized_ways = {'data':normlized_ways}
            assert key in normlized_ways,f"you want to normilize {key}{full_data.shape} but you only provide normlized_ways of {normlized_ways.keys()}"
            #assert len(methods) == metadata.shape[-1], f"get methods num = {len(methods)} and metadata shape {metadata.shape}"
            methods = normlized_ways[key]
            full_data = np.squeeze(full_data)
            if len(full_data.shape) ==1:full_data   = full_data[:,None]
            if full_data.shape[-1]==1 and not isinstance(methods[0],list):
                methods      = [methods]
            assert full_data.shape[-1] == len(methods), f"for {key}{full_data.shape} you must provide a list length={full_data.shape[-1]} for all slot, given length {len(methods)} "
            new_metadatas= []
            for i in range(full_data.shape[-1]):
                method = methods[i]
                data   = full_data[...,i]
                eg = torch if isinstance(data,torch.Tensor) else np
                _type  = method[0]
                if _type == 'gauss':
                    mean,std = method[1:]
                    if mode == 'orgin2distribution':
                        new_metadata = (data-mean)/std
                    else:
                        new_metadata = data*std + mean
                elif _type == 'lognorm':
                    max_response_value,unit,offset,scale = method[1:]
                    if mode == 'orgin2distribution':
                        new_metadata = eg.log((data-max_response_value)/unit + 1)
                        new_metadata = (new_metadata - offset)/scale
                    else:
                        data = data*scale + offset
                        data = (data - 1)*unit + max_response_value
                        new_metadata = eg.exp(data-1)
                elif _type == 'minmax':
                    min_value,max_value = method[1:]
                    if mode == 'orgin2distribution':
                        new_metadata = (data-min_value)/(max_value - min_value)
                    else:
                        new_metadata = data*(max_value - min_value) + min_value
                elif _type == 'none':
                    new_metadata = data
                else:
                    raise NotImplementedError(f"Type:{_type} is not support")
                new_metadatas.append(new_metadata)
            if len(new_metadatas)>1:
                new_metadatas = eg.stack(new_metadatas,-1)
            else:
                new_metadatas = new_metadatas[0]
            new_metadata_pool[key] = new_metadatas
        if return_array:
            new_metadata_pool = new_metadata_pool['data']
        return new_metadata_pool
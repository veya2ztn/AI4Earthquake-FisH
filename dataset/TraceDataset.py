import torch
import numpy as np
from typing import Optional
from .dataset_arguements import TraceDatasetConfig, SamplingStrategy
from .resource.resource_arguements import HandleCreater
from scipy.special import expit as sigmoid
import pandas as pd
from .normlizer import get_normlizer_convert, _NormlizerConvert
from typing import Dict
from scipy import signal
from scipy.interpolate import interp1d

class BaseTraceDataset(torch.utils.data.Dataset):
    downstream = None
    def set_downstream_pool(self, downstream_pool=None):
        assert downstream_pool is not None
            # self.downstream = {'time':get_normlizer_convert(),
            #                    'magnitude':get_normlizer_convert(),
            #                    'distance':get_normlizer_convert(),
            #                    'angle':get_normlizer_convert(),
            #                    'deepth':get_normlizer_convert()}
        
        self.downstream = {}
        for key, val in downstream_pool.items():
            #self.downstream[key] = 1 if ('unit' not in val or val['unit']=='auto') else val['unit']
            assert 'unit' not in val or val['unit']=='auto' or val['unit']==1, "After 20231212, the normlizer args should be assigned. Thus, please use unit flag in the normlizer"
            if 'normlizer' in val:
                self.downstream[key] = get_normlizer_convert(**val['normlizer'])
            else:
                self.downstream[key] = get_normlizer_convert()

    def set_sampling_strategy(self, sampling_strategy: Optional[SamplingStrategy]):
        if sampling_strategy is None:
            assert self.warning_window is not None , """
            warning_window and start_point_sampling_strategy are not compatible.
            The warning_window is old logic and we are going to use the start_point_sampling strategy.
            """
            self.early_warning = None
            if self.split == 'train':
                if self.warning_window > 0:
                    self.early_warning = self.warning_window
                    self.start_point_sampling_strategy = 'random_sample_before_p_in_warning_window'
                elif self.warning_window == -1:
                    self.start_point_sampling_strategy = 'random_sample_in_ahead_H_to_more_H'
                elif self.warning_window == -2:
                    self.start_point_sampling_strategy = 'random_sample_in_ahead_L_to_more_L'
                elif self.warning_window == -3:
                    self.start_point_sampling_strategy = 'random_sample_in_ahead_L_to_less_L'
                elif self.warning_window == -4:
                    self.start_point_sampling_strategy = 'random_sample_in_ahead_L_to_p'
                elif self.warning_window == 0:
                    self.start_point_sampling_strategy = 'the_beigin_of_the_sequence'
            else:
                self.start_point_sampling_strategy = 'early_warning_before_p'
                self.early_warning = self.warning_window//2 if self.warning_window>0 else 0
            print(f"[{self.split} Dataset]: we are using the start_point_sampling_strategy: {self.start_point_sampling_strategy}, with early_warning: {self.early_warning}")
        else:
            self.start_point_sampling_strategy = sampling_strategy.strategy_name
            self.early_warning = sampling_strategy.early_warning

    def post_init(self):
        pass

class EarthQuakePerTrack(BaseTraceDataset):
    return_idx     = False
    def __init__(self, metadatas, waveform_handle, split, namemap=None, normer=None, noise_engine=None, config: TraceDatasetConfig = None):
        assert config is not None, "config is not given"
        self.config          = config 
        
        #### below is the rawdata, which is injected by the resource loader rather then config
        self.metadatas=self.resource_metadatas       = metadatas
        self.waveform_handle = torch.from_numpy(waveform_handle).float() if isinstance(waveform_handle,np.ndarray) else waveform_handle
        if isinstance(waveform_handle, HandleCreater):
            self.waveform_handle = waveform_handle()
        self.noise_engine = noise_engine
        self.namemap         = namemap
        self.split           = split # <------ data augement should only work for training dataset
        self.trace_means = self.trace_stds = None
        if normer is not None:
            self.trace_means = torch.from_numpy(normer[0])
            self.trace_stds = torch.from_numpy(normer[1])


        #### below is the config 
        self.max_length      = config.max_length
        self.warning_window  = config.warning_window if config.warning_window is not None else config.max_length
        self.slide_stride    = config.slide_stride
        self.status_type     = config.status_type
        self.remove_noise    = config.remove_noise
        self.use_db_waveform = config.use_db_waveform
        self.start_point_sampling_strategy = None
        self.sampling_frequence = config.Resource.sampling_frequence
        self.channel_order   = config.Resource.channel_order
        self.bandfilter_rate = config.Resource.bandfilter_rate
        self.signal_lowpass  = config.Resource.signal_lowpass
        self.post_init()

    def __len__(self):
        return len(self.metadatas)
    
    def get_phase_signal(self, p_start_pos_list, s_start_pos_list, w_end_pos_list, status_type:str, sample_length:int, relaxtion:Optional[int]=None):
        if isinstance(p_start_pos_list, int): p_start_pos_list = [p_start_pos_list]
        if isinstance(s_start_pos_list, int): s_start_pos_list = [s_start_pos_list]
        if isinstance(w_end_pos_list  , int): w_end_pos_list   = [w_end_pos_list  ]
        num_samples = len(p_start_pos_list)
        assert np.all(np.array(p_start_pos_list)>=0), "p_start_pos_list should be positive"
        p_start_pos_arr = np.clip(p_start_pos_list, 0, sample_length)
        s_start_pos_arr = np.clip(s_start_pos_list, 0, sample_length)
        w_end_pos_arr   = np.clip(w_end_pos_list, 0, sample_length)

        indices = np.arange(sample_length)

        # Initialize status sequences with zeros
        status_seqs = np.zeros((num_samples, sample_length), dtype='int')

        if status_type == 'N0P1S2':
            for i in range(num_samples):
                
                status_seqs[i, int(p_start_pos_arr[i]):int(w_end_pos_arr[i])] += 1
                status_seqs[i, int(s_start_pos_arr[i]):int(w_end_pos_arr[i])] += 1
        elif status_type == 'PtoS':
            for i in range(num_samples):
                status_seqs[i, p_start_pos_arr[i]:s_start_pos_arr[i]] += 1
        elif status_type in ['whereisP','whereisS']:
            x = np.arange(num_samples)
            y = p_start_pos_arr if status_type == 'whereisP' else s_start_pos_arr
            x = x[y<sample_length]
            y = y[y<sample_length]
            x = x[y>0]
            y = y[y>0]
            if len(x)>0:
                status_seqs[x, y] += 1
        elif status_type == 'P0to5':
            frequency = int(self.sampling_frequence )
            for i in range(num_samples):
                status_seqs[i, max(p_start_pos_arr[i]-0*frequency,0) :min(p_start_pos_arr[i]+5*frequency, sample_length)] += 1
        elif status_type == 'P-2to10':
            frequency = int(self.sampling_frequence )
            for i in range(num_samples):
                status_seqs[i, max(p_start_pos_arr[i]-2*frequency,0) :min(p_start_pos_arr[i]+10*frequency, sample_length)] += 1
        elif status_type == 'P-2to30':
            frequency = int(self.sampling_frequence )
            for i in range(num_samples):
                status_seqs[i, max(p_start_pos_arr[i]-2*frequency,0) :min(p_start_pos_arr[i]+30*frequency, sample_length)] += 1
        elif status_type == 'P-2to60':
            frequency = int(self.sampling_frequence )
            for i in range(num_samples):
                status_seqs[i, max(p_start_pos_arr[i]-2*frequency,0) :min(p_start_pos_arr[i]+60*frequency, sample_length)] += 1
        elif status_type == 'P-10to120':
            frequency = int(self.sampling_frequence )
            for i in range(num_samples):
                status_seqs[i, max(p_start_pos_arr[i]-10*frequency,0) :min(p_start_pos_arr[i]+120*frequency, sample_length)] += 1
        elif status_type == 'probability':
            assert relaxtion is not None, "relaxtion should be given"
            assert len(p_start_pos_arr)==1, "probability only support one sample"
            p_start_pos = p_start_pos_arr[0]
            s_start_pos = s_start_pos_arr[0]
            w_end_pos   = w_end_pos_arr[0]
            unit = 1
            length = sample_length
            status_seq = np.zeros((length,3))
            start_of_p = max(p_start_pos//2, p_start_pos -  relaxtion)
            right_of_p = min((p_start_pos + s_start_pos)//2, p_start_pos + relaxtion)
            start_of_s = max((p_start_pos + s_start_pos)//2, s_start_pos - relaxtion)
            right_of_s = min((s_start_pos + w_end_pos)//2, s_start_pos + relaxtion)
            start_of_e = max((s_start_pos + w_end_pos)//2, w_end_pos - relaxtion)
            right_of_e = min((w_end_pos + length)//2, w_end_pos + relaxtion)
            status_seq[:start_of_p][:,0] = unit
            alpha= 2
            data = sigmoid(alpha*np.linspace(unit, -unit, right_of_p-start_of_p))
            status_seq[start_of_p:right_of_p][:,0] = data
            status_seq[start_of_p:right_of_p][:,1] = 1-data
            status_seq[right_of_p:start_of_s][:,1] = unit

            data = sigmoid(alpha*np.linspace(unit, -unit, right_of_s-start_of_s))
            status_seq[start_of_s:right_of_s][:,1] = data
            status_seq[start_of_s:right_of_s][:,2] = 1-data
            status_seq[right_of_s:start_of_e][:,2] = unit

            data = sigmoid(alpha*np.linspace(unit, -unit, right_of_e-start_of_e))
            status_seq[start_of_e:right_of_e][:,2] = data
            status_seq[start_of_e:right_of_e][:,0] = 1-data

            status_seq[right_of_e:][:,0] = unit
            status_seq=[status_seq]
        elif status_type == 'probabilityPeak':
            assert relaxtion is not None, "peak_relaxtion should be given"
            #assert len(p_start_pos_arr)==1, "probability only support one sample"
            status_seqs = []
            for center_of_p, center_of_s, center_of_e in zip(p_start_pos_arr, s_start_pos_arr, w_end_pos_arr):

                unit = 1
                alpha= 1
                length = sample_length
                status_seq = np.zeros((length,3))
                
                left_of_p  = max (center_of_p//2,                center_of_p - relaxtion)
                right_of_p = min((center_of_p + center_of_s)//2, center_of_p + relaxtion)

                left_of_s  = max((center_of_p + center_of_s)//2, center_of_s - relaxtion)
                right_of_s = min((center_of_s + center_of_e)//2, center_of_s + relaxtion)

                left_of_e  = max((center_of_s + center_of_e)//2, center_of_e - relaxtion)
                right_of_e = min((center_of_e + length)     //2, center_of_e + relaxtion)

                status_seq[:left_of_p][:,0] = 1
                data = alpha*np.linspace(1, 0, center_of_p-left_of_p)
                status_seq[left_of_p:center_of_p][:,0] = data
                status_seq[left_of_p:center_of_p][:,1] = 1-data
                data = alpha*np.linspace(0, 1, right_of_p-center_of_p)
                status_seq[center_of_p:right_of_p][:,0] = data
                status_seq[center_of_p:right_of_p][:,1] = 1-data

                status_seq[right_of_p:left_of_s][:,0] = 1
                data = alpha*np.linspace(1, 0, center_of_s-left_of_s)
                status_seq[left_of_s:center_of_s][:,0] = data
                status_seq[left_of_s:center_of_s][:,2] = 1-data
                data = alpha*np.linspace(0, 1, right_of_s-center_of_s)
                status_seq[center_of_s:right_of_s][:,0] = data
                status_seq[center_of_s:right_of_s][:,2] = 1-data

                status_seq[right_of_s:][:,0] = 1
                status_seqs.append(status_seq)
        elif status_type == 'P_Peak_prob':
            assert relaxtion is not None, "peak_relaxtion should be given"
            #assert len(p_start_pos_arr)==1, "probability only support one sample"
            status_seqs = []
            for center_of_p, center_of_s, center_of_e in zip(p_start_pos_arr, s_start_pos_arr, w_end_pos_arr):

                unit = 1
                alpha= 1
                length = sample_length
                status_seq = np.zeros((length))
                
                left_of_p  = max (center_of_p//2,  center_of_p - relaxtion)
                right_of_p = min((center_of_p + length)//2, center_of_p + relaxtion)

                status_seq[left_of_p:center_of_p]  = np.linspace(0, 1, center_of_p-left_of_p)
                status_seq[center_of_p:right_of_p] = np.linspace(1, 0, right_of_p-center_of_p)

                status_seqs.append(status_seq)
        else:
            raise NotImplementedError
        if len(status_seqs) ==1:
            status_seqs = status_seqs[0]
        
        return status_seqs
            
    
    def data_augment(self, waveform, status_seq, p_start_pos,s_start_pos,w_end_pos):
        ### must deal with pair data, the waveform and the status_seq both~~!
        raise NotImplementedError

    def get_sample_start(self, p_start_pos,s_start_pos,w_end_pos, length,start_point_sampling_strategy=None):
        start_point_sampling_strategy = self.start_point_sampling_strategy if start_point_sampling_strategy is None else start_point_sampling_strategy
        if start_point_sampling_strategy == 'random_sample_before_p_in_warning_window':
            waveform_start  = np.random.randint(p_start_pos - self.early_warning,p_start_pos) # can be negative
        elif start_point_sampling_strategy == 'random_sample_in_ahead_H_to_more_H':
            waveform_start  = np.random.randint(-self.max_length//2, length + self.max_length//2) # 
        elif start_point_sampling_strategy == 'random_sample_in_ahead_L_to_more_L':
            waveform_start  = np.random.randint(-self.max_length+1, length + self.max_length) # 
        elif start_point_sampling_strategy == 'random_sample_in_ahead_L_to_less_L':
            waveform_start  = np.random.randint(-self.max_length+1, length - self.max_length) # 
        elif start_point_sampling_strategy == 'random_sample_in_ahead_L_to_p':
            # for a recurrent task training, the start of the sequence must be case that hidden state is zero
            # which means the start of the sequence must be the noise rather than a quake event.
            waveform_start  = np.random.randint(-self.max_length+1, p_start_pos)
        elif start_point_sampling_strategy == 'random_sample_ahead_to_p':
            waveform_start  = np.random.randint(-self.early_warning, p_start_pos)
        elif start_point_sampling_strategy == 'early_warning_before_p':
            waveform_start  = p_start_pos - self.early_warning
        elif start_point_sampling_strategy == 'the_beigin_of_the_sequence':
            waveform_start = 0
        elif start_point_sampling_strategy == 'ahead_L_to_the_sequence':
            waveform_start = -self.early_warning
        else:
            raise NotImplementedError(f'start_point_sampling_strategy={start_point_sampling_strategy} is not implemented')
        this_item_start = max(0,waveform_start)
        return this_item_start, waveform_start


        
    def expanding_signal_to_correct_length(self, waveform, trendfrom, status_seq, waveform_start, tracename=None):
        if waveform_start < 0:
            pad_length = -waveform_start
            if waveform is not None:
                if self.noise_engine is not None:
                    assert tracename is not None, "tracename should be given"
                    noise = None
                    noise_length = 0
                    while noise_length<pad_length:
                        noise_tracename = self.noise_engine.get_trace_from_name(tracename,self.split)
                        new_noise = torch.from_numpy(self.get_single_waveform('noise',noise_tracename ))
                        if noise is None:
                            noise = new_noise
                        else:
                            noise = torch.cat([noise, new_noise],0)
                        noise_length = len(noise)
                    # if len(noise) < pad_length:
                    #     noise = torch.nn.functional.pad(noise, ((0,  0, pad_length - len(noise), 0)))
                    noise = noise[-pad_length:]
                    waveform   = torch.cat([noise, waveform],0) #(6000, 3) -> (6000+pad_length, 3)
                else:
                    waveform   = torch.nn.functional.pad(waveform, ((0,  0, pad_length, 0)))
            if trendfrom is not None:
                assert self.noise_engine is None, "noise_engine and trendfrom are not compatible"
                if self.config.dataset_version == 'alpha':
                    trendfrom = torch.nn.functional.pad(trendfrom, ((0,  0, pad_length, 0)))
                elif self.config.dataset_version == 'beta':
                    leftpadding = trendfrom[0][None].repeat(pad_length,1)
                    trendfrom   = torch.cat([leftpadding, trendfrom],0)
                else:
                    raise NotImplementedError("dataset_version should be alpha or beta")
            if status_seq is not None:
                if len(status_seq.shape) == 1:
                    status_seq = np.pad(status_seq,((pad_length,0)))
                elif len(status_seq.shape) == 2:
                    padding_value = np.array([[1,0,0]]).repeat(pad_length,0)
                    status_seq = np.concatenate([padding_value, status_seq],0)
                else:
                    raise NotImplementedError
        
        new_length = len(waveform) if waveform is not None else len(status_seq)
        if new_length<self.max_length:
            pad_length = self.max_length - new_length
            if waveform is not None:
                if self.noise_engine is not None:
                    assert tracename is not None, "tracename should be given"
                    noise = None
                    noise_length = 0
                    while noise_length<pad_length:
                        noise_tracename = self.noise_engine.get_trace_from_name(tracename,self.split)
                        new_noise = torch.from_numpy(self.get_single_waveform('noise',noise_tracename ))
                        if noise is None:
                            noise = new_noise
                        else:
                            noise = torch.cat([noise, new_noise],0)
                        noise_length = len(noise)
                    # if len(noise) < pad_length:
                    #     noise = torch.nn.functional.pad(noise, ((0,  0, 0, pad_length - len(noise))))
                    noise = noise[:pad_length]
                    waveform   = torch.cat([waveform, noise],0)
                else:
                    waveform   = torch.nn.functional.pad(waveform, ((0, 0, 0, pad_length)))
            if trendfrom is not None:
                assert self.noise_engine is None, "noise_engine and trendfrom are not compatible"
                if self.config.dataset_version == 'alpha':
                    trendfrom   = torch.nn.functional.pad(trendfrom, ((0, 0, 0, pad_length)))
                elif self.config.dataset_version == 'beta':
                    rigthpadding= trendfrom[-1][None].repeat(pad_length,1)
                    trendfrom   = torch.cat([trendfrom, rigthpadding],0)
                else:
                    raise NotImplementedError("dataset_version should be alpha or beta")
            if status_seq is not None:
                if len(status_seq.shape) == 1:
                    status_seq = np.pad(status_seq,((0,pad_length)))
                elif len(status_seq.shape) == 2:
                    padding_value = np.array([[1,0,0]]).repeat(pad_length,0)
                    #status_seq = np.pad(status_seq,((0,pad_length),(0,0)))
                    status_seq = np.concatenate([status_seq, padding_value],0)
                else:
                    raise NotImplementedError
        return waveform, trendfrom, status_seq
    

    def waveform_augment(self, waveform,metadata):
        if self.remove_noise:
            if 'noise_refer_level' in metadata:
                refer_level = torch.from_numpy(np.clip(eval(metadata['noise_refer_level']), 1, np.inf))
                waveform    = waveform/refer_level
            else:
                ENZ_noise   = torch.from_numpy(np.clip(np.array([metadata[f'basenoise_amps_{k}'] for k in 'ENZ'])[None,:], 1, np.inf))
                assert self.bandfilter_rate>0, "bandfilter_rate should be positive, because the noise level only for centrilized data"
                if self.bandfilter_rate != 0.005: print(f"Warning: the noise level is calculated for 0.005 bandfilter_rate, but your bandfilter_rate is {self.bandfilter_rate}")
                waveform    = waveform/ENZ_noise #<make sure the order is correct
        if self.trace_means is not None:
            waveform = (waveform - self.trace_means)/self.trace_stds
        

        if self.use_db_waveform:
            waveform = torch.clamp(torch.log(torch.abs(waveform)+1e-10),0)*torch.sign(waveform)
        return waveform
    

    def get_metadata(self, index):
        if index == 'all':
            metadata = self.metadatas.copy()
        else:
            metadata = self.metadatas.iloc[index]
        if self.config.Resource.downsample_rate:
            metadata = metadata.copy()
            metadata.p_arrival_sample = metadata.p_arrival_sample//self.config.Resource.downsample_rate
            metadata.s_arrival_sample = metadata.s_arrival_sample//self.config.Resource.downsample_rate
            metadata.coda_end_sample  = metadata.coda_end_sample //self.config.Resource.downsample_rate
        elif self.config.Resource.upsample_rate:
            metadata = metadata.copy()
            metadata.p_arrival_sample = metadata.p_arrival_sample*self.config.Resource.upsample_rate
            metadata.s_arrival_sample = metadata.s_arrival_sample*self.config.Resource.upsample_rate
            metadata.coda_end_sample  = metadata.coda_end_sample *self.config.Resource.upsample_rate
        return metadata

    
        
    def get_single_waveform(self, index, tracename=None):
        if tracename is None:
            metadata = self.get_metadata(index)
            tracename= metadata.trace_name
        if self.namemap:
            waveform = self.waveform_handle[self.namemap[tracename]]
            
        else:
            waveform = self.waveform_handle.get('data/'+str(tracename))[()]
        return waveform
    
    def get_waveform_rawdata(self, index, tracename=None):
        return self.get_single_waveform(index,tracename=tracename)

    def get_waveform(self, index):
        waveform = self.get_waveform_rawdata(index)
        order = [ self.channel_order.index(c) for c in 'ENZ']
        waveform = waveform[:,order]
        runtimemean = None
        if self.config.Resource.downsample_rate:
            assert  self.config.Resource.upsample_rate is None, "downsample_rate and upsample_rate are not compatible"
            waveform    = waveform[::self.config.Resource.downsample_rate]
        elif self.config.Resource.upsample_rate:
            assert  self.config.Resource.downsample_rate is None, "downsample_rate and upsample_rate are not compatible"
            x_known              = np.arange(len(waveform))
            linear_interpolation = interp1d(x_known, waveform, kind='linear',axis=0)
            waveform=linear_interpolation(np.linspace(0, len(waveform)-1, self.config.Resource.upsample_rate*len(waveform)))
        
        if self.bandfilter_rate:
            assert not self.signal_lowpass, "signal_lowpass and bandfilter_rate are not compatible"
            if isinstance(self.bandfilter_rate, float):
                runtimeewn  = pd.DataFrame(waveform).ewm(alpha=self.bandfilter_rate)
                runtimemean = runtimeewn.mean().values
                waveform    = waveform - runtimemean
                runtimemean = torch.from_numpy(runtimemean)
            elif isinstance(self.bandfilter_rate, str):
                if 'multiewn' in self.bandfilter_rate:
                    if self.bandfilter_rate == 'multiewn8':
                        alpha_list = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005]
                    else:
                        raise NotImplementedError
                    waveform_list   = []
                    runtimemean_list= []
                    for alpha in alpha_list:
                        runtimemean  = pd.DataFrame(waveform).ewm(alpha=alpha).mean().values
                        waveform_list.append(waveform - runtimemean)
                        runtimemean_list.append(runtimemean)
                    waveform    = np.concatenate(waveform_list, -1)
                    runtimemean = np.concatenate(runtimemean_list, -1)
                    runtimemean = torch.from_numpy(runtimemean)
                else:
                    # leave here for average filter
                    raise NotImplementedError
        
        if self.signal_lowpass:
            assert 'random' not in self.start_point_sampling_strategy, "signal_lowpass and random_sample/warning_window are not compatible"
            sos = signal.butter(5, (1,self.config.Resource.sampling_frequence//2-5), 'bandpass', fs=self.config.Resource.sampling_frequence, output='sos')
            waveform = signal.sosfilt(sos, waveform, axis=-2)
        
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform).float()

        if self.config.Resource.amplifier_signal:
            waveform   = waveform*self.config.Resource.amplifier_signal
            runtimemean= runtimemean*self.config.Resource.amplifier_signal if runtimemean is not None else None
            
        return waveform,  runtimemean
    
    def get_labels_from_metadata(self,metadata, normlize_function_pool:Dict[str, _NormlizerConvert]):
        labels = {}
        if 'time' in normlize_function_pool:
            raise NotImplementedError
            #quake_happened_time = -((metadata.quake_start_timestamp - metadata.trace_start_timestamp) - this_item_start/100) # make it positive, unit is s
            # unit is s. from x to x+60
            #quake_happened_time = quake_happened_time + np.arange(self.max_length)*0.01  # make it become time that start from current
            #labels['time'] = normlize_function_pool['time'].convert_in_to_machine_data(quake_happened_time)
        if 'magnitude' in normlize_function_pool:
            quake_magnitude     = metadata.source_magnitude
            labels['magnitude'] = normlize_function_pool['magnitude'].convert_in_to_machine_data(quake_magnitude)
        if 'distance' in normlize_function_pool:
            quake_distance      = metadata.source_distance_km
            labels['distance'] = normlize_function_pool['distance'].convert_in_to_machine_data(quake_distance)
        if 'angle' in normlize_function_pool:
            quake_angle      = (metadata.back_azimuth_deg - 180 )/180  # <--- in range (-1, 1)
            labels['angle'] = normlize_function_pool['angle'].convert_in_to_machine_data(quake_angle)
        if 'line' in normlize_function_pool:
            degree = metadata.back_azimuth_deg%180
            quake_angle  = degree/180  # <--- in range (0, 1)
            labels['line'] = normlize_function_pool['line'].convert_in_to_machine_data(quake_angle)
        if 'angle_x' in normlize_function_pool:labels['angle_x'] = normlize_function_pool['angle_x'].convert_in_to_machine_data(np.cos(metadata.back_azimuth_deg/180*np.pi))
        if 'angle_y' in normlize_function_pool:labels['angle_y'] = normlize_function_pool['angle_y'].convert_in_to_machine_data(np.sin(metadata.back_azimuth_deg/180*np.pi))
        if 'angle_vector' in normlize_function_pool:
            angle_vector = np.stack([np.cos(metadata.back_azimuth_deg/180*np.pi), np.sin(metadata.back_azimuth_deg/180*np.pi)],-1)
            labels['angle_vector'] = normlize_function_pool['angle_vector'].convert_in_to_machine_data(angle_vector)
        if 'line_vector' in normlize_function_pool:
            degree = metadata.back_azimuth_deg%180
            degree = 2*degree # now in [0, 360]
            line_vector = np.stack([np.cos(degree/180*np.pi), np.sin(degree/180*np.pi)],-1)
            labels['line_vector'] = normlize_function_pool['line_vector'].convert_in_to_machine_data(line_vector)
        if 'deepth' in normlize_function_pool:labels['deepth']  = normlize_function_pool['deepth'].convert_in_to_machine_data(metadata.source_depth_km)
        if 'x' in normlize_function_pool:labels['x'] = normlize_function_pool['x'].convert_in_to_machine_data(metadata.source_distance_km* np.cos(metadata.back_azimuth_deg/180*np.pi))
        if 'y' in normlize_function_pool:labels['y'] = normlize_function_pool['y'].convert_in_to_machine_data(metadata.source_distance_km* np.sin(metadata.back_azimuth_deg/180*np.pi))
        if 'ESWN' in normlize_function_pool:
            x = metadata.source_distance_km* np.cos(metadata.back_azimuth_deg/180*np.pi)
            y = metadata.source_distance_km* np.sin(metadata.back_azimuth_deg/180*np.pi)
            if isinstance(x, pd.Series): x = x.values
            if isinstance(y, pd.Series): y = y.values
            rightQ = (x >= 0)
            upperQ = (y >= 0)
            ESWN   = (rightQ.astype(int) << 1) | upperQ.astype(int) 
            ### """
            ### def f(a, b):return (a.astype(int) << 1) | b.astype(int)
            ### a_array = np.array([True, True, False, False, False,True])
            ### b_array = np.array([True, False, True, False, False,True])
            ### y = f(a_array, b_array)
            ### ----> [3 2 1 0 0 3]
            ### """
            labels['ESWN'] = normlize_function_pool['ESWN'].convert_in_to_machine_data(ESWN)
        
        if 'SPIN' in normlize_function_pool:
            x = metadata.source_distance_km* np.cos(metadata.back_azimuth_deg/180*np.pi)
            y = metadata.source_distance_km* np.sin(metadata.back_azimuth_deg/180*np.pi)
            if isinstance(x, pd.Series): x = x.values
            if isinstance(y, pd.Series): y = y.values
            SPIN = np.logical_xor(x >= 0, y >= 0).astype('int')
            labels['SPIN'] = normlize_function_pool['SPIN'].convert_in_to_machine_data(SPIN)
        return labels
    
    

    def all_single_labels(self, downstream=None):
        downstream = self.downstream if downstream is None else downstream
        return self.get_labels_from_metadata(self.get_metadata('all'), downstream)

    def pick_wave_and_labels(self, waveform, runtime_mean, metadata, p_start_pos, s_start_pos, w_end_pos,this_item_start, waveform_start):
        origin_length = len(waveform)
        waveform      = self.waveform_augment(waveform,metadata)
        
        
            
        #### decide the start of the wave
        if self.slide_stride is None or self.slide_stride <= 0:
            
            this_item_end  = waveform_start + self.max_length
            assert this_item_end>0, "this_item_end should be positive"
            #this_item_end   = this_item_start + self.max_length
            waveform    =    waveform[this_item_start:this_item_end] 
            if runtime_mean is not None:runtime_mean=runtime_mean[this_item_start:this_item_end] 
            waveform, runtime_mean, _ = self.expanding_signal_to_correct_length(waveform, runtime_mean, None, waveform_start, metadata.trace_name)

            #status_seq=status_seq[this_item_start:this_item_end] 
            #None, status_seq = self.expanding_signal_to_correct_length(None, status_seq, waveform_start)
            status_seq = self.get_phase_signal(p_start_pos - waveform_start, 
                                               s_start_pos - waveform_start, 
                                               w_end_pos   - waveform_start, 
                                               self.status_type, self.max_length)

        else:
            assert runtime_mean is None, "runtime_mean should be None, if you use slide_stride mode"
            status_seq = self.get_phase_signal(p_start_pos,s_start_pos,w_end_pos, self.status_type, origin_length)
            assert self.split != 'train', "slide_stride only work for valid and test dataset"
            assert isinstance(self.slide_stride,int), "slide_stride should be int"
            assert 'time' not in self.downstream, "time property is invalid for slide_stride mode"
            assert 'status' in self.status_type, "status property is invalid for slide_stride mode"
            paded_left = int(self.max_length*1.25 - p_start_pos)
            paded_rigt = int(p_start_pos + 0.25*self.max_length)
            #(Length + Window, 3) and start of the sequence always is L sequence before P arrive
            waveform   = torch.nn.functional.pad(waveform, ((0, 0,paded_left, paded_rigt))) 
            status_seq = torch.LongTensor(np.pad(status_seq, ((paded_left, paded_rigt))))
            waveform   = waveform.unfold(0, self.max_length, self.slide_stride).transpose(2,1) #(S,3, L) -> (S,L,3)
            status_seq = status_seq.unfold(0, self.max_length, self.slide_stride) #(S, L) -> (S,L)

        #### [TODO]: label smoothing 
        #### -----------------------
        labels = self.get_labels_from_metadata(metadata, self.downstream)
        if 'hasP' in self.downstream:
            ## label_seq = self.get_phase_signal(waveform, p_start_pos,s_start_pos,w_end_pos, 'whereisP') ### <-- this logic produce no pad status which is a bug
            label_seq  = self.get_phase_signal(p_start_pos - waveform_start, s_start_pos - waveform_start, w_end_pos - waveform_start, 'whereisP', self.max_length)
            labels['hasP'] = np.array([int((label_seq > 0).any())])
        for key in ['findP', 'findS']:
            if key in self.downstream:
                label_seq  = self.get_phase_signal(p_start_pos - waveform_start, s_start_pos - waveform_start, w_end_pos - waveform_start, 
                                                   'whereisP' if key == 'findP' else 'whereisS', 
                                                   self.max_length)
                if len(label_seq.shape) > 1:
                    position = []
                    for status in label_seq:
                        pos = np.where(status>0)[0]
                        pos = np.array([-1]) if len(pos) == 0 else pos/self.max_length
                        position.append(pos)
                    position = np.stack(position)
                else: 
                    position = np.where(label_seq>0)[0]
                    position = np.array([-1]) if len(position) == 0 else position/self.max_length
                labels[key] = position
            
        if 'status' in self.downstream:
            labels['status'] = self.get_phase_signal(p_start_pos - waveform_start, s_start_pos - waveform_start, w_end_pos - waveform_start, 'N0P1S2', self.max_length)
        if 'phase_probability' in self.downstream:
            #assert self.status_type == 'probability'
            labels['phase_probability'] = self.get_phase_signal(p_start_pos - waveform_start, s_start_pos - waveform_start, w_end_pos - waveform_start, 'probability', self.max_length)
            labels['status'] = self.get_phase_signal(p_start_pos - waveform_start, s_start_pos - waveform_start, w_end_pos - waveform_start, 'N0P1S2', self.max_length)
        if 'probabilityPeak' in self.downstream:
            #assert self.status_type == 'probability'
            labels['probabilityPeak'] = self.get_phase_signal(p_start_pos - waveform_start, s_start_pos - waveform_start, w_end_pos - waveform_start, 'probabilityPeak', self.max_length, self.config.peak_relaxtion)
            labels['status'] = self.get_phase_signal(p_start_pos - waveform_start, s_start_pos - waveform_start, w_end_pos - waveform_start, 'N0P1S2', self.max_length)
        if 'P_Peak_prob' in self.downstream:
            #assert self.status_type == 'probability'
            labels['P_Peak_prob'] = self.get_phase_signal(p_start_pos - waveform_start, s_start_pos - waveform_start, w_end_pos - waveform_start, 'P_Peak_prob', self.max_length, self.config.peak_relaxtion)
            labels['status'] = self.get_phase_signal(p_start_pos - waveform_start, s_start_pos - waveform_start, w_end_pos - waveform_start, 'N0P1S2', self.max_length)
        
        labels = dict([ (k,v.astype(np.float32)) for k,v in labels.items()])

        # if runtime_mean is not None: ## we so far deal with runtime_mean aka trendform splitly
        #     waveform, runtime_mean = waveform.chunk(2, dim=-1)
        input_dict = {'status_seq':status_seq,
                      'waveform_seq': waveform.float(),
                      'labels':labels
                      }
        if self.config.return_trend:
            assert runtime_mean is not None , "runtime_mean should not be None, if you return the trend"
            input_dict['trend_seq'] = runtime_mean.float()
        return input_dict
    
    def __getitem__(self, index):
        
        assert self.downstream is not None, "please set the downstream pool via dataset.set_downstream_pool(downstream_pool)"
        assert self.start_point_sampling_strategy is not None, "please set the start_point_sampling_strategy via dataset.set_sampling_strategy(sampling_strategy)"
        metadata = self.get_metadata(index) 
        waveform, runtime_mean = self.get_waveform(index)
        
        p_start_pos   = int(metadata.p_arrival_sample)
        s_start_pos   = int(metadata.s_arrival_sample)
        w_end_pos     = int(metadata.coda_end_sample)
        origin_length = len(waveform)
        
        this_item_start, waveform_start = self.get_sample_start(p_start_pos,s_start_pos,w_end_pos, origin_length)
        
        input_dict    = self.pick_wave_and_labels(waveform, runtime_mean, metadata, p_start_pos, s_start_pos, w_end_pos, this_item_start, waveform_start)
        if self.return_idx:
            input_dict['idx'] = index
        return input_dict
    
class ConCatDataset(EarthQuakePerTrack):
    intervel_length = None

    def all_single_labels(self, downstream=None):
        downstream = self.downstream if downstream is None else downstream
        concat_traceindex_list_all = self.concat_traceindex_list.flatten()
        labels =  self.get_labels_from_metadata(self.get_metadata('all').iloc[concat_traceindex_list_all], downstream)
        concat_labels = {}
        for k,v in labels.items():
            if len(v.shape)==1:
                concat_labels[k] = v.values.reshape(self.concat_traceindex_list.shape)
            else:
                concat_labels[k] = v.values.reshape(*self.concat_traceindex_list.shape, *v.shape[1:])
        return concat_labels

    def post_init(self):
        assert self.config.component_concat_file is not None, "component_concat_file should be given if you use ConCatDataset"
        self.metadatas['theindex'] = np.arange(len(self.metadatas))
        self.tracename_index_map = self.metadatas[['trace_name', 'theindex']].set_index('trace_name')['theindex'].to_dict()
        self.resource_metadatas = df = self.metadatas
        concat_trace_list  = np.load(self.config.component_concat_file, allow_pickle=True)
        concat_trace_df    = pd.DataFrame(concat_trace_list.flatten())
        filted_row = np.all(concat_trace_df.isin(df['trace_name'].values).values.reshape(*concat_trace_list.shape),axis=-1)
        concat_trace_list = concat_trace_list[filted_row]
        assert len(concat_trace_list) > 0, "seem your resource too small to covery the minimal concat perpose"
        self.concat_trace_list = concat_trace_list
        self.concat_traceindex_list = np.array([[self.tracename_index_map[t] for t in trace_list]  for trace_list in concat_trace_list])
        self.set_padding_length(self.config.component_intervel_length)
        
    def __len__(self):
        return len(self.concat_trace_list)

    def set_padding_length(self, intervel_length):
        self.intervel_length = intervel_length


    

    def get_waveform_rawdata(self, index):
        ## thus, just need enter the list of metadata of concated trace.
        tracename_list = self.concat_trace_list[index]
        tracenames    = tracename_list
        previous_mean = None
        waveforms = []
        for i, tracename in enumerate(tracenames):
            if self.namemap:
                current_waveform = self.waveform_handle[self.namemap[tracename]]
            else:
                current_waveform = self.waveform_handle.get('data/'+str(tracename))[()]
            current_mean = current_waveform.mean(0) #(3)
            if previous_mean is not None:
                start_point = previous_mean
                end_point   = current_mean
                
                if self.config.padding_rule == 'zero': #use_zero_padding_in_concat:
                    transition_sequence = np.zeros((self.intervel_length,len(current_mean))) #(L,3)
                elif self.config.padding_rule == 'interpolation': #use_zero_padding_in_concat:
                    transition_sequence=[]
                    for start, end in zip(previous_mean, current_mean):
                        intervel_sequence = np.linspace(start,end, self.intervel_length)
                        transition_sequence.append(intervel_sequence)
                    transition_sequence=np.stack(transition_sequence,-1)
                elif self.config.padding_rule == 'noise': 
                    assert self.noise_engine is not None
                    assert tracename is not None, "tracename should be given"
                    noise_length = 0
                    noise = None
                    while noise_length<self.intervel_length:
                        noise_tracename = self.noise_engine.get_trace_from_name(tracename,self.split)
                        new_noise = torch.from_numpy(self.get_single_waveform('noise',noise_tracename ))
                        if noise is None:
                            noise = new_noise
                        else:
                            noise = torch.cat([noise, new_noise],0)
                        noise_length = len(noise)

                    pad_length = self.intervel_length
                    # if len(noise) < pad_length:
                    #     noise = torch.nn.functional.pad(noise, ((0,  0, pad_length - len(noise), 0)))
                    transition_sequence = noise[-pad_length:]
                elif self.config.padding_rule == 'repeat':
                    # this is design to use the tail noise of the first quake and head noise of the second quake
                    raise NotImplementedError
                waveforms.append(transition_sequence)
            previous_mean = current_mean  
            waveforms.append(current_waveform)
            
        return np.concatenate(waveforms) ### (3000,3) (L,3) (3000,3) (L,3) (3000,3)
    
    def get_phase_signal(self, psw_tuple_list, status_type:str, sample_length:int, relaxtion:Optional[int]=None):
        assert isinstance(psw_tuple_list, list)
        indices = np.arange(sample_length)
        num_samples = 1
        # Initialize status sequences with zeros
        status_seqs = np.zeros((sample_length), dtype='int')

        if status_type == 'N0P1S2':
            for p,s,w in psw_tuple_list:
                status_seqs[p:w] += 1
                status_seqs[s:w] += 1
        else:
            raise NotImplementedError(f'status so far only support N0P1S2')
        return status_seqs
    
    def __getitem__(self, index):
        assert self.intervel_length is not None , "please set the intervel_length via dataset.set_padding_length(:int)"
        assert self.downstream is not None, "please set the downstream pool via dataset.set_downstream_pool(downstream_pool)"
        assert self.start_point_sampling_strategy is not None, "please set the start_point_sampling_strategy via dataset.set_sampling_strategy(sampling_strategy)"
        tracename_list = self.concat_trace_list[index]
        metadata  = [self.metadatas[self.metadatas['trace_name']==tracename] for tracename in tracename_list]
        waveform, runtime_mean = self.get_waveform(index)
        invervel_length = self.intervel_length
        one_single_trace_length   = self.config.Resource.resource_length


        p_start_pos   = int(metadata[0].p_arrival_sample.values)
        s_start_pos   = int(metadata[0].s_arrival_sample.values)
        w_end_pos     = int( metadata[0].coda_end_sample.values)

        assert self.start_point_sampling_strategy == 'ahead_L_to_the_sequence'
        this_item_start, waveform_start = self.get_sample_start(p_start_pos,s_start_pos,w_end_pos, len(waveform))
        waveform       = self.waveform_augment(waveform,metadata)
        this_item_end  = waveform_start + self.max_length
        assert this_item_end>0, "this_item_end should be positive"
        #this_item_end   = this_item_start + self.max_length
        waveform    =    waveform[this_item_start:this_item_end] 
        if runtime_mean is not None:runtime_mean=runtime_mean[this_item_start:this_item_end] 
        waveform, runtime_mean, _ = self.expanding_signal_to_correct_length(waveform, runtime_mean, None, waveform_start, tracename_list[0])

        #status_seq=status_seq[this_item_start:this_item_end] 
        #None, status_seq = self.expanding_signal_to_correct_length(None, status_seq, waveform_start)
        psw_status_position = []
        for i in range(len(metadata)):
            psw_status_position.append([(one_single_trace_length+invervel_length)*i+ int(metadata[i].p_arrival_sample.values) - waveform_start, 
                                        (one_single_trace_length+invervel_length)*i+ int(metadata[i].s_arrival_sample.values) - waveform_start, 
                                        (one_single_trace_length+invervel_length)*i+ int( metadata[i].coda_end_sample.values) - waveform_start
                                        ])
        status_seq = self.get_phase_signal(psw_status_position, 
                                           self.status_type, self.max_length)
        input_dict = {'status_seq':status_seq,
                      'waveform_seq': waveform.float(),
                      'labels':{'status':status_seq}
                      }
        if self.config.return_trend:
            assert runtime_mean is not None , "runtime_mean should not be None, if you return the trend"
            input_dict['trend_seq'] = runtime_mean.float()
        if self.return_idx:
            input_dict['idx'] = index
        return input_dict
  

from typing import Tuple, List

def dict_to_dict_of_lists(output_dict, input_dict):
    for key, value in input_dict.items():
        if isinstance(value, dict):
            if key not in output_dict:
                output_dict[key] = {}
            dict_to_dict_of_lists(output_dict[key], value)
        else:
            if key not in output_dict:
                output_dict[key] = []
            output_dict[key].append(value)


def dict_of_lists_to_numpy(input_dict, dim=0, mode='stack'):
    output_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, dict):
            output_dict[key] = dict_of_lists_to_numpy(
                value, dim=dim, mode=mode)
        elif isinstance(value, list):
            if isinstance(value[0], np.ndarray):
                output_dict[key] = np.stack( value, axis=dim) if mode == 'stack' else np.concatenate(value, axis=dim)
            elif isinstance(value[0], torch.Tensor):
                output_dict[key] = torch.stack( value, dim=dim) if mode == 'stack' else torch.cat(value, dim=dim)
            else:
                output_dict[key] = np.array(value)
        else:
            output_dict[key] = value
    return output_dict

import scipy
from itertools import combinations

class EarthQuakePerGroup(EarthQuakePerTrack):
    def __len__(self):
        return len(self.groups)
    
    def post_init(self):
        self.metadatas['index'] = np.arange(len(self.metadatas))
        group_key= self.config.Resource.find_group_key(self.metadatas)
        self.groups = self.metadatas.groupby(group_key)['index'].apply(lambda x: x.to_numpy()).tolist() 
        self.has_quake_start_pos = False
        metadata_keys = self.metadatas.keys()
        if (('trace_start_time' in metadata_keys and 'source_origin_time' in metadata_keys) or 
            ('trace_start_timestamp' in metadata_keys and 'quake_start_timestamp' in metadata_keys)):
            self.has_quake_start_pos = True
        if self.has_quake_start_pos:
            if 'trace_start_timestamp' not in self.metadatas.keys():
                self.metadatas['trace_start_timestamp'] = pd.to_datetime(self.metadatas['trace_start_time']).apply(lambda x: x.timestamp()) 
            if 'quake_start_timestamp' not in self.metadatas.keys():
                self.metadatas['quake_start_timestamp'] = pd.to_datetime(self.metadatas['source_origin_time']).apply(lambda x: x.timestamp())

            
    def get_sample_start(self, p_start_pos_list:List[int],
                               s_start_pos_list:List[int],
                               w_end_pos_list:  List[int], 
                               quake_start_pos_list: Optional[List[int]], max_length):
        if quake_start_pos_list is not None:
            absolute_p_start_pos_list  = [p_start_pos_list[i] -  quake_start_pos_list[i] for i in range(len(p_start_pos_list))]
            absolute_s_start_pos_list  = [s_start_pos_list[i] -  quake_start_pos_list[i] for i in range(len(s_start_pos_list))]
            absolute_w_end_pos_list    = [w_end_pos_list[i]   -  quake_start_pos_list[i] for i in range(len(w_end_pos_list))]
            absolute_first_p_start_pos = min(absolute_p_start_pos_list)
        
        if self.start_point_sampling_strategy == 'alignment.random_sample_ahead_first_p':
            assert self.early_warning is not None
            absolute_waveform_start_pos  = np.random.randint(absolute_first_p_start_pos - self.early_warning, absolute_first_p_start_pos) # can be negative
            waveform_start_list = [absolute_waveform_start_pos + quake_start_pos_list[i] for i in range(len(quake_start_pos_list))]
        elif self.start_point_sampling_strategy == 'alignment.fix_sample_ahead_first_p':
            absolute_waveform_start_pos = absolute_first_p_start_pos - self.early_warning
            waveform_start_list = [absolute_waveform_start_pos + quake_start_pos_list[i] for i in range(len(quake_start_pos_list))]
        elif self.start_point_sampling_strategy == 'unalignment.fix_sample_ahead_p':
            waveform_start_list = [p_start_pos_list[i]- self.early_warning for i in range(len(p_start_pos_list))]
        else:
            raise NotImplementedError(f'start_point_sampling_strategy={self.start_point_sampling_strategy} is not implemented')
        
        this_item_start_list =[ max(0,waveform_start) for waveform_start in waveform_start_list]
        return this_item_start_list, waveform_start_list
    
    def get_group_labels_from_metadata(self,metadata_list, normlize_function_pool:Dict[str, _NormlizerConvert]):
        labels = {}
        extra_information = {}
        max_station = self.config.max_station
        N = len(metadata_list)
        indices = np.arange(N)
        mask = np.zeros(max_station)
        mask[:N]=1
        station_mask = mask.astype('bool')
        extra_information['station_mask'] = station_mask       # ( G, G)       ## which station we used
        if 'group_vector' in normlize_function_pool:
            coordinations=[]
            for metadata in metadata_list:
                coordinations.append(np.array([metadata.source_distance_km* np.cos(metadata.back_azimuth_deg/180*np.pi),  metadata.source_distance_km* np.sin(metadata.back_azimuth_deg/180*np.pi)]))
            coordinations= np.stack(coordinations)
            labels['group_vector'] = np.pad(coordinations, ((0,max_station-len(coordinations)),(0,0)))
            coordinations= -coordinations # take center as orignal point and get the coord for each station
            distance_matrix = scipy.spatial.distance_matrix(coordinations,coordinations)
            inv_dis         = 1/(distance_matrix+1e-6)
            inv_dis[indices,indices]=1
            distance_matrix[indices,indices]=1
            
            distance_matrix_padded = np.zeros((max_station, max_station))
            distance_matrix_padded[:N,:N] = distance_matrix
            
            inv_dis_padded = np.zeros((max_station, max_station))
            inv_dis_padded[:N,:N] = inv_dis
            
            index_pairs = list(combinations(range(max_station), 2))
            distance_intra_matrix  = np.array([distance_matrix_padded[:,i] * distance_matrix_padded[:,j] for i, j in index_pairs]).T
            inv_intra_dis          = np.array([inv_dis_padded[:,i] * inv_dis_padded[:,j] for i, j in index_pairs]).T
            
            distance_expansion = np.concatenate([distance_intra_matrix, inv_intra_dis], -1) # (N, 2*(N,2)+1)
            distance_expansion = np.pad(distance_expansion,((0,0),(1,0)),constant_values=1)
            #print(distance_expansion.shape)
            vector_state = np.zeros((max_station, max_station, 2))
            vector_state[:N,:N] = coordinations - coordinations[:, np.newaxis]
            #station_mask     = (mask[:,None]@mask[None]).astype('bool')
            
            extra_information['distance_expansion'] = distance_expansion.astype('float32') # ( G, 2*(G,2)+1) ## the 1, di,...,   didj expansion and the 1/di,..., 1/didj expansion
            extra_information['vector_state'] = vector_state.astype('float32')       # ( G, 2)       ## the vector state
            
                                                          
        return labels, extra_information 


    def group_pad_the_output(self, group_dict, max_station):
        N = len(group_dict['status_seq'])
        group_dict['status_seq']   = np.pad(group_dict['status_seq'], ((0,max_station-N),(0,0)))
        group_dict['waveform_seq'] = torch.nn.functional.pad(group_dict['waveform_seq'], (0,0,0,0,0,max_station-N))
        if 'trend_seq' in group_dict:
            group_dict['trend_seq'] = torch.nn.functional.pad(group_dict['trend_seq'], (0,0,0,0,0,max_station-N))
        keys = list(group_dict['labels'].keys())
        for key in keys:
            val = group_dict['labels'][key]
            if isinstance(val, np.ndarray):
                padding_tuple = ((0,max_station-N),) + ((0,0),)*(len(val.shape)-1)
                group_dict['labels'][key] = np.pad(val, padding_tuple)
            elif isinstance(val, torch.Tensor):
                padding_tuple = [0,0]*(len(val.shape)-1) + [0,max_station-N]
                group_dict['labels'][key] = torch.nn.functional.pad(group_dict['labels'][key],padding_tuple)
            else:
                raise NotImplementedError    
        return group_dict
        

        
        

    def __getitem__(self, i):
        assert self.downstream is not None, "please set the downstream pool via dataset.set_downstream_pool(downstream_pool)"
        assert self.start_point_sampling_strategy is not None, "please set the start_point_sampling_strategy via dataset.set_sampling_strategy(sampling_strategy)"
        group_ids = self.groups[i]
        max_station = self.config.max_station
        pickup_num = min(len(group_ids), max_station)
        if self.split == 'train':
            group_ids = np.random.choice(group_ids, pickup_num, replace=False)
        else:
            group_ids = group_ids[:pickup_num]
        N = len(group_ids)
        metadatas = [self.get_metadata(index) for index in group_ids]
        waveforms = [self.get_waveform(index) for index in group_ids]

        p_start_pos_list     = [int(meta.p_arrival_sample)   for meta in metadatas]
        s_start_pos_list     = [int(meta.s_arrival_sample)   for meta in metadatas]
        w_end_pos_list       = [int(meta.coda_end_sample)    for meta in metadatas]
        if self.has_quake_start_pos:
            quake_start_pos_list = [(int(meta.trace_start_timestamp)-int(meta.quake_start_timestamp))//self.sampling_frequence for meta in metadatas] #(must be negative)
        else:
            quake_start_pos_list = None
        
        max_length           = max([len(w) for w,_ in waveforms])
        this_item_start_list, waveform_start_list = self.get_sample_start(p_start_pos_list,s_start_pos_list, 
                                                                w_end_pos_list, quake_start_pos_list, 
                                                                max_length)

        group_dict = []
        for index,metadata,(waveform, runtime_mean), this_item_start, waveform_start  in zip(group_ids,metadatas, waveforms, this_item_start_list, waveform_start_list):
            p_start_pos = int(metadata.p_arrival_sample)
            s_start_pos = int(metadata.s_arrival_sample)
            w_end_pos   = int(metadata.coda_end_sample)
            input_dict  = self.pick_wave_and_labels(waveform, runtime_mean, metadata, p_start_pos, s_start_pos, w_end_pos, this_item_start, waveform_start)
            group_dict.append( input_dict)
   
        output = {}
        for pppp in group_dict:
            dict_to_dict_of_lists(output, pppp)
        group_dict = dict_of_lists_to_numpy(output, dim=0, mode='stack') #(B,G,L,3)
        
        group_dict = self.group_pad_the_output(group_dict, max_station)
        group_labels,extra_input = self.get_group_labels_from_metadata(metadatas, self.downstream)
        group_dict = group_dict | extra_input
        group_dict['labels'] = group_dict['labels'] | group_labels

        if self.return_idx:
            group_ids = np.pad(group_ids, (0,max_station-N), constant_values=-1)
            group_dict['idx'] = group_ids
        return group_dict

class DummyDataset(BaseTraceDataset):
    return_idx=False
    def __init__(self, *args, config: TraceDatasetConfig = None, length=1000, **kargs):
        self.max_length      = config.max_length
        self.slide_stride    = config.slide_stride
        self.length          = length
        self.warning_window  = None
        self.config    = config
    def __len__(self):
        return self.length
    
    def set_downstream_pool(self, downstream_pool=None):
        assert downstream_pool is not None
        self.downstream = {}
        self.downstream_shape = {}
        for key, val in downstream_pool.items():
            #self.downstream[key] = 1 if ('unit' not in val or val['unit']=='auto') else val['unit']
            assert 'unit' not in val or val['unit']=='auto' or val['unit']==1, "After 20231212, the normlizer args should be assigned. Thus, please use unit flag in the normlizer"
            if 'normlizer' in val:
                self.downstream[key] = get_normlizer_convert(**val['normlizer'])
            else:
                self.downstream[key] = get_normlizer_convert()
            self.downstream_shape[key] = val['channel']
    def __getitem__(self, index):
        labels = {}
        for key, channel in self.downstream_shape.items():
            if channel == 1:
                labels[key] = np.random.randn(1)[0]
            else:
                labels[key] = np.random.randn(channel)
        if 'time' in self.downstream:labels['time'] = 0.001*np.arange(self.max_length)
        if 'hasP' in self.downstream:labels['hasP'] = np.random.randint(0,2,(1,))
        if 'findP' in self.downstream:labels['findP'] = np.random.rand(1)
        if 'findS' in self.downstream:labels['findS'] = np.random.rand(1)
        if 'status' in self.downstream:labels['status'] = np.random.randint(0,3,(self.max_length,))
        if 'phase_probability' in self.downstream:labels['phase_probability'] = np.random.rand(self.max_length,3)
        labels = dict([ (k,v.astype(np.float32)) for k,v in labels.items()])
        if self.slide_stride:
            input_dict = {'status_seq': np.random.randint(0, 3, (self.slide_stride, self.max_length)),
                          'waveform_seq': np.random.randn(self.slide_stride,self.max_length, 3).astype(np.float32),
                          'labels': labels
                          }
        else:
            input_dict = {'status_seq': np.random.randint(0, 3, (self.max_length,)),
                        'waveform_seq':np.random.randn(self.max_length,3).astype(np.float32),
                        'labels':labels
                        }
        if self.return_idx:
            input_dict['idx'] = index
        if self.config.return_trend:
            input_dict['trend_seq'] = np.random.randn(self.max_length,3).astype(np.float32)
        return input_dict
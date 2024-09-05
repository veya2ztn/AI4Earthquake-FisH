
from .resource_arguements import ResourceConfig,HandleCreater,ResourceGroup
import pandas as pd
from .load_stead import STEAD_Loader
from ..utils import load_wave_data
import numpy as np
import os
from typing import List, Dict
import os
import h5py
import json
import io
from urllib.parse import urlparse
try:
    import boto3
except:
    pass

def load_resource(args: ResourceConfig, only_metadata=False):
    """
    If use hdf5, there are two options:
    1. pass a hdf5 handle. This means the multi-worker dataloader will read data from same hdf5 handle.
    2. create the hdf5 handle in dataset. This means the multi-worker dataloader will read data from different hdf5 handle.(which is faster)
    """
    local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
    #the dataset is different with the resource
    metadata_path, wavedata_source_information, normdata = args.get_data_from_name(args.resource_source)
    df = pd.read_csv(metadata_path) ## each process will load the metadata, while the waveform may be dispatched 
    df['ptos'] = df['s_arrival_sample'] - df['p_arrival_sample']
    origin_len = len(df)
    df=df[df['ptos']>0]
    new_len = len(df)
    if origin_len != new_len:
        if local_rank==0:
            print(f"Warning: {origin_len-new_len} rows are removed because of ptos<=0, this is temperary method to avoid the bug in the dataset. ")

    index_map = dtfl = None #dtfl = h5py.File("datasets/STEAD/stead.hdf5", 'r')
    normer = None
    noise_engine = None
    if only_metadata:
        return df, dtfl, index_map, normer,noise_engine
    noise_engine = args.NoiseGenerate.build_noise_engine()
    if local_rank == 0 or args.load_in_all_processing:
        if isinstance(wavedata_source_information,(list,str)):
            index_map, dtfl = load_wave_data(wavedata_source_information)
            if index_map is not None:df = df[df['trace_name'].isin(index_map.keys())]
            if normdata is not None:normer = [np.load(normdata[0]), 
                                            np.load(normdata[1])]
        elif isinstance(wavedata_source_information,dict):
            
            # if 'group' in metadata_path:
            #     group2name = df.groupby('ev_id')['trace_name'].apply(lambda x: list(x)).to_dict()
            #     dtfl = DiTingGroupHDF5Loader(group2name=group2name, name2part=name2part, hdf5_part_list=part_list)
            # else:
            if wavedata_source_information['resource_type']=='hdf5.path.list':
                
                name2part      = wavedata_source_information['name2part']
                the_needed_waveform_keys = set(df['trace_name'].values)
                if args.NoiseGenerate.name is not None:
                    for split_noise, noise_eg in noise_engine.items():
                        the_needed_waveform_keys = the_needed_waveform_keys | set(noise_eg.trace_mapping.keys())
                if name2part is not None:
                    with open(name2part,'r') as f: name2part = json.load(f)
                    name2part = {str(name):name2part[str(name)] for name in the_needed_waveform_keys}
                else:
                    name2part = {str(name):0 for name in the_needed_waveform_keys}
                part_list = wavedata_source_information['part_list']
                class identitymap:
                    def __getitem__(self, x):
                        return x
                index_map = identitymap()
                targs = []
                kargs = {'name2part':name2part, 'hdf5_part_list':part_list,  
                             'dataflag':wavedata_source_information['dataflag']}
                if args.use_resource_buffer:
                    kargs['buffer_size'] = len(name2part)
                    kargs['create_handle_oncall']=not args.initial_handle_inside_dataset
                    kargs['buffered_waveform_length']=args.resource_length
                    dtfl = HandleCreater(DiTingHDF5NumpyBuffer,targs,kargs)
                else:
                    kargs['create_handle_oncall']=args.initial_handle_inside_dataset
                    dtfl = HandleCreater(DiTingHDF5Loader,targs,kargs)
                #if not args.initial_handle_inside_dataset:
                dtfl = dtfl()
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        
    if isinstance(args, ResourceGroup):
        group_key = args.find_group_key(df)
        df['groupsize'] = df.groupby(group_key)[group_key].transform('size')
        df = df[df['groupsize']>=4]
        
    return df, dtfl, index_map, normer,noise_engine


def load_multi_station_resource(args: ResourceConfig):
    resource_config = args
    loader = STEAD_Loader(resource_config)
    inputs_waveform, inputs_metadata, source_metadata, event_set_list = loader.get_data()
    return inputs_waveform, inputs_metadata, source_metadata, event_set_list

class S3File(io.RawIOBase):

    def __init__(self, s3url, access_key, secret_key, endpoint_url, log=False):
        self._cursor = 0
        purl = urlparse(s3url)
        self._s3 = boto3.resource(
            service_name='s3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            use_ssl=False if 'https' not in endpoint_url else True
        )
        self._s3obj = self._s3.Object(purl.netloc, purl.path[1:])
        self._log = log

    def __repr__(self):
        bucket = self._s3obj.bucket_name
        key = self._s3obj.key
        return f'<{type(self).__name__} s3://{bucket}/{key} at 0x{id(self):x}>'

    def readable(self):
        return True

    def seekable(self):
        return True

    @property
    def size(self):
        return self._s3obj.content_length

    def tell(self):
        return self._cursor

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self._cursor = offset
        elif whence == io.SEEK_CUR:
            self._cursor += offset
        elif whence == io.SEEK_END:
            self._cursor = self.size + offset
        else:
            raise ValueError(f'{whence}: Unknown whence value')
        return self._cursor

    def read(self, size=-1):
        if size == -1:
            range_header = f'bytes={self._cursor}-'
            self.seek(offset=0, whence=io.SEEK_END)
        else:
            new_pos = self._cursor + size
            if new_pos >= self.size:
                return self.read()

            range_header = f'bytes={self._cursor}-{new_pos - 1}'
            self.seek(offset=size, whence=io.SEEK_CUR)
        if self._log:
            print(f'GET {range_header} SIZE {size}')
        return self._s3obj.get(Range=range_header)["Body"].read()

    def readinto(self, buff):
        data = self.read(len(buff))
        buff[:len(data)] = data
        return len(data)


def smart_get_hdf5_handle(path):
    if "s3:" in path:
        access_key = "QNL1575AXM6DF9QUZDA9"
        secret_key = "BiulXCfnNpIx6tl6P14I8W5QDw6NSU3yqSWVdbkH"
        endpoint_url = "http://10.140.27.254:80"
        spath = S3File(path, access_key, secret_key, endpoint_url, log=False)
    else:
        spath = path
    return h5py.File(spath,'r')

class DiTingHDF5Loader:
    def __init__(self, name2part:dict, 
                 hdf5_part_list:Dict[int,str], 
                 create_handle_oncall=False,
                 dataflag='earthquake'):
        self.hdf5_handle_mapping = None
        self.hdf5_part_list = hdf5_part_list

        # for hdf5_part in hdf5_part_list:
        #     part = int(os.path.basename(hdf5_part).replace('.hdf5','').split('_')[-1])
        #     self.hdf5_handle_mapping[part] = h5py.File(hdf5_part,'r') if not create_handle_oncall else hdf5_part
        self.create_handle_oncall = create_handle_oncall
        self.name2part = name2part
        self.dataflag  = dataflag
        self.dataprefix= "earthquake" if dataflag=='diting' else "data"

    def create_handle(self):
        hdf5_handle_mapping = {part: smart_get_hdf5_handle(p) for part,p in self.hdf5_part_list.items()}
        return hdf5_handle_mapping
    
    def forname(self, name):
        if not isinstance(name,str) and self.dataflag == 'diting':
            name = str(name)
            key_correct = name.split('.')
            key = key_correct[0].rjust(6,'0') + '.' + key_correct[1].ljust(4,'0')
        else:
            key = name.strip()
        return key
    
    def get_correct_name(self, name):
        if self.dataflag=='diting':
            correct_name = str(float(name))
        else:
            correct_name = name
        return correct_name
    
    def reset(self):
        if self.hdf5_handle_mapping is None:return
        for handle in self.hdf5_handle_mapping.values():
            handle.close()
        self.hdf5_handle_mapping = None


    def __getitem__(self, name):
        key  = self.forname(name)
        correct_name = self.get_correct_name(name)
        part = self.name2part[correct_name]
        if self.hdf5_handle_mapping is None or len(self.hdf5_handle_mapping)==0:
            self.hdf5_handle_mapping = self.create_handle()

        data = self.hdf5_handle_mapping[part].get(f'{self.dataprefix}/'+str(key))   
        assert data is not None, f"cannot find {key} in {part}"
        data = np.array(data).astype(np.float32)
        assert len(data.shape) == 2, f"the shape of {key} is {data.shape}, not 2"
        if data.shape[0] == 3:
            data =  data.T # (3, 12000) -> (12000, 3)
        
        return data



class DiTingGroupHDF5Loader(DiTingHDF5Loader):
    def __init__(self, group2name:dict, name2part:dict, hdf5_part_list:Dict[int,str], dataflag='earthquake'):
        super().__init__(name2part, hdf5_part_list)
        self.group2name = group2name
        self.dataflag = dataflag
        self.dataprefix= "earthquake" if dataflag=='diting' else "data"
    def __getitem__(self, group_name):
        data_list = []
        for name in self.group2name[group_name]:
            key  = self.forname(name)
            correct_name = self.get_correct_name(name)
            part = self.name2part[correct_name]
            data = self.hdf5_handle_mapping[part].get(f'{self.dataprefix}/'+str(key))   
            assert data is not None, f"cannot find {key} in {part}" 
            data = np.array(data).astype(np.float32)
            assert len(data.shape) == 2, f"the shape of {key} is {data.shape}, not 2"
            if data.shape[0] == 3:
                data =  data.T # (3, 12000) -> (12000, 3)
            data_list.append(data)
        data = np.stack(data_list)
        return data
    
    


import torch
from multiprocessing import shared_memory
import time
class NumpyBuffer:
    is_initialized = False
    set_length = None
    default_shape = {}
    def initialize(self, preloaded=False):
        local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0  
        rank = int(os.environ["RANK"]) if "RANK" in os.environ else 0

        self.hold = {}
        for name in self.shape.keys():
            self.hold[name] =self.create_cached_buffer(name)
        print(f"=====> GPU{local_rank}/{rank}:waiting for shared memory to be created")
        # if torch.cuda.device_count() > 1 and torch.distributed.is_available():
        #     torch.distributed.barrier()

        while True:
            now = time.time()
            try:
                self.cache_pool=dict([(name,self.get_cached_buffer(name)) for name in self.shape.keys()])
                break
            except:
                pass
            time.sleep(0.5)
            if time.time() - now>10:
                raise Exception("shared memory not created, Time out!")
        
        self.is_initialized = True

        buffer, loaded_buffer = self.get_cached_buffer('waveform')
        shared_waveform = np.ndarray(self.shape['waveform'], dtype=self.dtype, buffer=buffer.buf)
        shared_indices  = np.ndarray((self.shape['waveform'][0],), dtype=np.bool8, buffer=loaded_buffer.buf)

        local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
        if local_rank==0:print(f"""
              Each GPU will share same memory via numpy mmap mode: 
              shared_waveform: {shared_waveform.shape}: 
              shared_indices: {shared_indices.shape}: {shared_indices[:10]}
              """)
        # if local_rank ==0 and preloaded:
        #     for name in self.shape.keys():
        #         self.preloader_index_and_tensor(name, preloaded[name])
    
    # def preloader_index_and_tensor(self, name, preload = None):
        
    #     if preload is None:return
    #     buffer, loaded_buffer = self.cache_pool[name]
    #     loaded_indices = np.ndarray((self.shape[name][0],), dtype=np.int64, buffer=loaded_buffer.buf)
    #     loaded_tensor  = np.ndarray(self.shape[name], dtype=self.dtype, buffer=buffer.buf)
    #     if preload == 'random':
    #         loaded_indices[:] = np.ones_like(loaded_indices)[:]
    #         loaded_tensor[:]  = np.random.randn(*self.shape[name]).astype(self.dtype)[:]
    #     else:
    #         raise NotImplementedError
    #         preload_index, preload_tensor = preload
    #         loaded_indices[:] = preload_index[:]
    #         loaded_tensor[:]  = preload_tensor[:]
    #         del preload_index
    #         del preload_tensor

    def create_cached_buffer(self, name):
        local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0  
        rank = int(os.environ["RANK"]) if "RANK" in os.environ else 0
        #assert rank ==0, f"we have not test the multi-node case, not safe! now rank {rank}"
        
        shm = shm_indexes = None
        if local_rank ==0:
            try:
                last_shm = shared_memory.SharedMemory(name=f"shared_buffer_{name}")
                last_shm.close()
                last_shm.unlink()
            except:
                pass

            try:
                last_shm = shared_memory.SharedMemory(name=f"shared_buffer_{name}_indices")
                last_shm.close()
                last_shm.unlink()
            except:
                pass
            
            shape = self.shape[name] # Get the shape
            dtype = self.dtype # Define the data type
            num_bytes = np.prod(shape) * np.dtype(dtype).itemsize
            shm = shared_memory.SharedMemory(name=f"shared_buffer_{name}", create=True, size=num_bytes)
            # b = np.ndarray(a.shape, dtype=a.dtype, buffer=shm.buf)
            # b[:] = a[:]  # Copy the original data into shared memory
            #shm.close()
            
            
            shape       = (self.shape[name][0],)  # Get the shape
            dtype       = np.bool8  # Define the data type
            num_bytes   = np.prod(shape) * np.dtype(dtype).itemsize
            shm_indexes = shared_memory.SharedMemory(name=f"shared_buffer_{name}_indices", create=True, size=num_bytes)
            a = np.zeros((self.shape[name][0],),dtype=np.bool8) 
            c = np.ndarray(a.shape, dtype=a.dtype, buffer=shm_indexes.buf)
            c[:] = a[:]  # Copy the original data into shared memory

            #shm_indexes.close()
            
        return shm, shm_indexes
    
    

    def get_cached_buffer(self, name):
        tensor_buffer = shared_memory.SharedMemory(name=f"shared_buffer_{name}")
        loaded_buffer = shared_memory.SharedMemory(name=f"shared_buffer_{name}_indices")
        return tensor_buffer,loaded_buffer
    
    @property
    def shape(self):
        if self.set_length is None:
            return self.default_shape
        else:
            return {name:(self.set_length,*shape[1:]) for name, shape in self.default_shape.items()}

    def set_buffer_size(self, length):
        self.set_length = length

class DiTingHDF5NumpyBuffer(NumpyBuffer,DiTingHDF5Loader):
    def __init__(self, name2part:dict, hdf5_part_list:Dict[int,str], buffer_size, 
                 buffered_waveform_length,create_handle_oncall, 
                 dtype=np.float32, dataflag='earthquake'):
        self.hdf5_handle_mapping = None
        self.hdf5_part_list = hdf5_part_list
        self.name2part = name2part
        self.default_shape = {
            'waveform':(buffer_size, buffered_waveform_length, 3),
        }
        self.dtype = dtype
        self.reterive_length = buffered_waveform_length
        
        #torch.distributed.barrier()
        #self.initialize(preloaded=False) ### lets initilize the loader out if
        self.inner_name_index_pool = {}
        if create_handle_oncall:
            self.hdf5_handle_mapping = self.create_handle()
        self.dataflag = dataflag
        self.dataprefix= "earthquake" if dataflag=='diting' else "data"
    
    def set_name2index(self, allnames, strict=True):
        #allnames = self.name2part.keys()
        for index, name in enumerate(allnames):
            name = self.get_correct_name(name)
            if strict: assert name not in self.inner_name_index_pool, f"{name} is duplicated"
            self.inner_name_index_pool[name] = index   
        self.set_buffer_size(len(allnames))

    def __getitem__(self, name, hdf5_handle_mapping=None):
        assert len(self.inner_name_index_pool) > 1, "you should call set_name2index(allnames) first"
        assert self.is_initialized, "you should call initialize() first"
        #assert len(self.inner_name_index_pool) == self.shape['waveform'][0], "why the length is not equal? Call set_buffer_size"
        key  = self.forname(name)
        correct_name = self.get_correct_name(name)
        index = self.inner_name_index_pool[correct_name]
        buffer, loaded_buffer = self.cache_pool['waveform']
        shared_indices  = np.ndarray((self.shape['waveform'][0],), dtype=np.bool8, buffer=loaded_buffer.buf)
        shared_waveform = np.ndarray(self.shape['waveform'], dtype=self.dtype, buffer=buffer.buf)
        if not shared_indices[index]:
            part = self.name2part[correct_name]
            if hdf5_handle_mapping is None:
                if self.hdf5_handle_mapping is None:
                    self.hdf5_handle_mapping = self.create_handle()
                hdf5_handle_mapping = self.hdf5_handle_mapping
            data = hdf5_handle_mapping[part].get(f'{self.dataprefix}/'+str(key)) 
            
            assert data is not None, f"cannot find {key} in {part}"
            data = np.array(data).astype(self.dtype)
            assert len(data.shape) == 2, f"the shape of {key} is {data.shape}, not 2"
            if data.shape[0] == 3:
                data =  data.T # (3, 12000) -> (12000, 3)
            shared_waveform[index] = data[:self.shape['waveform'][1]]
            shared_indices[index] = True
        return shared_waveform[index]
    
    

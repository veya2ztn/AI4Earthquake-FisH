from .TraceDataset import *
from .dataset_arguements import TraceDatasetConfig,MultistationDatasetConfig
from .resource.load_resource import load_resource,load_multi_station_resource
from .EarthDataset import Stationed_PreNormlized_Dataset

def load_test_dataset(args: TraceDatasetConfig, branch:str):
    df, dtfl, index_map, normer, noise_engine = load_resource(args.Resource, only_metadata=True)
    dataset = EarthQuakePerTrack(df[df['split'] == branch],   dtfl, 'valid', namemap = index_map, normer = normer, noise_engine=noise_engine, config = args)

    return dataset

def load_data(args: TraceDatasetConfig, needed=['train', 'valid'], only_metadata=False):
    
    if args.debug:
        train_dataset = DummyDataset(config = args, length=10000)
        valid_dataset = DummyDataset(config = args, length=300)
        test_dataset  = DummyDataset(config = args, length=300)
        return {'train':train_dataset, 
                'valid':valid_dataset, 
                'test':test_dataset,
                'dev':valid_dataset}

    
    train_dataset = valid_dataset = test_dataset = None
    df, dtfl, index_map, normer,noise_engine = load_resource(args.Resource, only_metadata = only_metadata)
    needed = [n.lower() for n in needed]
    datasetclass = eval(args.dataclass_type)

    if 'train' in needed:
        train_dataset = datasetclass(df[df['split'].str.lower() == 'train'].copy(), dtfl, 'train', namemap = index_map, normer = normer, noise_engine =noise_engine['train'] if noise_engine is not None else None, config = args)
    if 'valid' in needed or 'dev' in needed:
        valid_dataset = datasetclass(df[df['split'].str.lower().isin(['dev','valid'])].copy(),   dtfl, 'valid', namemap = index_map, normer = normer, noise_engine =noise_engine['valid'] if noise_engine is not None else None, config = args)
    if 'test'  in needed:
        test_dataset  = datasetclass(df[df['split'].str.lower() == 'test'].copy(),  dtfl, 'test',  namemap = index_map, normer = normer, noise_engine =noise_engine['test'] if noise_engine is not None else None, config = args)
    return {'train':train_dataset, 
            'valid':valid_dataset, 
            'test':test_dataset,
            'dev':valid_dataset}


def load_multi_station_data(args: MultistationDatasetConfig, needed=['train', 'valid']):
    train_dataset = valid_dataset = test_dataset = None
    
    if 'train' in needed:
        args.Resource.split = 'TRAIN'
        inputs_waveform, inputs_metadata, source_metadata, event_set_list = load_multi_station_resource(args.Resource)
        train_dataset = Stationed_PreNormlized_Dataset(inputs_waveform, inputs_metadata, 
                                                source_metadata, event_set_list, args)
    if 'valid' in needed:
        args.Resource.split = 'DEV'
        inputs_waveform, inputs_metadata, source_metadata, event_set_list = load_multi_station_resource(args.Resource)
        valid_dataset = Stationed_PreNormlized_Dataset(inputs_waveform, inputs_metadata,
                                                    source_metadata, event_set_list, args)
    if 'test'  in needed:
        args.Resource.split = 'TEST'
        inputs_waveform, inputs_metadata, source_metadata, event_set_list = load_multi_station_resource(args.Resource)
        test_dataset = Stationed_PreNormlized_Dataset(inputs_waveform, inputs_metadata,
                                                    source_metadata, event_set_list, args)
    
    return {'train':train_dataset, 
            'valid':valid_dataset, 
            'test':test_dataset}



from mltool.dataaccelerate import DataSimfetcher    
from dataset.build import load_events, PreloadedEventGenerator, Stationed_PreNormlized_Dataset
from geopy.distance import geodesic
import sklearn.metrics as metrics
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch
from train_utils import make_data_regular,merger_ruler_create
import numpy as np
from dataset.build import Generatorfetcher
from dataset.utils import pickout_right_data_depend_on_dataset
from train_utils import show_df_in_table
from mltool.visualization import *
from evaluator.visualization import gauss_mimic_smoothhist, topepi,deltaepi
import os
delta_name_list = ['delta_latitude', 'delta_longitude', 'delta_deepth'] \
                + ['delta_vector_x', 'delta_vector_y' , 'delta_vector_z']
class Evaluator:
    D2KM = 111.19492664455874
    dataloader_pool={}

    def __init__(self, args, limit=None, branch='DEV', bits=torch.float32, dev_metadata=None):
        self.args = args
        #limit = 1000 if args.debug else limit
        limit = args.data.limit if limit is None else limit
        self.generator_params = generator_params = args.data.data_configs[0]
        print(f"we now only take first dataset as {branch} dataset")
        overwrite_sampling_rate = args.data.overwrite_sampling_rate
        data_path = generator_params['data_path']
        waveform_format =  generator_params.get('waveform_format',None)
        if dev_metadata is None:
            one_event_metadata, one_data, one_metadata,event_set_list = load_events(
                data_path,
                limit=limit, parts=(False, branch == 'DEV', branch == 'TEST'),
                overwrite_sampling_rate=overwrite_sampling_rate,
                shuffle_train_dev=generator_params.get('shuffle_train_dev', False),
                custom_split=generator_params.get('custom_split', None),
                min_mag=generator_params.get('min_mag', None), waveform_format=waveform_format,
                mag_key=generator_params.get('key', 'MA'))
            
        else:
            one_event_metadata, one_data, one_metadata, event_set_list = dev_metadata
        
        #if 'STEAD' in data_path:
        inputs_waveform, inputs_metadata, source_metadata, event_set_list = pickout_right_data_depend_on_dataset(data_path, 
            one_data, one_event_metadata, event_set_list=event_set_list,waveform_format=waveform_format,
            target_type=args.data.target_type)
        self.inputs_waveform = inputs_waveform
        self.inputs_metadata = inputs_metadata
        self.source_metadata = source_metadata
        self.event_set_list = event_set_list
        self.metadata = one_metadata
        # else:
        #     self.event_metadata = one_event_metadata
        #     self.data = one_data
        #     self.metadata = one_metadata
        
        
        self.generator = None
        self.bits = bits
        self.merger_ruler = merger_ruler_create(args)

        args = self.args
        time = 2
        generator_params = self.generator_params
        data_path = generator_params['data_path']
        sampling_rate = self.metadata['sampling_rate']
        noise_seconds = generator_params.get('noise_seconds', 5)
        cutout = int(sampling_rate * (noise_seconds + time))
        cutout = (cutout, cutout + 1)

        max_stations  = args.model.max_stations
        n_pga_targets = args.model.pga_targets_config['n_pga_targets']
        generator_params['magnitude_resampling'] = 1
        generator_params['transform_target_only'] = generator_params.get('transform_target_only', True)
        generator_params['upsample_high_station_events'] = None
        generator_params['oversample'] = 1
        # if 'TEAM' in data_path:
        #     validation_generator = PreloadedEventGenerator(data=self.data,
        #                                                 event_metadata=self.event_metadata,
        #                                                 coords_target=True,
        #                                                 cutout=cutout,
        #                                                 pga_targets=n_pga_targets,
        #                                                 max_stations=max_stations,
        #                                                 sampling_rate=sampling_rate,
        #                                                 select_first=True,
        #                                                 shuffle=False,shard_size=args.valid.valid_batch_size,
        #                                                 pga_mode=args.valid.pga,
        #                                                 **generator_params)
            
        #elif 'STEAD' in data_path:
        validation_generator = Stationed_PreNormlized_Dataset(self.inputs_waveform, self.inputs_metadata, self.source_metadata, self.event_set_list,
                                                        coords_target=True,
                                                        cutout=cutout,
                                                        pga_targets=n_pga_targets,
                                                        max_stations=max_stations,
                                                        sampling_rate=sampling_rate,
                                                        select_first=True,
                                                        shuffle=False,shard_size=args.valid.valid_batch_size,
                                                        pga_mode=args.valid.pga,
                                                        **generator_params)
        # else:
        #     raise NotImplementedError
        self.dataloader =  validation_generator
    def get_generator(self, time=2):
        #if time in self.dataloader_pool:return self.dataloader_pool[time]
        sampling_rate = self.metadata['sampling_rate']
        noise_seconds = self.generator_params.get('noise_seconds', 5)
        cutout = int(sampling_rate * (noise_seconds + time))
        cutout = (cutout, cutout + 1)
        self.dataloader.set_cutout(cutout)
        
        return self.dataloader
    
    
    def evaluate(self, model, logsys, time=2):
        model.eval()
        logsys.eval()
        args = self.args
        test_dataset =  self.get_generator(time=time)
        device = next(model.parameters()).device
        # test_datasampler = DistributedSampler(test_dataset,  shuffle=False) if args.distributed else None
        # test_dataloader = DataLoader(test_dataset, args.valid.valid_batch_size, shuffle=False,sampler=test_datasampler, num_workers=args.data.num_workers, pin_memory=False)
        # prefetcher = DataSimfetcher(test_dataloader, self.device)
        test_dataloader = test_dataset
        prefetcher = Generatorfetcher(test_dataloader, device)
        
        inter_b = logsys.create_progress_bar(len(test_dataloader), unit=' img', unit_scale=test_dataloader.batch_size)
        inter_b.lwrite(f"load everything, start_evaluating......", end="\r")
        result_collection = {'predic_result':{}, 'shoud_result':{}}
        result_in_reality = {'predic_result':{}, 'shoud_result':{}}
        
        count = 0
        while inter_b.update_step():
            batch = prefetcher.next()
            _input, target_pool = make_data_regular(batch, self.bits, device=device,removepad=not args.data.use_single_processing_reading)
            count += len(_input['waveform_inp'])
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=args.train.use_amp):
                    result_pool = model(**_input)
            
            result_pool = self.merge_result(result_pool)
            # result_pool = dict([(k, v.cpu().detach().numpy()) for k,v in result_pool.items()])
            # target_pool = dict([(k, v.cpu().detach().numpy()) for k,v in target_pool.items()])
            
            for key, all_data in zip(['predic_result','shoud_result'], [result_pool,target_pool]):
                for k, data in all_data.items():
                    
                    if (k in delta_name_list) :
                        masked_length  = (_input['waveform_inp']!= 0).any(3).any(2).long().sum(1).cpu()
                        #masked_length = (data != 0).long().sum(1).cpu()
                        data           = torch.nn.utils.rnn.pack_padded_sequence(data, masked_length, batch_first=True, enforce_sorted=False).data
                    if k not in result_collection[key]:result_collection[key][k] = []
                    result_collection[key][k].append(data.cpu().detach().numpy())    
            
            #result_pool = dict((k,v.cpu().detach().numpy()) for k,v in result_pool.items())
            #target_pool = dict((k,v.cpu().detach().numpy()) for k,v in target_pool.items())

            result_pool_in_real_world = self.invs_normalize_result(result_pool)
            target_pool_in_real_world = self.invs_normalize_result(target_pool)

            # for k,v in result_pool_in_real_world.items():print(f"{k}==>{v.shape}")
            # for k,v in target_pool_in_real_world.items():print(f"{k}==>{v.shape}")
            for key, all_data in zip(['predic_result', 'shoud_result'], [result_pool_in_real_world, target_pool_in_real_world]):
                for k, data in all_data.items():
                    if k not in result_in_reality[key]:result_in_reality[key][k] = []
                    if k in delta_name_list:
                        odata = data + _input['metadata_inp'][k.replace('delta_','receiver_')] ## <-- this code is not safe, or we should modify the _input['metadata_inp'] to a dict rather than a numpy                      
                        data = odata.sum(1)/((odata != 0).sum(1)+1e-5) ### the orginal deepth can be 0
                    result_in_reality[key][k].append(data.cpu().detach().numpy())
            
        if self.args.valid.pga:
            raise NotImplementedError("PGA is not support")
            if 'magnitude' in final_result:
                final_result['magnitude'] = final_result['magnitude'][test_dataset.reverse_index[:-1]]
            if 'location' in final_result:
                final_result['location'] = final_result['location'][test_dataset.reverse_index[:-1]]
            pga_pred = []
            pga_result = final_result['pga']
            for i, (start, end) in enumerate(zip(test_dataset.reverse_index[:-1], test_dataset.reverse_index[1:])):
                sample_pga_pred = pga_result[start:end].reshape(
                    (-1,) + pga_result.shape[-2:])
                sample_pga_pred = sample_pga_pred[:len(test_dataset.pga[i])]
                pga_pred += [sample_pga_pred]
            final_result['pga'] = pga_pred

        for big_pool in [result_collection,result_in_reality]:
            for k1 in ['predic_result', 'shoud_result']:
                pool = big_pool[k1]
                for k2 in pool.keys():
                    pool[k2] = np.concatenate(pool[k2])

        
        #final_result  = self.merge_result(final_result)
        return (result_collection['predic_result'], result_collection['shoud_result'],  
                result_in_reality['predic_result'], result_in_reality['shoud_result'])

    @staticmethod
    def calc_mag_stats(mean_mag, true_mag):
        
        r2   = metrics.r2_score(true_mag, mean_mag)
        rmse = np.sqrt(metrics.mean_squared_error(true_mag, mean_mag))
        mae  = metrics.mean_absolute_error(true_mag, mean_mag)
        return {
                "r2":r2, 
                "rmse":rmse, 
                "mae":mae
            }

    @staticmethod
    def detect_location_keys(columns):
        candidates = [['LAT', 'Latitude(°)', 'Latitude'],
                      ['LON', 'Longitude(°)', 'Longitude'],
                      ['DEPTH', 'JMA_Depth(km)', 'Depth(km)', 'Depth/Km']]
        coord_keys = []
        for keyset in candidates:
            for key in keyset:
                if key in columns:
                    coord_keys += [key]
                    break
        if len(coord_keys) != len(candidates):
            raise ValueError('Unknown location key format')
        return coord_keys

    def calc_loc_stats(self, mean_coords, true_coords,root_path=None,epoch=-1):
        
        # mean_coords *= 100
        # mean_coords[:, :2] /= Evaluator.D2KM
        # mean_coords[:, 0] += pos_offset[0]
        # mean_coords[:, 1] += pos_offset[1]
        
        dist_epi = np.zeros(len(mean_coords))
        dist_hypo = np.zeros(len(mean_coords))
        for i, (pred_coord, true_coord) in enumerate(zip(mean_coords, true_coords)):
            dist_epi[i] = geodesic(pred_coord[:2], true_coord[:2]).km
            
            dist_hypo[i] = np.sqrt(dist_epi[i] ** 2 + (pred_coord[2] - true_coord[2]) ** 2)
        

        if epoch == -1:
            torch.save({'predic_coords':mean_coords,
                        'should_coords':true_coords,
                        'dist_epi':dist_epi,
                        'dist_hypo':dist_hypo
                        },'debug/metric_pool.pt')
        if root_path is None:root_path = self.args.SAVE_PATH

        topepi(dist_epi, root_path,epoch=epoch)

        deltaepi(dist_epi, true_coords, mean_coords, root_path, epoch=epoch)
        rmse_epi = np.sqrt(np.mean(dist_epi ** 2))
        mae_epi = np.mean(np.abs(dist_epi))

        rmse_hypo = np.sqrt(np.mean(dist_hypo ** 2))
        mae_hypo = np.mean(dist_hypo)

        return {
            "rmse_hypo":rmse_hypo,
            "mae_hypo":mae_hypo, 
            "rmse_epi":rmse_epi, 
            "mae_epi":mae_epi
        }

    @staticmethod
    def calc_pga_stats(pga_pred, pga_true):
        if len(pga_pred) == 0:
            return np.nan, np.nan, np.nan
        else:
            pga_pred = np.concatenate(pga_pred, axis=0)
            mean_pga = np.sum(pga_pred[:, :, 0] * pga_pred[:, :, 1], axis=1)
            pga_true = np.concatenate(pga_true, axis=0)
            mask = ~np.logical_or(np.isnan(pga_true), np.isinf(pga_true))
            pga_true = pga_true[mask]
            mean_pga = mean_pga[mask]
            r2 = metrics.r2_score(pga_true, mean_pga)
            rmse = np.sqrt(metrics.mean_squared_error(pga_true, mean_pga))
            mae = metrics.mean_absolute_error(pga_true, mean_pga)
            return {
                "r2":r2, 
                "rmse":rmse, 
                "mae":mae
            }

    def merge_result(self, result_pool):
        new_pool = {}
        for key, result  in result_pool.items():
            if key == 'magnitude':
                new_pool[key] = torch.squeeze((result[:, :, 0] * result[:, :, 1]).sum(1))[:,None] 
            elif key == 'location':
                #print(result.shape)
                new_pool[key] = torch.squeeze(self.merger_ruler(result)) 
            elif key in ['delta_latitude', 'delta_longitude', 'delta_deepth',
                         'delta_vector_x', 'delta_vector_y', 'delta_vector_z']:
                new_pool[key] = torch.squeeze(result)  # (B,L) -> (B,L)
            else:
                raise NotImplementedError(f"{key} is not supported")
        return new_pool
    

    def invs_normalize_result(self, result_pool):
        new_pool = {}
        for k, v in result_pool.items():
            if 'source_normlized_ways' in self.generator_params:
                method_pool= {k:self.generator_params['source_normlized_ways'][k]}
            else:
                if k == 'magnitude':method_pool={k:self.generator_params['magnitude_normlized_ways']}
                elif k == 'location':method_pool={k:self.generator_params['location_normlized_ways']}
                else:
                    raise NotImplementedError
            if k in delta_name_list:
                masked_length  = (v != 0).long().sum(1).cpu()
                _input      = torch.nn.utils.rnn.pack_padded_sequence(v, masked_length, batch_first=True, enforce_sorted=False)
                real_result = PreloadedEventGenerator.normlized_data_base({k:_input.data}, method_pool, mode='distribution2orgin')
                real_result = torch.nn.utils.rnn.pad_sequence(real_result[k].split(_input.batch_sizes.tolist()), batch_first=True).transpose(0, 1)[_input.unsorted_indices]
                new_pool[k] = real_result
            else:
                new_pool[k] = PreloadedEventGenerator.normlized_data_base({k:v}, method_pool, mode='distribution2orgin')[k]
        return new_pool
            
        new_pool = {}
        for key,result  in result_pool.items():
            if key == 'magnitude':
                if 'magnitude_normlized_ways' in self.generator_params:
                    magnitude_normlized_ways = self.generator_params['magnitude_normlized_ways']
                else:
                    magnitude_normlized_ways = self.generator_params['source_normlized_ways'][-1:]
                new_pool[key] = PreloadedEventGenerator.normlized_data_base(result, magnitude_normlized_ways, mode='distribution2orgin')
            elif key == 'location':
                if 'location_normlized_ways' in self.generator_params:
                    location_normlized_ways = self.generator_params['location_normlized_ways']
                else:
                    location_normlized_ways = self.generator_params['source_normlized_ways'][:3]
                new_pool[key] = PreloadedEventGenerator.normlized_data_base(np.squeeze(result),location_normlized_ways, mode='distribution2orgin')
            else:
                raise NotImplementedError
        return new_pool
    
    def compute_once_metric(self, final_result, true_result=None, root_path=None, epoch=-1):
        metric_pool = {}
        for key, result in final_result.items():
            if key == 'magnitude':               
                true_mag = self.event_metadata['Magnitude']  if true_result is None else true_result[key]
                mean_mag = result
                metrics  = self.calc_mag_stats(np.squeeze(mean_mag), np.squeeze(true_mag))
                for k, v in metrics.items():metric_pool[f"mag_{k}"]  = v
            elif key == 'location':
                if true_result is None:
                    coord_keys  = Evaluator.detect_location_keys(self.event_metadata.columns)
                    true_coords = self.event_metadata[coord_keys].values  
                else:
                    true_coords = true_result[key]
                mean_coords = result

                metrics = self.calc_loc_stats(mean_coords, true_coords, root_path=root_path, epoch=epoch)
                for k, v in metrics.items():metric_pool[f"loc_{k}"]  = v
            elif key == 'pga':
                raise NotImplementedError
                pga_key = self.generator_params.get('pga_key', 'pga')
                metrics = self.calc_pga_stats(result, self.data[pga_key])
                for k, v in metrics.items():metric_pool[f"pga_{k}"]  = v
        
        if 'delta_latitude' in final_result:
            ### the delta_latitude is actually latitude since we input the predic_in_real_world, should_in_real_world
            mean_coords = np.stack([final_result['delta_latitude'],
                                    final_result['delta_longitude'],
                                    final_result['delta_deepth']], -1)
            true_coords = np.stack([true_result['delta_latitude'],
                                    true_result['delta_longitude'],
                                    true_result['delta_deepth']], -1)
            metrics = self.calc_loc_stats(mean_coords, true_coords, root_path=root_path, epoch=epoch)
            for k, v in metrics.items():
                metric_pool[f"loc_{k}"] = v
        
        if epoch == -1:
            torch.save({'final_result': final_result,
                        'true_result': true_result,
                        }, 'debug/result_pool.pt')
        if 'delta_vector_x' in final_result:
            from pyproj import Geod, Transformer
            reverse_transformer = Transformer.from_crs("EPSG:4978", "EPSG:4326")
            pred_x, pred_y, pred_z   = reverse_transformer.transform(final_result['delta_vector_x']*1000,
                                                                     final_result['delta_vector_y']*1000,
                                                                     final_result["delta_vector_z"]*1000)
            pred_z = -pred_z
            pred_pool = { 'latitude':pred_x,
                         'longitude':pred_y,
                            'deepth':pred_z/1000} # to km

            pred_coords = np.stack([pred_x, pred_y, pred_z],-1)

            true_x, true_y, true_z = reverse_transformer.transform(true_result['delta_vector_x']*1000, 
                                                                   true_result['delta_vector_y']*1000, 
                                                                   true_result["delta_vector_z"]*1000)
            true_z = -true_z
            true_pool = { 'latitude':true_x,
                         'longitude':true_y,
                          'deepth': true_z/1000}  # to km
            
            true_coords = np.stack([true_x, true_y, true_z], -1)

            self.visualize_result(pred_pool, true_pool,pred_pool, true_pool, root_path=root_path, epoch=epoch)
            metrics = self.calc_loc_stats(pred_coords, true_coords, root_path=root_path, epoch=epoch)
            for k, v in metrics.items():
                metric_pool[f"loc_{k}"] = v

        return metric_pool


    def visualize_result(self, predic_result, should_result, predic_in_real_world, should_in_real_world, 
                         root_path="debug", epoch=0):
        key_list = list(predic_result.keys())
        predic_result_origin = predic_in_real_world#self.invs_normalize_result(predic_result)
        should_result_origin = should_in_real_world#self.invs_normalize_result(should_result)
        for key in key_list:
            predic        = np.squeeze(predic_result[key])
            should        = np.squeeze(should_result[key])
            predic_origin = np.squeeze(predic_result_origin[key])
            should_origin = np.squeeze(should_result_origin[key])
            
            if len(predic.shape)==1:       predic=predic[:,None]              
            if len(should.shape)==1:       should=should[:,None]              
            if len(predic_origin.shape)==1:predic_origin=predic_origin[:,None]
            if len(should_origin.shape)==1:should_origin=should_origin[:,None]

                        

            channel_num = should.shape[-1]
            assert channel_num<4
            for channel in range(channel_num):
                predic_now        = np.squeeze(predic[:,channel])
                should_now        = np.squeeze(should[:,channel])
                predic_origin_now = np.squeeze(predic_origin[:,channel])
                should_origin_now = np.squeeze(should_origin[:, channel])

                fig,axes = plt.subplots(2,2,figsize=(12,6))
                
                ax = axes[0][0]
                smoothhist(predic_now,color='g',label='pred',ax=ax)
                smoothhist(should_now,color='r',label='groud',ax=ax)
                ax.legend()
                mae = np.mean(abs(predic_now - should_now))
                rmse= np.sqrt(np.mean((predic_now - should_now)**2))
                mean_std_text = f'MAE: {mae:.2f}, RMSE: {rmse:.2f}'
                ax.text(0.1, 0.95, mean_std_text, transform=ax.transAxes,fontsize=12, verticalalignment='top')

                ax = axes[0][1]
                gauss_mimic_smoothhist(predic_now - should_now,ax = ax)
                
                ax = axes[1][0]
                
                smoothhist(predic_origin_now,color='g',label='pred',ax=ax)
                try:
                    smoothhist(should_origin_now,color='r',label='groud',ax=ax)
                except:
                    print(key)
                    print(should_origin_now.shape)
                    print(np.isnan(should_origin_now).sum())
                    print(np.isnan(should_origin_now).any())
                    raise
                ax.legend()
                mae = np.mean(abs(predic_origin_now - should_origin_now))
                rmse = np.sqrt(np.mean( (predic_origin_now - should_origin_now)**2))
                mean_std_text = f'MAE: {mae:.2f}, RMSE: {rmse:.2f}'
                ax.text(0.1, 0.95, mean_std_text, transform=ax.transAxes,fontsize=12, verticalalignment='top')



                ax = axes[1][1]
                gauss_mimic_smoothhist(predic_origin_now - should_origin_now, ax=ax)

                ax.set
                save_dir = os.path.join(root_path, 'trace', f'{key}_{channel}')
                if not os.path.exists(save_dir):os.makedirs(save_dir)
                plt.suptitle(f'{key}_{channel}')
                if isinstance(epoch,int):
                    name = f"epoch-{epoch:03d}.png"
                else:
                    name = f"{epoch}.png"
                fig.savefig(os.path.join(save_dir, name))
                fig.clear()
        plt.clf()
    def get_return_metric(self):
        metric_list = []
        if not self.args.model.event_config['no_event_token']:
            metric_list+=[
                "loc_rmse_hypo","loc_mae_hypo","loc_rmse_epi","loc_mae_epi",
                "mag_r2", "mag_rmse", "mag_mae"
            ]
        if self.args.valid.pga:
            metric_list+=[
                "pga_r2", "pga_rmse", "pga_mae"
            ]
        return metric_list
        

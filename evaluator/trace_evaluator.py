import torch
from tqdm import tqdm
import numpy as np
import os
from .visualization import *
from torch.utils.data import DataLoader
import re
import pandas as pd
from mltool.visualization import *
from model.utils import  compute_accu_matrix
from dataset.utils import load_wave_data
from project_arguements import ProjectConfig
from typing import Union
from .evaluator_arguements import EvaluatorConfig, EvalPlotConfig, RecurrentInferenceMode, SlideInferenceMode, NormalInferenceMode
from dataset.dataset_arguements import TraceDatasetConfig
from dataset import  load_data
from config.utils import print_namespace_tree
from model.predictor.DownstremBuilder import extra_metric_computing
from dataset.dataset_arguements import ConCatDatasetConfig,GroupDatasetConfig
from trace_utils import think_up_a_unique_name_for_inference
from .phase_picking_strategy import *
from typing import Dict
from dataset.TraceDataset import EarthQuakePerTrack, EarthQuakePerGroup
EVENT_PROPERTY = ['x', 'y', 'deepth', 'magnitude', 'distance', 'angle', 'shift','pred_angle_deg','group_shift','GM_shift']



def load_pred_data(args:ProjectConfig,save_data_root):
    save_data_root = save_data_root.strip('/')
    save_root = os.path.dirname(save_data_root.strip('/'))
    files = [os.path.join(save_data_root, data_path) for data_path in os.listdir(save_data_root) if ".npy" in data_path]
    branch = os.path.split(save_root)[-1] 
    dataset_name = re.findall(r"checkpoints\/(.*?)\/", save_root)[0]
    print(f"Notice!!!! Your test set  branch is {branch} of {dataset_name}")
    dataset_pool = load_data(args.DataLoader.Dataset, needed=[branch.lower()])
    dataset = dataset_pool[branch.lower()]
    dataset.set_downstream_pool(args.model.Predictor.downstream_pool)
    dataset.set_sampling_strategy(args.task.sampling_strategy.valid_sampling_strategy)
    
    metadata = dataset.metadatas
    
    print(f"use follow files to plot")
    print(f"Your dataset length is ====>  {len(dataset)}  <===== ")
    for file in files:print(file)
    alldata = {}
    for file in files:read_dict_of_numpy(alldata, file)
    alldata = dict_of_lists_to_numpy(alldata, dim=0, mode='concat')
    return alldata, metadata, dataset, save_root

# def retrieve_real_data(args:ProjectConfig,alldata, dataset):
#     all_single_labels = dataset.all_single_labels()
#     preded_tensor_pool = {}
#     target_tensor_pool = {}
#     for key, val in alldata.items():
#         if key not in all_single_labels: continue
#         preded_tensor_pool[key]  = val['pred']
#         target =  all_single_labels[key].values if isinstance(all_single_labels[key], pd.Series) else all_single_labels[key]
#         if len(target.shape) == 1: target = target[:,None]
#         target_tensor_pool[key]  = target
#     return preded_tensor_pool, target_tensor_pool

def retrieve_real_data(args:ProjectConfig,alldata, dataset):
    all_single_labels = dataset.all_single_labels()
    for key, val in alldata.items():
        if key not in all_single_labels: continue
        target = all_single_labels[key].values if isinstance(all_single_labels[key], pd.Series) else all_single_labels[key]
        target = target[alldata['idx']]   #### make sure the order is correct
        if len(target.shape) == 1: target = target[:,None]
        
        if 'real' in alldata[key]:
            print(f"""
                  real value of {key} has already in the alldata, why you double create??? please make sure they are same. 
                  The existed one shape is {alldata[key].shape} and the new one shape is {target.shape}
                """)
            assert alldata[key].squeeze().shape == target.squeeze().shape
        alldata[key]['real']  = target
    return alldata

def align_the_data(args:ProjectConfig, alldata, dataset):
    if isinstance(args.DataLoader.Dataset,ConCatDatasetConfig):
        true_sample_numbers = len(dataset.concat_trace_list)
    else: 
        true_sample_numbers = len(dataset.metadatas)
    
    if isinstance(args.DataLoader.Dataset,GroupDatasetConfig):
        station_mask   = alldata['idx']>=0 
        alldata['idx'] = alldata['idx'][station_mask]
        alldata['station_mask'] = station_mask
    if alldata['idx'].max() + 1 == true_sample_numbers:
        print(f"The test dataset contain whole the data in origin datatset, those we will realign order of row prediction and make it consistant to the metadata")
        should_order  = np.arange(true_sample_numbers)
    else:
        should_order  = alldata['idx']
        print(f"You only prediction part of the test dataset. The unique row in your prediction is {len(alldata['idx'])} and the length of metadata is {true_sample_numbers}")
    
    
    if len(alldata['idx']) != true_sample_numbers:
        print(f"""
              Warning: the length of alldata['idx'] is {len(alldata['idx'])}, but the length of metadata is {true_sample_numbers}. It usually means the 
              prediction data is collect from a multiGPU processing and there a batch expansion for fair GPU dispatch.
              we will realign the prediction and make it consistant to the metadata 
              """)
        
        true_idx_pool= {}
        repeat_idxs  = []
        for slot_now, true_idx in enumerate(alldata['idx']):
            if true_idx in true_idx_pool:repeat_idxs.append(slot_now)    
            true_idx_pool[true_idx] = slot_now
        print(f"There are totally {len(repeat_idxs)} repeat idxs")
        print(repeat_idxs)
        
        correct_order = np.array([true_idx_pool[iii] for iii in should_order])

        for key in alldata.keys():
            if key in ['idx']:continue
            alldata[key]['pred'] = alldata[key]['pred'][correct_order]
            if 'real' in alldata[key]:
                alldata[key]['real'] = alldata[key]['real'][correct_order]
            
        alldata['idx'] = should_order
    return alldata

def align_and_complete(args:ProjectConfig, alldata, dataset):
    alldata =     align_the_data(args, alldata, dataset)
    alldata = retrieve_real_data(args,alldata, dataset)
    # you can only retrieve real data after correct align it, since we do not 
    return alldata

def plot_for_continue_quake(args:ProjectConfig, dataset, alldata, save_root, infer_mode):
    prefix = think_up_a_unique_name_for_inference(args)
    real_time_for_one_stamp=1.0/args.DataLoader.Dataset.Resource.sampling_frequence
    expantion=expansion=args.model.Embedding.resolution
    waveform_start = - dataset.early_warning
    one_single_trace_length = dataset.config.Resource.resource_length
    intervel_length = dataset.intervel_length
    trace_name_p_arrive_map = dataset.metadatas[['trace_name','p_arrival_sample']].set_index('trace_name')['p_arrival_sample'].to_dict()
    trace_name_s_arrive_map = dataset.metadatas[['trace_name','s_arrival_sample']].set_index('trace_name')['s_arrival_sample'].to_dict()
    trace_name_coda_end_map = dataset.metadatas[['trace_name', 'coda_end_sample']].set_index('trace_name')[ 'coda_end_sample'].to_dict()
    psw_collect = {}
    for tracename_list in tqdm(dataset.concat_trace_list):
        for i, tracename in enumerate(tracename_list):
            if i not in psw_collect:psw_collect[i]={'p':[],'s':[],'e':[]}
            psw_collect[i]["p"].append((one_single_trace_length+intervel_length)*i + trace_name_p_arrive_map[tracename]  - waveform_start )
            psw_collect[i]["s"].append((one_single_trace_length+intervel_length)*i + trace_name_s_arrive_map[tracename]  - waveform_start )
            psw_collect[i]["e"].append((one_single_trace_length+intervel_length)*i + trace_name_coda_end_map[tracename]  - waveform_start )
    for i in psw_collect.keys():
        psw_collect[i]["p"] = np.array(psw_collect[i]["p"])
        psw_collect[i]["s"] = np.array(psw_collect[i]["s"])
        psw_collect[i]["e"] = np.array(psw_collect[i]["e"])   




    sorted_list = [int((one_single_trace_length+intervel_length)*(i+1/2)  - waveform_start ) for i in range(len(psw_collect))]# indicate where two divide the sequence into different part AAAAAAAAAA00000000000BBBBBBBBBB
    #sorted_list = [one_single_trace_length*i+intervel_length*(i-1/2)  - waveform_start ) for i in range(len(psw_collect))]# indicate where two divide the sequence into different part AAAAAAAAAA00000000000BBBBBBBBBB
    real_start_location=[]
    for i in range(len(psw_collect)):
        real_start_location.append( int((one_single_trace_length+intervel_length)*(i)  - waveform_start ))
    slots = [0,1,2,3,4,5,6,7,8] if not args.task.do_fasttest_eval else list(alldata['idx'])[:8]
    
    if 'findP' in alldata:
        long_term_findP_plot(alldata, dataset, args.model.Predictor.slide_feature_window_size, args.model.Predictor.slide_stride_in_training, 
                            real_time_for_one_stamp=1.0/args.DataLoader.Dataset.Resource.sampling_frequence,
                            save_path=os.path.join(save_root, f'{prefix}.{infer_mode}_findP.snap.png'),
                            expansion=args.model.Embedding.resolution)
    if 'status' in alldata:
        
        windows_size = 7
        ### ground truth
        

        
        #### pred
        ###### by_status     
        phasepicking_strategy = PhasePickingStratagy_Status(expansion=args.model.Embedding.resolution, 
                                                            status_strategy=StatusDeterminer_Threshold(p_threshold=80,s_threshold=80),
                                                            windows_size=7, judger=0.98, timetype='realtime')
        p_position_map_pool, s_position_map_pool = phasepicking_strategy(alldata['status']['pred'])
        long_term_phasepicking(args, p_position_map_pool, s_position_map_pool, dataset,slots= slots, save_path=os.path.join(save_root, f'{prefix}.{infer_mode}_phasepeaking.bystatus.snap.png'))
        sequence_limit_length = dataset.max_length
        results = calculate_longterm_score(args, sequence_limit_length, psw_collect,sorted_list,p_position_map_pool)
        for quake_index, metric_pool in results.items():
            if args.task.upload_to_wandb:wandb.log({f"multiquake/{metric_name}": metric_value for metric_name, metric_value in metric_pool.items()}|{f"multiquake/quake_index": quake_index})

        print(f"the result of phase picking by status is {results}")
        ###### by_prob: pick wavepacket
        phasepicking_strategy = PhasePickingStratagy_Prob(tri_th_l=70, tri_th_r=70, expansion=args.model.Embedding.resolution, output_one=False, offset=0)
        p_position_map_pool, s_position_map_pool = phasepicking_strategy(alldata['status']['pred'])
        sequence_limit_length = dataset.max_length
        long_term_phasepicking(args, p_position_map_pool, s_position_map_pool, dataset,slots= slots, save_path=os.path.join(save_root, f'{prefix}.{infer_mode}_phasepeaking.byprob.snap.png'))
        results = calculate_longterm_score(args, sequence_limit_length, psw_collect,sorted_list,p_position_map_pool)
        for quake_index, metric_pool in results.items():
            if args.task.upload_to_wandb:wandb.log({f"multiquake/{metric_name}": metric_value for metric_name, metric_value in metric_pool.items()}|{f"multiquake/quake_index": quake_index})
        print(f"the result of phase picking by_prob is {results}")


        ###### by_prob: pick hard max
        expansion = args.model.Embedding.resolution
        
        divide_pos = (real_start_location[0] + real_start_location[1])//2//expansion
        phasepicking_strategy = PhasePickingStratagy_Prob(tri_th_l=70, tri_th_r=70, expansion=expansion, output_one=True, offset=0, end = divide_pos)
        p_position_map_pool_1, s_position_map_pool_1 = phasepicking_strategy(alldata['status']['pred'])
        phasepicking_strategy = PhasePickingStratagy_Prob(tri_th_l=70, tri_th_r=70, expansion=expansion, output_one=True, offset=divide_pos, end =None )
        p_position_map_pool_2, s_position_map_pool_2 = phasepicking_strategy(alldata['status']['pred'])
        p_position_map_pool = {}
        s_position_map_pool = {}
        for k in p_position_map_pool_1.keys():
            p_position_map_pool[k] = (p_position_map_pool_1[k] if k in p_position_map_pool_1 else set()) | (p_position_map_pool_2[k] if k in p_position_map_pool_2 else set())
            s_position_map_pool[k] = (s_position_map_pool_1[k] if k in s_position_map_pool_1 else set()) | (s_position_map_pool_2[k] if k in s_position_map_pool_2 else set())
        sequence_limit_length = dataset.max_length
        long_term_phasepicking(args, p_position_map_pool, s_position_map_pool, dataset,slots= slots, 
                                        save_path=os.path.join(save_root, f'{prefix}.{infer_mode}_phasepeaking.byhardmax.snap.png'))
        results = calculate_longterm_score(args, sequence_limit_length, psw_collect,sorted_list,p_position_map_pool)
        for quake_index, metric_pool in results.items():
            if args.task.upload_to_wandb:wandb.log({f"multiquake/{metric_name}": metric_value for metric_name, metric_value in metric_pool.items()}|{f"multiquake/quake_index": quake_index})
        print(f"the result of phase picking by max prob is {results}")
    
    function_to_origin = dataset.downstream
    preded_tensor_pool_long_life = {}
    target_tensor_pool_long_life = {}
    target_sequence_pool_long_life= {}
    #sorted_list=[0] + sorted_list
    # for i in range(len(psw_collect)):
    #     sorted_list.append( int((one_single_trace_length+intervel_length)*(i+1/2)  - waveform_start ))
    
    sorted_list = [0]+[int(one_single_trace_length*(i+1)+intervel_length*(i+1/2)  - waveform_start ) for i in range(len(psw_collect))]
    print(sorted_list)
    # indicate the start of each sequence
    
    for metric_item in ['magnitude','x','y']:
        if metric_item not in alldata:continue
        preded_tensor_pool_long_life[metric_item] = function_to_origin[metric_item].recovery_from_machine_data(alldata[metric_item]['pred'][...,0])
        target_tensor_pool_long_life[metric_item] = function_to_origin[metric_item].recovery_from_machine_data(alldata[metric_item]['real'])
        if target_tensor_pool_long_life[metric_item].shape[1] != preded_tensor_pool_long_life[metric_item].shape[1]:
            print(f"we rebuild the target sequence of {metric_item}")
            whole_sequence_length = preded_tensor_pool_long_life[metric_item].shape[1]
            target_sequence_pool_long_life[metric_item] = []
            for region in range(len(sorted_list)-1):
                start = sorted_list[region]//3
                end   = sorted_list[region+1]//3
                end   = min(end,whole_sequence_length)
                target_sequence_pool_long_life[metric_item].append(target_tensor_pool_long_life[metric_item][:,region:region+1]*np.ones((1,end - start ))) #(B,1) -> (B, end-start) ## (B, end1-start1) (B, end2-start2)
            target_sequence_pool_long_life[metric_item] = np.concatenate(target_sequence_pool_long_life[metric_item],-1)
            if target_sequence_pool_long_life[metric_item].shape[1] < preded_tensor_pool_long_life[metric_item].shape[1]:
                target_sequence_pool_long_life[metric_item] = np.pad(target_sequence_pool_long_life[metric_item], ((0,0),(0, preded_tensor_pool_long_life[metric_item].shape[1] - target_sequence_pool_long_life[metric_item].shape[1])))
    if len(preded_tensor_pool_long_life)>0:
        long_term_target_plot(args, preded_tensor_pool_long_life, target_tensor_pool_long_life, dataset, save_path=os.path.join(save_root, f'ConCat_{infer_mode}_target.snap.png'))
    if len(target_sequence_pool_long_life)>0:
        error_pool = {}
        for metric_item in target_sequence_pool_long_life.keys():
            print(f'we compute the error of {metric_item}, the shape of pred is {preded_tensor_pool_long_life[metric_item].shape} and the shape of real is {target_sequence_pool_long_life[metric_item].shape}')
            error_pool[metric_item] = np.abs(target_sequence_pool_long_life[metric_item] - preded_tensor_pool_long_life[metric_item])
        if 'x' in error_pool and 'y' in error_pool:
            error_shift = np.sqrt((preded_tensor_pool_long_life['x'] - target_sequence_pool_long_life['x'])**2 + 
                                    (preded_tensor_pool_long_life['y'] - target_sequence_pool_long_life['y'])**2)
            error_distance = np.abs(np.sqrt(  preded_tensor_pool_long_life['x']**2 +   preded_tensor_pool_long_life['y']**2) - 
                                    np.sqrt(target_sequence_pool_long_life['x']**2 + target_sequence_pool_long_life['y']**2))
            error_pool['shift'] = error_shift
            error_pool['distance'] = error_distance
        long_term_error_plot(args, dataset, error_pool, real_start_location,save_path=os.path.join(save_root, f'{prefix}_errorpool.snap.png'),
                                wandb_key=f'multiquake/errorpool' if args.task.upload_to_wandb else None)
    return 

def metric_computing(args:ProjectConfig, alldata, dataset):
    
    preded_tensor_pool, target_tensor_pool = {},{}
    for key, val in alldata.items():
        if 'real' not in val:continue
        real = val['real']
        pred = val['pred']
        print(f"use key={key} real={real.shape} pred={pred.shape} to do extra metrix computing")
        preded_tensor_pool[key] = pred
        target_tensor_pool[key] = real

    if isinstance(args.DataLoader.Dataset,GroupDatasetConfig): #### add group vector and station mask into tensor pool
        preded_tensor_pool['group_vector'] = alldata['group_vector']['pred']
        target_tensor_pool['group_vector'] = alldata['group_vector']['real'][:,None]
        target_tensor_pool['station_mask'] = alldata['station_mask']
    
    print("The keys of preded_tensor_pool is ", preded_tensor_pool.keys())
    print("The keys of target_tensor_pool is ", target_tensor_pool.keys())

    # ==========================================================================================================
    ## the reason we build preded_tensor_pool and target_tensor_pool is it need only support Dict[str, Tensor]
    ## However, the alldata structure is Dict[str, Dict[str, Tensor]]
    ## Thus, we need reveal the two pool from alldata here .
    error_record, prediction_pool = extra_metric_computing(preded_tensor_pool, target_tensor_pool, get_prediction='pred+real', 
                                                           function_to_origin = dataset.downstream, report_error = True)
    #print(error_record)
    alldata = alldata|prediction_pool
    return alldata
def single_metric_ploting_in_normal_mode(args:ProjectConfig, key:str, real:np.ndarray, pred:np.ndarray,dataset:EarthQuakePerTrack, save_root, infer_mode):
    if key in ['findP', 'findS']:
        true_actived = (real.squeeze() >= 0)
        pred_active  = (np.argmax(pred.squeeze(), -1) >= 0)
        out = compute_accu_matrix(torch.from_numpy(true_actived), torch.from_numpy(pred_active))
        fig = accu_matrix_plot(out, 'accu_matrix', os.path.join(save_root, f'accu_matrix.png'))              
    if key in ['pred_angle_deg']:
        pred_angle = pred
        real_angle = real
        errr_angle = np.abs(pred_angle - real_angle)
        errr_angle[errr_angle>90] = 180 - errr_angle[errr_angle>90]
        fig = angle_depend_error2real(real_angle, errr_angle, save_path=os.path.join(save_root, f'{infer_mode}_{key}_depend.png'),
                                      wandb_key=f'{infer_mode}_{key}_depend' if args.task.upload_to_wandb else None)

    if len(np.squeeze(pred).shape) > 1:
        metric_evaluate_slide_plot(real, pred,
                                f'Error_Analyis_of_{key}',
                                p_arrive_position=dataset.early_warning,
                                unit=1/dataset.sampling_frequence,
                                warning_position=dataset.early_warning,
                                save_path=os.path.join(save_root, f'slide_{key}.png'))
        print("The unit may not correct, please check it by yourself")
    else:
        print(f"error of {key}: {np.mean(np.abs(np.squeeze(pred) - np.squeeze(real)))}")

def single_metric_ploting_in_slide_mode(args:ProjectConfig, key:str,  real:np.ndarray, pred:np.ndarray, dataset:EarthQuakePerTrack, save_root, infer_mode):
    metric_evaluate_plot(real, pred, f'Error_Analyis_of_{key}', save_path=os.path.join(save_root, f'{infer_mode}_{key}.png'))
    if key in EVENT_PROPERTY:
        scatter_real_vs_pred_diagram(real, pred, f'{key}', save_path=os.path.join(save_root, f'scatter.{key}.png'))

def single_metric_ploting_in_recurrent_mode(args:ProjectConfig, key:str,  real:np.ndarray, pred:np.ndarray, dataset:EarthQuakePerTrack, save_root, infer_mode):
    
    if key not in EVENT_PROPERTY: return
    dataset_unit = 1.0/args.DataLoader.Dataset.Resource.sampling_frequence
    assert dataset.early_warning is not None and 'random' not in dataset.start_point_sampling_strategy
    #print(f"plot=>{key}=>real({real.shape})=>pred({pred.shape})")
    metric_evaluate_slide_plot(real, pred,f'Error_Analyis_of_{key}(Alignment on P!)',
                                p_arrive_position= dataset.early_warning//args.model.Embedding.resolution,
                                unit             = dataset_unit*args.model.Embedding.resolution,
                                warning_position =-1,
                                save_path=os.path.join(save_root, f'{infer_mode}_{key}.png'),
                                wandb_key=f'{infer_mode}_{key}' if args.task.upload_to_wandb else None)
    
def single_metric_ploting(args:ProjectConfig, dataset:EarthQuakePerTrack, alldata, save_root, infer_mode):
    correlation={}
    
    for key, val in alldata.items(): ## the key of alldata more than the preded_tensor_pool, target_tensor_pool
        if key in ['idx']:
            continue
        if 'real' not in alldata[key]:
            print(f'skip {key} due to it has not real')
            continue
        if 'vector' in key: 
            print(f"skip {key} since it is a vector")
            continue
        pred = val['pred']
        real = val['real']
        if key in ['findP']:
            pred = pred.squeeze()
            L = pred.shape[1]
            real = real.squeeze()
            real[real == -1] = -1/L
            pred = (np.argmax(pred, -1) - 1)/L


        print(f"{key} => pred: {pred.shape} => real: {real.shape}")
        
        if infer_mode in ['normal']:
            single_metric_ploting_in_normal_mode(args, key, real, pred , dataset, save_root, infer_mode)
        elif infer_mode in ['slide']:
            single_metric_ploting_in_slide_mode(args, key,  real, pred, dataset, save_root, infer_mode)
        elif infer_mode in ['recurrent','RecPad3000']:
            single_metric_ploting_in_recurrent_mode(args, key,  real, pred, dataset, save_root, infer_mode)
        else:
            raise NotImplementedError("what inference mode you are using?")
        
        correlation[key] = {}
        ### The goal is to build the correlation map for the error depend on series quake metadata like distance/magnitude/snr...
        if args.task.calculate_the_correlation:
            metadata = dataset.metadatas
            parameters = ['source_magnitude','source_distance_km','back_azimuth_deg','snr_db','Z_QNR','N_QNR','E_QNR']    
            for para in parameters:
                if infer_mode in ['normal']:
                    if para not in metadata:continue
                    if len(real.shape)>1 and real.shape[-1]>1:continue
                    a = np.squeeze(pred)
                    b = np.squeeze(real)
                    b = b if len(a.shape)==1 else b[:,None]
                    error  = np.abs(a-b) 
                    parad  = metadata[para].values[alldata['idx']]
                    
                    ### get the 95% data
                    datatable = pd.DataFrame({'error':error, 'para':parad})
                    datatable = datatable[datatable['para']<np.percentile(parad,95)]
                    fig=scatter_error_vs_para_diagram(datatable['error'].values, datatable['para'].values, 
                                                    f'{key}-{para}',save_path=os.path.join(save_root, f'scatter.{key}-{para}.png'))
                    plt.close(fig)
                elif infer_mode in ['recurrent','RecPad3000']:
                    raise NotImplementedError('sequence to sequence correletion is quite consuming, please do it smartly.')
                    correlation[key][para]=None
                    print(f"calculate the correlation between {key}-{para}")
                    param_series = metadata[para].reset_index(drop=True)
                    pred_shape   = np.squeeze(pred[:2]).shape
                    if len(pred_shape)==2:
                        pred_df     = pd.DataFrame(np.squeeze(pred))
                        real_series = pd.Series(np.squeeze(real), index=pred_df.index)
                        error_df    = pred_df.subtract(real_series, axis=0).abs()
                        correlation[key][para] = error_df.apply(lambda col: col.corr(param_series))
                    elif len(pred_shape)==1:
                        metadata['error'] = np.abs(np.squeeze(pred) - np.squeeze(real))
                        correlation[key][para]= [metadata[para].corr(metadata['error'])]
                    else:
                        raise 
        
def single_metric_dependency(args:ProjectConfig, dataset:EarthQuakePerTrack, alldata, save_root, infer_mode):
    print(f"Now we plot the dependence of error diagram, it cost time.........")
    print("we firstly collect the reference data from validation dataset")
    metadata = dataset.metadatas
    error_depend_data={}
    for key  in alldata.keys():
        if key in ['idx']:continue
        if 'vector' in key: continue
        b = np.squeeze(alldata[key]['real'])
        a = np.squeeze(alldata[key]['pred'])
        if len(a.shape) == len(b.shape)+1:a=a[...,0]
        error_depend_data[key] = np.abs(a - b)
    error_depend_data['real_magnitude'] =  metadata['source_magnitude'].values[alldata['idx']]
    error_depend_data['real_distance']  =  metadata['source_distance_km'].values[alldata['idx']]
    error_depend_data['real_angle']     =  metadata['back_azimuth_deg'].values[alldata['idx']]
    depend_keys = ['real_magnitude','real_distance','real_angle']
    intervel_keys = ['real_magnitude','real_distance']
    plot_error_dependence(error_depend_data, depend_keys, intervel_keys, save_root)

def findP_series_plotting(args:ProjectConfig, dataset:EarthQuakePerTrack, alldata, save_root, infer_mode):
    metadata = dataset.metadatas
    if dataset.start_point_sampling_strategy == "ahead_L_to_the_sequence":
        wavestart = 0
    elif dataset.start_point_sampling_strategy == 'early_warning_before_p':
        wavestart = metadata.p_arrival_sample.values  # metadata and dataset.metadatas is same as they share the point, however, lets use metadata for clerify
    else:
        raise NotImplementedError
    
    p_arrival_samples= metadata.p_arrival_sample.values - wavestart + dataset.early_warning
    s_arrival_samples= metadata.s_arrival_sample.values - wavestart + dataset.early_warning
    coda_end_samples = metadata.coda_end_sample.values  - wavestart + dataset.early_warning

    p_arrival_samples= p_arrival_samples[alldata['idx']]
    s_arrival_samples= s_arrival_samples[alldata['idx']]
    coda_end_samples =  coda_end_samples[alldata['idx']] 

    for key in ['findP', 'findS']:
        
        if key in alldata:
            findX_pred = alldata[key]['pred']
            if findX_pred.shape[1] != 1:
                ## findP_pred   = alldata['findP']['pred'][:,0]*args.model.Embedding.resolution
                print(f"we find its like a slide findP prediction, please manual plot it, we skip here.")
                continue 
            
            findX_offset = np.arange(findX_pred.shape[1])[None]
            findX_pred = findX_offset + findX_pred
            findX_pred = findX_pred*args.model.Embedding.resolution
            if key == 'findP':
                target = p_arrival_samples
            elif key == 'findS':
                target = s_arrival_samples
            
            target_ppicks={}
            for sample_index, psw_this_quake in enumerate(target):
                target_ppicks[sample_index] = set([psw_this_quake])
            
            ppicks={}
            for row, pred in enumerate(findX_pred):
                pred = pred[pred>=0]
                if len(pred)>0:
                    ppicks[row] = set([pred.mean().astype('int')])
            print(f"""
                ====> Below is result of {key}. Notice our frequency is {args.DataLoader.Dataset.Resource.sampling_frequence}. 
                """)
            counting_type= key

            # metrix_PS(pred=ppicks, target=p_arrival_samples,
            #         freq= args.DataLoader.Dataset.Resource.sampling_frequence,
            #         max_length= dataset.max_length ,
            #         flag=f'{counting_type}',
            #         wandb_key=f'metrix_PS' if args.task.upload_to_wandb else None,
            #         save_path=os.path.join(save_root, f'{infer_mode}_{counting_type}.png',),
            #         metric_types = ['alpha'])
            result = metrix_PS_for_set_set(pred=ppicks, target=target_ppicks,
                    freq= args.DataLoader.Dataset.Resource.sampling_frequence,
                    max_length= dataset.max_length ,
                    flag=f'{counting_type}',
                    metric_types = ['alpha'])
            result.to_csv(os.path.join(save_root, f'{infer_mode}_{counting_type}.csv'))
            print(result)
def probabilityPeak_series_plotting(args:ProjectConfig, dataset:EarthQuakePerTrack, alldata, save_root, infer_mode):
    metadata = dataset.metadatas
    if dataset.start_point_sampling_strategy == "ahead_L_to_the_sequence":
        wavestart = 0
    elif dataset.start_point_sampling_strategy == 'early_warning_before_p':
        wavestart = metadata.p_arrival_sample.values
    else:
        raise NotImplementedError
    
    p_arrival_samples= metadata.p_arrival_sample.values - wavestart + dataset.early_warning
    s_arrival_samples= metadata.s_arrival_sample.values - wavestart + dataset.early_warning
    coda_end_samples = metadata.coda_end_sample.values  - wavestart + dataset.early_warning

    p_arrival_samples= p_arrival_samples[alldata['idx']]
    s_arrival_samples= s_arrival_samples[alldata['idx']]
    coda_end_samples =  coda_end_samples[alldata['idx']] 

    
    for take_one in [True, False]:
        ffff = 'Single' if take_one else 'Multi'
        for threshold in [20, 35, 50, 70]:
            if 'probabilityPeak' in alldata:
                counting_type = 'probabilityPeak'
                ppicks_o = findAllP_Peak(alldata['probabilityPeak']["pred"][...,1],threshold,15,expansion= args.model.Embedding.resolution,output_one=take_one)
                ppicks = ppicks_o
                metrix_PS(pred=ppicks, target=p_arrival_samples,
                                    freq= args.DataLoader.Dataset.Resource.sampling_frequence,
                                    max_length= dataset.max_length ,
                                    flag=f'{counting_type}.p.T{threshold}.{ffff}',
                                    wandb_key=f'metrix_PS' if args.task.upload_to_wandb else None,
                                    save_path=os.path.join(save_root, f'{counting_type}_P.T{threshold}.{ffff}.png'))
                spicks_o = findAllP_Peak(alldata['probabilityPeak']["pred"][...,2],threshold,15,expansion= args.model.Embedding.resolution,output_one=take_one)
                spicks = spicks_o
                metrix_PS(pred=spicks, target=s_arrival_samples,
                                    freq= args.DataLoader.Dataset.Resource.sampling_frequence,
                                    max_length= dataset.max_length ,
                                    flag=f'{counting_type}.s.T{threshold}.{ffff}',
                                    wandb_key=f'metrix_PS' if args.task.upload_to_wandb else None,
                                    save_path=os.path.join(save_root, f'{counting_type}_S.T{threshold}.{ffff}.png'))
            elif 'P_Peak_prob' in alldata:
                counting_type = 'P_Peak_prob'
                ppicks_o = findAllP_Peak(alldata[counting_type]["pred"][...,0],threshold,15,expansion= args.model.Embedding.resolution,output_one=take_one)
                ppicks = ppicks_o
                metrix_PS(pred=ppicks, target=p_arrival_samples,
                                    freq= args.DataLoader.Dataset.Resource.sampling_frequence,
                                    max_length= dataset.max_length ,
                                    flag=f'{counting_type}.p.T{threshold}.{ffff}',
                                    wandb_key=f'metrix_PS' if args.task.upload_to_wandb else None,
                                    save_path=os.path.join(save_root, f'{counting_type}_P.T{threshold}.{ffff}.png'))

def status_evaluation_plotting(args:ProjectConfig, dataset:EarthQuakePerTrack, alldata, save_root, infer_mode):
    key = 'status'
    metadata = dataset.metadatas
 
    assert dataset.early_warning is not None and 'random' not in dataset.start_point_sampling_strategy

    if dataset.start_point_sampling_strategy == "ahead_L_to_the_sequence":
        wavestart = 0
    elif dataset.start_point_sampling_strategy == 'early_warning_before_p':
        wavestart = metadata.p_arrival_sample.values
    else:
        raise NotImplementedError
    
    p_arrival_samples= metadata.p_arrival_sample.values - wavestart + dataset.early_warning
    s_arrival_samples= metadata.s_arrival_sample.values - wavestart + dataset.early_warning
    coda_end_samples = metadata.coda_end_sample.values  - wavestart + dataset.early_warning

    p_arrival_samples= p_arrival_samples[alldata['idx']]
    s_arrival_samples= s_arrival_samples[alldata['idx']]
    coda_end_samples =  coda_end_samples[alldata['idx']] 


    frequency = args.DataLoader.Dataset.Resource.sampling_frequence
    results = []
    for max_filter_time in [0.5]:#[1,3,7]: #we find do not use window max filte vote is better
        max_filter_window_size = max_filter_time*frequency//args.model.Embedding.resolution
        if max_filter_window_size%2==0:max_filter_window_size+=1
        filter_time = max_filter_window_size*args.model.Embedding.resolution/frequency
        phasepicking_strategy = PhasePickingStratagy_Status(expansion=args.model.Embedding.resolution, 
                                                        status_strategy=StatusDeterminer_Threshold(p_threshold=80,s_threshold=80),
                                                        windows_size=max_filter_window_size, judger=0.98, timetype='all')
        ppicks_pool, spicks_pool = phasepicking_strategy(alldata['status']['pred'])
        

        for counting_type in ppicks_pool.keys():
            print(f"we now calculate the max_filter_time={max_filter_time} and counting_type={counting_type} ")
            ppicks = ppicks_pool[counting_type]
            spicks = spicks_pool[counting_type]
            result = metrix_PS(pred=ppicks, target=p_arrival_samples,
                    freq= args.DataLoader.Dataset.Resource.sampling_frequence,
                    max_length= dataset.max_length ,
                    flag=f'{counting_type}.p/wt{filter_time}',
                    wandb_key=f'metrix_PS' if args.task.upload_to_wandb else None,
                    save_path=os.path.join(save_root, f'{infer_mode}_{counting_type}_P_metrix.ws{filter_time}.png'),verbose=False)
            results.append([counting_type, max_filter_time, 'p'] + list(result.values()))
            result= metrix_PS(pred=spicks, target=s_arrival_samples,
                    freq= args.DataLoader.Dataset.Resource.sampling_frequence,
                    max_length= dataset.max_length ,
                    flag=f'{counting_type}.s/wt{filter_time}',
                    wandb_key=f'metrix_PS' if args.task.upload_to_wandb else None,
                    save_path=os.path.join(save_root, f'{infer_mode}_{counting_type}_S_metrix.ws{filter_time}.png'),verbose=False)
            results.append([counting_type, max_filter_time, 's'] + list(result.values()))
            #print("="*40)
    
    counting_type = 'prob'
    max_filter_time = 'max'
    phasepicking_strategy = PhasePickingStratagy_Prob(tri_th_l=70, tri_th_r=70, expansion=args.model.Embedding.resolution, output_one=True, offset=0)
    ppicks, spicks = phasepicking_strategy(alldata['status']['pred'])
    result = metrix_PS(pred=ppicks, target=p_arrival_samples,
                    freq= args.DataLoader.Dataset.Resource.sampling_frequence,
                    max_length= dataset.max_length ,
                    flag=f'{counting_type}.p/wt{filter_time}',
                    wandb_key=f'metrix_PS' if args.task.upload_to_wandb else None,
                    save_path=os.path.join(save_root, f'{infer_mode}_{counting_type}_P_metrix.ws{filter_time}.png'),verbose=True)
    results.append([counting_type, max_filter_time, 'p'] + list(result.values()))
    result= metrix_PS(pred=spicks, target=s_arrival_samples,
                    freq= args.DataLoader.Dataset.Resource.sampling_frequence,
                    max_length= dataset.max_length ,
                    flag=f'{counting_type}.s/wt{filter_time}',
                    wandb_key=f'metrix_PS' if args.task.upload_to_wandb else None,
                    save_path=os.path.join(save_root, f'{infer_mode}_{counting_type}_S_metrix.ws{filter_time}.png'),verbose=True)
    results.append([counting_type, max_filter_time, 's'] + list(result.values()))
    out_df = pd.DataFrame(results, columns=['counting_type', 'filter_window', 'phase_type'] + list(result.keys()))
    print(out_df)

def plot_evaluate(args:ProjectConfig,save_data_root):
    #print_namespace_tree(args)
    #save_root    = os.path.join(args.output_dir,'visualize')
    #save_data_root = os.path.join(save_root, 'data')
    infer_mode   = args.task.infer_mode
    dataset_unit = 1.0/args.DataLoader.Dataset.Resource.sampling_frequence
    alldata, metadata, dataset, save_root = load_pred_data(args, save_data_root)

    if args.task.do_fasttest_eval and 'kvflow' in alldata:
        print(f"Detected You are Using do_fasttest_eval=True, this usually means your want to plot the kvflow")
        kvflow_monitor_plot(args, dataset, alldata['kvflow']['pred'][5],save_path=os.path.join(save_root, f'kvflow.slot5.png'))
        kvflow_monitor_plot(args, dataset, alldata['kvflow']['pred'][1],save_path=os.path.join(save_root, f'kvflow.slot1.png'))
        kvflow_monitor_plot(args, dataset, alldata['kvflow']['pred'][3],save_path=os.path.join(save_root, f'kvflow.slot3.png'))
        return         
    
    alldata = align_and_complete(args, alldata, dataset)

    if isinstance(args.DataLoader.Dataset,ConCatDatasetConfig):
        plot_for_continue_quake(args, dataset, alldata, save_root, infer_mode)
        return 
    
    alldata = metric_computing(args, alldata, dataset)

    single_metric_ploting(args, dataset, alldata, save_root, infer_mode)
    #raise
    
    if infer_mode in ['normal']:
        single_metric_dependency(args, dataset, alldata, save_root, infer_mode)
    
    if infer_mode in ['recurrent','RecPad3000'] and ('findP' in alldata or 'findS' in alldata) : 
        findP_series_plotting(args, dataset, alldata, save_root, infer_mode)
    
    if infer_mode in ['recurrent','RecPad3000'] and ('probabilityPeak' in alldata or 'P_Peak_prob' in alldata):
        probabilityPeak_series_plotting(args, dataset, alldata, save_root, infer_mode)
    
    if infer_mode in ['recurrent','RecPad3000'] and 'status' in alldata and not isinstance(args.DataLoader.Dataset,GroupDatasetConfig):
        status_evaluation_plotting(args, dataset, alldata, save_root, infer_mode)

def calculate_longterm_score(args, sequence_limit_length, psw_collect,sorted_list,p_position_map_pool): ### leak the code for S 
    

    part_nums = len(sorted_list)+1
    p_position_map_region = {part:{} for part in range(part_nums)}

    for slot, pset in p_position_map_pool.items():
        pset = [p  for p in  pset]
        for part in range(part_nums):p_position_map_region[part][slot] = []
        for part, p in zip(np.searchsorted(sorted_list,pset),pset):
            p_position_map_region[part][slot].append(p)
        for part in range(part_nums):
            if len(p_position_map_region[part][slot])==0:
                p_position_map_region[part][slot].append(-1)
    result_for_region = {}
    for region in range(len(p_position_map_region)-1):

        result_for_region[region] = metrix_PS(pred=p_position_map_region[region], target=psw_collect[region]['p'],
                                            freq= args.DataLoader.Dataset.Resource.sampling_frequence,
                                            max_length= sequence_limit_length,
                                            flag=f'test',verbose=False)
    return result_for_region

def plot_error_dependence(error_depend_data, depend_keys, intervel_keys, save_root, infer_mode=""):

    df = pd.DataFrame(error_depend_data)
    heatmap = False
    error_keys = [k for k in error_depend_data.keys() if k not in depend_keys]
    for error_key in error_keys:
        for depend_key in depend_keys:
            if not heatmap:continue
            print(f"now plot error_dependence diagram for {error_key}_vs_{depend_key}")
            error_dependence_diagram(df, error_key, depend_key, os.path.join(save_root, f'{infer_mode+error_key}_vs_{depend_key}.png'))
        for depend_key in intervel_keys:
            error_intervel_hist_diagram(df, error_key, depend_key, os.path.join(save_root, f'{infer_mode+error_key}_vs_{depend_key}_hist.png'))
     
def read_dict_of_numpy(saved_pool, npy_path):
    keysequence = os.path.split(npy_path)[-1].split('.')[:-1]
    order = int(keysequence[0].replace('infer_result_GPU', ""))
    keysequence = keysequence[1:]
    temp = saved_pool
    for i in range(len(keysequence)-1):
        key = keysequence[i]
        if key not in temp:
            temp[key] = {}
        temp = temp[key]
    final_key = keysequence[-1]
    if final_key not in temp:
        temp[final_key] = []
    while len(temp[final_key]) <= order:
        temp[final_key].append(None)
    temp[final_key][order] = np.load(npy_path)


def save_dict_of_numpy(dict_of_numpy, path):
    for key, val in dict_of_numpy.items():
        if isinstance(val, dict):
            save_dict_of_numpy(val, path+f".{key}")
        else:
            assert isinstance(val, np.ndarray)
            np.save(path+f".{key}", val)


def get_chunked_batch(batch, chunk_start, chunk_end):
    if isinstance(batch, torch.Tensor):
        return batch[chunk_start:chunk_end]
    elif isinstance(batch, dict):
        return {key: get_chunked_batch(val, chunk_start, chunk_end) for key, val in batch.items()}
    elif isinstance(batch, list):
        return [get_chunked_batch(val, chunk_start, chunk_end) for val in batch]
    else:
        raise NotImplementedError


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
                output_dict[key] = np.stack(
                    value, axis=dim) if mode == 'stack' else np.concatenate(value, axis=dim)
            else:
                output_dict[key] = np.array(value)
        else:
            output_dict[key] = value
    return output_dict


def print_dict_of_numpy_shape(input_dict):
    for key, val in input_dict.items():
        if isinstance(val, np.ndarray):
            print(f"{key}=>{val.shape}")
        elif isinstance(val, dict):
            for k, v in val.items():
                print(f"{key}.{k}=>{v.shape}")

def dummy_tqdm(x,*args,**kargs):return x

from trace_utils import print0
import  scipy
def get_evaluate_detail(
    model: torch.nn.Module,
    dataloader: DataLoader,
    config: EvaluatorConfig
):
    infer_mode = config.infer_mode
    model = model.eval()
    local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
    ProgressBar = tqdm if local_rank == 0 else dummy_tqdm
    if isinstance(config, RecurrentInferenceMode):# in ['recurrent','RecPad3000']:
        #origin_max_length = dataloader.dataset.max_length
        #dataloader.dataset.max_length = 6000
        # dataloader.dataset.warning_window = 0
        #chunk_size = origin_max_length if config.recurrent_chunk_size is None else config.recurrent_chunk_size
        chunk_size = config.recurrent_chunk_size
        print0(f"""[BETA FEATURE: This may change in the future.] 
                    This is [Recurrent mode]"
                    We force set the slide_stride from dataset as 0, which means the true start of the raw data.
                    We force set the warning_window from dataset as 0, which means the true start of the raw data This may change in the future.
                    In this mode, we directly obtain labels from the raw data, those may get slightly different metrix to the runtime case.
                    The sampling method is {dataloader.dataset.start_point_sampling_strategy} and the early warning stamp is {dataloader.dataset.early_warning}.
                    Notice: The origin trained length of the model maybe any number 3000 or 6000, check is by yourself!
                    You will force retreive a sequence [max length]={dataloader.dataset.max_length}, and use chunk size = {chunk_size} for recurrent """)
        unwrapper_model = model
        while hasattr(unwrapper_model,'module'):
            unwrapper_model = unwrapper_model.module
        model = unwrapper_model
    error_tracker = {}
    error_records = {}
    with torch.inference_mode():
        for batch_id, batch in ProgressBar(enumerate(dataloader), desc="Iter the dataset", total=len(dataloader), position=0):
            if isinstance(config, NormalInferenceMode):
                idx = batch.pop('idx')
                batch_output = model(get_prediction='pred+real', **batch)
                prediction   = batch_output['prediction']
                error_record = batch_output['error_record']
                prediction['idx'] = idx.detach().cpu().numpy()            
            elif isinstance(config, SlideInferenceMode):
                status_seq = batch['status_seq']
                waveform_seq = batch['waveform_seq']
                labels = batch['labels']
                prediction = {}
                error_record = {}
                for index, (status, waveform) in ProgressBar(enumerate(zip(torch.split(status_seq, 1, dim=1),torch.split(waveform_seq, 1, dim=1))), 
                                                    desc="Slide the window", position=1, leave=False):
                    new_label = {}
                    for key, val in labels.items():
                        if len(val.shape) < 2: ## the target is number 
                            new_label[key] = val
                        else:
                            new_label[key] = val[:, index]
                    batch_output = model(get_prediction='pred+real', status_seq=status[:, 0], waveform_seq=waveform[:, 0], labels=new_label)
                    dict_to_dict_of_lists(prediction, batch_output['prediction'])
                    dict_to_dict_of_lists(error_record, batch_output['error_record'])
                prediction   = dict_of_lists_to_numpy(prediction, dim=1, mode='stack')
                error_record = dict_of_lists_to_numpy(error_record, dim=0, mode='stack')
            elif isinstance(config, RecurrentInferenceMode):
                
                idx          = batch.pop('idx')
                status_seq   = batch.pop('status_seq')
                waveform_seq = batch.pop('waveform_seq')
                trend_seq    = batch.pop('trend_seq') if 'trend_seq' in batch else None
                labels       = batch.pop('labels') ## < --- we will not use label in this mode
                assert status_seq.shape[1] == waveform_seq.shape[1]
                #assert status_seq.shape[1] == 6000  
                start_status_seq  =  status_seq[..., :config.recurrent_start_size] #(B,L) for single station and (B, N, L) for multistaion 
                new_status_seq    =  status_seq[..., config.recurrent_start_size:] #(B,L) for single station and (B, N, L) for multistaion

                start_waveform_seq=waveform_seq[..., :config.recurrent_start_size,:] #(B,L,D) for single station and (B, N, L, D) for multistaion
                new_waveform_seq  =waveform_seq[..., config.recurrent_start_size:,:] #(B,L,D) for single station and (B, N, L, D) for multistaion
                
                if trend_seq is not None:
                    start_trend_seq = trend_seq[..., :config.recurrent_start_size,:]
                    new_trend_seq   = trend_seq[..., config.recurrent_start_size:,:]
                else:
                    start_trend_seq = None
                    new_trend_seq   = None
                findPSfeature = {'findP':None, 'findS':None}
                findfeature = None
                rnn_pred={}
                last_state, start_state,downstream_feature = model.generate_start_pointing(start_status_seq,start_waveform_seq,start_trend_seq=start_trend_seq,**batch)
                
                for findkey in ['findP','findS']:
                    if findkey in downstream_feature:
                        findPSfeature[findkey] = downstream_feature.pop(findkey) #(B, L, D)
                prediction = model.get_downstream_prediction(downstream_feature,**batch)

                for k,v in prediction.items():
                    if k not in rnn_pred:rnn_pred[k] = []
                    rnn_pred[k].append(v.detach().cpu().numpy())
                if last_state['monitor'] is not None and batch_id < config.save_monitor_num: ### this only save disk, but still slow when using kv_first mode. Thus, it is better use a independent branch for kvflow monitor
                    for k,v in last_state['monitor'].items():
                        if k not in rnn_pred:rnn_pred[k] = []
                        ## the value of monitor can be in a low precision to save the memory and disk
                        data = v.detach().cpu().numpy()
                        data = (data*100)
                        data[data>65_535] = 65_535
                        data = data.astype('uint16')
                        rnn_pred[k].append(data)
                chunk_nums = np.ceil(new_waveform_seq.shape[1]/chunk_size).astype('int')
                for i in ProgressBar(range(chunk_nums), desc="Iter the sequence", position=1, leave=False):
                    now_status_seq   =   new_status_seq[...,chunk_size*i:chunk_size*(i+1)]
                    now_waveform_seq = new_waveform_seq[...,chunk_size*i:chunk_size*(i+1),:]
                    now_trend_seq    =    new_trend_seq[...,chunk_size*i:chunk_size*(i+1),:] if trend_seq is not None else None
                    last_state, output_state, downstream_feature =model.generate_next(last_state,now_status_seq,now_waveform_seq,now_trend_seq=now_trend_seq,**batch)
                    for findkey in ['findP','findS']:
                        if findkey in downstream_feature:
                            findPSfeature[findkey] = torch.cat([findPSfeature[findkey],downstream_feature.pop(findkey)],1) #(B, L+l, D)
                    prediction = model.get_downstream_prediction(downstream_feature,**batch) ### modify here so it works for the multi station model
                    for k,v in prediction.items():
                        if k not in rnn_pred:rnn_pred[k] = []
                        rnn_pred[k].append(v.detach().cpu().numpy())
                        #print(f"{k}=>{v.shape}")
                    if last_state['monitor'] is not None and batch_id < config.save_monitor_num:
                        for k,v in last_state['monitor'].items():
                            if k not in rnn_pred:rnn_pred[k] = []
                            data = v.detach().cpu().numpy()
                            data = (data*100)
                            data[data>65_535] = 65_535
                            data = data.astype('uint16')
                            rnn_pred[k].append(data)
                rnn_pred    = dict_of_lists_to_numpy(rnn_pred,dim=1, mode='cat')
                
                for findkey in ['findP','findS']:
                    if findPSfeature[findkey] is not None:
                        ### for findS feature, we should use a slide window to obtain its value
                        findfeature = findPSfeature[findkey]
                        assert findfeature.shape[1]>=model.slide_feature_window_size, f"the {findkey} length is {findfeature.shape[1]}, but the slide_feature_window_size is {model.slide_feature_window_size}"
                        # config.recurrent_slide_output_strider is decrapted, use slide_stride_in_training instead
                        prediction = model.downstream_prediction({findkey:findfeature}) # (B, N, L+1)
                        Plocations = torch.argmax(prediction[findkey], dim=-1) - 1 # (B, N)
                        # Plocations = []
                        # for start_point in ProgressBar(range(0, findfeature.shape[1] - model.slide_feature_window_size + 1, 
                        #                                      config.recurrent_slide_output_strider), 
                        #                             desc="Slide iter the sequence", position=1, leave=False):
                        #     end_point = start_point + model.slide_feature_window_size
                        #     feature_now = findfeature[:,start_point:end_point]
                        #     downstream_feature={findkey:feature_now}
                        #     prediction = model.downstream_prediction(downstream_feature)
                        #     Plocation  = torch.argmax(prediction[findkey], dim=1) - 1 ## (B,) -1 means no P and other value means the P location in the window
                        #     Plocations.append(Plocation)
                        # Plocations = torch.cat(Plocations,-1) #(B, L - window_size)
                        rnn_pred[findkey] = Plocations.detach().cpu().numpy()

                prediction  = {}
                for k,v in rnn_pred.items():
                    if k in ['status','probabilityPeak','probability']: 
                        v = scipy.special.softmax(v,-1)
                        v = np.round(v*100).astype('int8') ## <-- made the prob to [0,100] and round to int8
                    if k in ['P_Peak_prob']:
                        v = np.round(scipy.special.expit(v)*100).astype('int8')
                    prediction[k] = {'pred': v}
                    if k in ['group_vector']: ## the group label must be recoreded since it is not efficient to recompute it in later phase. 
                        prediction[k]['real'] =  labels['group_vector'][batch['station_mask'].bool()].detach().cpu().numpy() ### each information 
                # if 'station_mask' in batch:
                #     prediction['idx'] = idx[batch['station_mask'].bool()].detach().cpu().numpy()      
                # else:
                # we dont flatten the 'idx' even in the multistation mode, since it can be a flag for recovery the multistation group information
                prediction['idx'] = idx.detach().cpu().numpy()      
                error_record = {}
            dict_to_dict_of_lists(error_tracker, prediction)
            dict_to_dict_of_lists(error_records, error_record)
            if batch_id > 10 and config.do_fasttest_eval:break
    error_tracker = dict_of_lists_to_numpy(error_tracker, dim=0, mode='concat')
    error_records = dict_of_lists_to_numpy(error_records, dim=0, mode='stack')
    #print_dict_of_numpy_shape(error_tracker)
    #print_dict_of_numpy_shape(error_records)

    for key in error_records.keys():
        if key in ['idx']:continue
        print(f"{key} ==> {error_records[key].shape} ==> {np.mean(error_records[key]):.4f}")
    return error_tracker
import os
from project_arguements import get_args
from train_single_station_via_llm_way import *
from evaluator.trace_evaluator import *
from tqdm.auto import tqdm
import sys
from pathlib import Path
from obspy.geodetics import gps2dist_azimuth
import obspy
from obspy import read_inventory,read_events
from obspy.core import read,Stream
import numpy as np
from tqdm.auto import tqdm
import obspy
from obspy import read_events
# Read your inventory which includes the orientation of the sensors
#inventory = read_inventory("/mnt/data/oss_beijing/zhangtianning/dataset/realworldearthquake")
import pickle
from obspy import read_events, read_inventory
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from geographiclib.geodesic import Geodesic

def get_input_data(self, waveform):
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



    return waveform,  runtimemean
def get_result(ShrinkedENZWaveform,dataset):
    waveform_seq,  trend_seq = get_input_data(dataset,ShrinkedENZWaveform)
    waveform_seq = waveform_seq.cuda().unsqueeze(0).float()
    trend_seq    = trend_seq.cuda().unsqueeze(0).float() if trend_seq is not None else None
    batch = {'idx': torch.LongTensor([0]),
            'status_seq': torch.zeros_like(waveform_seq[...,0]), 
            'waveform_seq': waveform_seq,
            'trend_seq':trend_seq,
            'labels':None}
    error_tracker = {}
    error_records = {}
    with torch.inference_mode():
        #for batch_id, batch in ProgressBar(enumerate(dataloader), desc="Iter the dataset", total=len(dataloader), position=0):
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
            assert status_seq is None or status_seq.shape[1] == waveform_seq.shape[1]
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
    return rnn_pred
def _resample(waveform, target_sampling_rate, source_sampling_rate,sample_axis, eps=1e-4):
        """
        Resamples waveform from source to target sampling rate.
        Automatically chooses between scipy.signal.decimate and scipy.signal.resample
        based on source and target sampling rate.

        :param waveform:
        :param target_sampling_rate:
        :param source_sampling_rate:
        :param eps: Tolerance for equality of source an target sampling rate
        :return:
        """
        if 1 - eps < target_sampling_rate / source_sampling_rate < 1 + eps:
            return waveform
        else:
            if sample_axis is None:
                raise ValueError(
                    "Trace can not be resampled because of missing or incorrect dimension order."
                )

            if waveform.shape[sample_axis] == 0:
                seisbench.logger.info(
                    "Trying to resample empty trace, skipping resampling."
                )
                return waveform

            if (source_sampling_rate % target_sampling_rate) < eps:
                q = int(source_sampling_rate // target_sampling_rate)
                return scipy.signal.decimate(waveform, q, axis=sample_axis)
            else:
                num = int(
                    waveform.shape[sample_axis]
                    * target_sampling_rate
                    / source_sampling_rate
                )
                return scipy.signal.resample(waveform, num, axis=sample_axis)


THEPATH=sys.argv[1]
flag = sys.argv[2]#"TW20240409"
#preprocess_mode = sys.argv[3]
ConfigPath=os.path.join(THEPATH,"train_config.json")
theweightpath = list(Path(THEPATH).glob("pytorch_model.bin"))
if len(theweightpath) ==0:
    theweightpath = list(Path(THEPATH).glob("model.safetensors"))
assert len(theweightpath) > 0
theweightpath = theweightpath[0]

 
argcommend = ["--task","recurrent_infer", "--Resource", "STEAD", "--resource_source", "stead.trace.BDLEELSSO.hdf5", "--preload_weight", str(theweightpath)]
with open(ConfigPath,'r') as f:
    config_fast = json.load(f)
if config_fast.get('noise_config_tracemapping_path'):
    argcommend += [ '--NoiseGenerate', 'pickalong_receive']
args = get_args(config_path=ConfigPath,args=argcommend)
figure_path = f"figures/realworlddata/{flag}"
model = load_model(args.model)
if args.task.Checkpoint.preload_weight:
    printg(f"LOADING MODEL from {args.task.Checkpoint.preload_weight}")
    unwrapper_model = model
    while hasattr(unwrapper_model,'module'):
        unwrapper_model = unwrapper_model.module
    weight = smart_read_weight_path(args.task.Checkpoint.preload_weight, 'cpu') #
    smart_load_weight(unwrapper_model, weight,
                      strict=not args.task.Checkpoint.load_weight_partial, shape_strict=not args.task.Checkpoint.load_weight_ignore_shape)
    needed_dataset = args.task.eval_dataflag if isinstance(args.task, EvaluatorConfig) else None
train_dataloader, valid_dataloader, test_dataloader = load_dataloader(args.DataLoader, 
                                                                      downstream_pool=args.model.Predictor.downstream_pool, 
                                                                      sampling_strategy=args.task.sampling_strategy,
                                                                      infer=False,
                                                                      )
#valid_dataloader.dataset.set_padding_length(3000)
dataloader = valid_dataloader
config = args.task
model = model.cuda().eval()
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

with open(f'/mnt/data/oss_beijing/zhangtianning/dataset/realworldearthquake/{flag}/grouped_st.aligned.pkl', 'rb') as f:origin_st = pickle.load(f)
with open(f'/mnt/data/oss_beijing/zhangtianning/dataset/realworldearthquake/{flag}/grouped_st.response.aligned.pkl', 'rb') as f:response = pickle.load(f)
catalog = read_events(f'/mnt/data/oss_beijing/zhangtianning/dataset/realworldearthquake/{flag}/quakeevent.xml')
event = catalog[0]
print(catalog)

for mag in event.magnitudes:
    if mag.magnitude_type =='Mwr':
        break
magnitude = mag.mag
print(magnitude)

for preprocess_mode in ['L','LD','D',#'SL','S','SD','SLD'
                        ]:
    # ===========> filter station
    processed_stes={}
    processed_stes_temp={}
    for name,st in origin_st.items():
        
        response_list = response[name]
        processed_stes_temp[name]=[]
        notENZcase  = False
        for tr, resp in zip(st, response_list):
            tr  = tr.copy()
            if tr.stats.channel[-1] == "1":
                print(f"skip [{name}] due to it is `12Z` mode")
                notENZcase = True
            if notENZcase:continue
            # Event coordinates (replace with actual values)
            event_lat   = event.origins[0].latitude
            event_lon   = event.origins[0].longitude

            # Station coordinates (replace with actual values)
            for n in response[name][0]:
                for s in n:
                    station_lat = s.latitude
                    station_lon = s.longitude
            distance_m, azimuth, back_azimuth = obspy.geodetics.base.gps2dist_azimuth(
                                                                            float(event_lat), 
                                                                            float(event_lon),
                                                                            float(station_lat), 
                                                                            float(station_lon), 
                                                                            a=6378137.0, 
                                                                            f=0.0033528106647474805)

            distance_km = distance_m/1000
            if distance_km > 100:
                print(f"skip by far away: d={distance_km}")
                continue
            if 'S' in preprocess_mode:
                tr.remove_response(inventory=resp, output="ACC")
            if 'L' in preprocess_mode:
                tr.detrend("linear")
            if 'D' in preprocess_mode:
                tr.detrend('demean')
            #tr.detrend("spline", order=3, dspline=500)
            processed_stes_temp[name].append(tr)
        if len(processed_stes_temp[name])!=3:continue
        processed_stes[name] = [Stream(processed_stes_temp[name]),distance_km, azimuth, back_azimuth,station_lat,station_lon,event_lat,event_lon]
    print("totally remain station:", len(processed_stes))    

    # ==================>  compute result 
    names = list(processed_stes.keys())
    #plt.plot(processed_stes[name][1].data)
    results = {}
    for name in names:
        st,distance_km, azimuth, back_azimuth,station_lat,station_lon,event_lat,event_lon = processed_stes[name]
        start_times  = [tr.stats.starttime for tr in st]
        end_times    = [tr.stats.endtime for tr in st]
        latest_start = max(start_times)
        earliest_end = min(end_times)
        print(st)
        # Trim (or pad with zeros) the traces to the common time window
        st.trim(starttime=latest_start, endtime=earliest_end, pad=True, fill_value=0)
        NWAV_E= _resample(st[0].data,100,st[0].stats.sampling_rate, 0)
        NWAV_N= _resample(st[1].data,100,st[1].stats.sampling_rate, 0)
        NWAV_Z= _resample(st[2].data,100,st[2].stats.sampling_rate, 0)
        ENZWaveform = np.stack([NWAV_E, NWAV_N,NWAV_Z],-1)[:6000]
        #ENZWaveform = np.stack([NWAV_Z, NWAV_Z,NWAV_Z],-1)[:4000]
        padding_length = int(3*np.ceil(len(ENZWaveform)/3)- len(ENZWaveform))
        ShrinkedENZWaveform = np.pad(ENZWaveform,((padding_length,0),(0,0)))

        ### run================
        rnn_pred = get_result(ShrinkedENZWaveform,dataloader.dataset)
        results[name] = [rnn_pred,ShrinkedENZWaveform]+processed_stes[name][1:] 

        # print(f"Distance between event and station: {distance_km:.2f} km")
        # print(f"Back azimuth: {back_azimuth:.2f} degrees")
        # print(f"azimuth: {azimuth:.2f} degrees")

    def get_event_coordinates(station_lat, station_lon, distance_km, back_azimuth):
        geod = Geodesic.WGS84
        result = geod.Direct(station_lat, station_lon, back_azimuth, distance_km * 1000)
        return result['lon2'], result['lat2']

    plot_names = names[:]
    # ===============> plot
    from matplotlib.ticker import AutoLocator
    from matplotlib.ticker import MultipleLocator


    fig, axes = plt.subplots(len(plot_names),1 , figsize=(12, 8))
    axes = axes.flatten() if len(plot_names)>1 else [axes]
    for name, ax1  in zip(plot_names, axes):
        rnn_pred,waveform, distance_km, azimuth, back_azimuth,station_lat,station_lon,event_lat,event_lon = results[name]
        axis = np.arange(len(rnn_pred['magnitude'][0,:,0]))*0.03
        real_x = distance_km* np.cos(azimuth/180*np.pi)# we actually use the azimuth
        real_y = distance_km* np.sin(azimuth/180*np.pi)# we actually use the azimuth
        x = rnn_pred['x'][0,:,0]*args.model.Predictor.downstream_pool['x']['normlizer']['unit']
        y = rnn_pred['y'][0,:,0]*args.model.Predictor.downstream_pool['y']['normlizer']['unit']
        shift  = np.sqrt((x - real_x)**2 + (y-real_y)**2)
        shift2 = np.sqrt((np.abs(x) - np.abs(real_x))**2 + (np.abs(y)-np.abs(real_y))**2)
        time = np.arange(len(waveform))*0.01
        # Plot the first dataset on ax1, using the left y-axis
        ax1.plot(time, ShrinkedENZWaveform, 'g-')  # 'g-' is a green solid line
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Waveform', color='g')
        ax1.tick_params(axis='y', labelcolor='g')

        # Create the second set of axes that shares the same x-axis
        ax2 = ax1.twinx()

        # Plot the second dataset on ax2, using the right y-axis
        ax2.plot(axis, shift, 'b-')  # 'b-' is a blue solid line
        ax2.plot(axis, shift2, '-',color='gray')  # 'b-' is a blue solid line
        ax2.axvline(10, color='r')
        ax2.set_ylabel('Shift Error', color='b')
        ax2.set_title(f'{args.trial_name} ==> event:[({event_lat:.2f},{event_lon:.2f})] station: [{name}({station_lat:.2f},{station_lon:.2f})] D={distance_km:.1f}km')
        ax2.tick_params(axis='y', labelcolor='b')
        locator = AutoLocator()
        ax2.yaxis.set_major_locator(MultipleLocator(30))  # Sets a tick every 0.5 units

    name=f"{flag}-{args.trial_name}-{preprocess_mode}"
    
    os.makedirs(figure_path,exist_ok=True)
    fig.savefig(f'{figure_path}/{name}_shift.png')

    event_lon = event.preferred_origin().longitude
    event_lat = event.preferred_origin().latitude
    show_names = names[0:3]
    fig = plt.figure(figsize=(10, 8))
    set_colors=['green','yellow','pink']
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    # Add map features (e.g., coastlines, borders)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.OCEAN, color='lightblue')
    ax.add_feature(cfeature.LAND, color='lightgray')

    # map_extent = [event_lon- 1.5, event_lon + 0.5,
    #               event_lat- 1.2, event_lat + 1.4]
    # ax.set_extent(map_extent, crs=ccrs.PlateCarree())
    ax.plot(event_lon, event_lat, marker='*', color='red', markersize=10, transform=ccrs.PlateCarree())
    for show_name,point_color in zip(show_names,set_colors):
        
        rnn_pred,waveform, distance_km, azimuth, back_azimuth,station_lat,station_lon,event_lat,event_lon = results[show_name]
        axis = np.arange(len(rnn_pred['magnitude'][0,:,0]))*0.03
        real_x =  distance_km* np.cos(azimuth/180*np.pi)# we actually use the azimuth
        real_y =  distance_km* np.sin(azimuth/180*np.pi)# we actually use the azimuth
        pred_x =  rnn_pred['x'][0,:,0]*args.model.Predictor.downstream_pool['x']['normlizer']['unit']
        pred_y =  rnn_pred['y'][0,:,0]*args.model.Predictor.downstream_pool['y']['normlizer']['unit']
        time=axis- 10

        selected_index = np.concatenate([
        #np.where(time<0)[0][::50],
        np.where(np.logical_and(time>0,time<3))[0][::10],
        np.where(time>3)[0][::50],
        ]
        )

        x = pred_x[selected_index]
        y = pred_y[selected_index]
        T =   time[selected_index] 
        distances     =  np.sqrt(x**2+y**2)
        angles        =  np.rad2deg(np.arctan2(y, x)) + 180
        real_distance =  np.sqrt(real_x**2+real_y**2)
        real_angle    =  np.rad2deg(np.arctan2(real_y, real_x)) + 180
        real_lon, real_lat = get_event_coordinates(station_lat, station_lon, real_distance, real_angle)
        pred_lonlats = [get_event_coordinates(station_lat, station_lon, d, baz) for d,baz in zip(distances, angles)]
        predicted_lons=np.array([a for a,b in pred_lonlats])
        predicted_lats=np.array([b for a,b in pred_lonlats])
        
        ax.plot(station_lon, station_lat, marker='^', color=point_color, markersize=20, transform=ccrs.PlateCarree())

        
        #ax.scatter(real_lon, real_lat,marker='*', color='red', s=100, transform=ccrs.PlateCarree(), label='Predicted Quake')
        
        
        ooooi=0
        offset_x=[-20, -20, -40, 20,  10,-70, 0, 20]
        offset_y=[ 25,  25,  25, 25,  35, 25, 55, 25]
        for i in range(len(predicted_lons) - 1):
            time_now = T[i]
            dx = predicted_lons[i+1] - predicted_lons[i]
            dy = predicted_lats[i+1] - predicted_lats[i]
            linestyle = '--' #if time_now < 0 else "-"
            alpha = 0.3 if time_now < 0 else 1
            arrow_length = 0.9  # Adjust this value to control the arrow length
            

            annotation = f"T={time_now:0.1f}s"
            if time_now<0:continue
            if time_now>2:continue

    #         ax.annotate(annotation, (predicted_lons[i], predicted_lats[i]),
    #                     xytext=(offset_x[ooooi], offset_y[ooooi]), textcoords='offset points',
    #                     fontsize=10, color='black',
    #                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2"),
    #                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))
            ooooi +=1
            alphas = np.ones_like(T)
            alphas[T<0]=0.3
            ax.scatter(predicted_lons, predicted_lats, alpha=alphas, marker='.', color=point_color, s=100, transform=ccrs.PlateCarree(), label='Predicted Quake')
            ax.arrow(predicted_lons[i], predicted_lats[i], dx * arrow_length, dy * arrow_length,
                    transform=ccrs.PlateCarree()._as_mpl_transform(ax),
                    head_width=0.05, head_length=0.03, fc='black', ec='black',
                    length_includes_head=True,linestyle=linestyle,alpha=alpha)
    gl=ax.gridlines(draw_labels=True,linestyle=":",linewidth=0.3,color='k')
    gl.xlabel_style = {'size': 15, 'color': 'black'}
    gl.ylabel_style = {'size': 15, 'color': 'black'}
    gl.top_labels = False
    gl.right_labels = False
    plt.title('Event and Station Location')
    plt.xlabel('Longitude')

    name=f"{flag}-{args.trial_name}-{preprocess_mode}"
    fig.savefig(f'{figure_path}/{name}_earth.png')

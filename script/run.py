from evaluator.trace_evaluator import PhasePickingStratagy_Status,  StatusDeterminer_Threshold, metrix_PS

def process_setting(name, status_pred, p_arrival_samples):
    frequency = 100
    resolution= 3
    (threshold, judger, max_filter_time) = name
    max_filter_window_size = int(max_filter_time * frequency // resolution)
    if max_filter_window_size % 2 == 0:
        max_filter_window_size += 1
    filter_time = max_filter_window_size * resolution / frequency

    phasepicking_strategy = PhasePickingStratagy_Status(
        expansion=resolution, 
        status_strategy=StatusDeterminer_Threshold(p_threshold=threshold, s_threshold=threshold),
        windows_size=max_filter_window_size, judger=judger, timetype='all'
    )

    ppicks_pool, spicks_pool = phasepicking_strategy(status_pred)
    result_dict = {}
    for counting_type in ppicks_pool.keys():
        ppicks = ppicks_pool[counting_type]
        spicks = spicks_pool[counting_type]
        result = metrix_PS(
            pred=ppicks, target=p_arrival_samples,
            freq=100,
            max_length=9000,
            flag=f'{counting_type}.p/wt{filter_time}', verbose=False
        )
        result_dict[counting_type] = result
    return result_dict


if __name__ == '__main__':
    import sys, os, json
    import numpy as np
    from tqdm.auto import tqdm    

    assert len(sys.argv)==3, "Usage: python create_cluster_for_each_station.py split_num split_idx"
    split_num = int(sys.argv[1])
    split_idx = int(sys.argv[2])
    assert split_idx < split_num   
    thresholds = np.arange(50, 99, 5)
    judgers    = np.linspace(0.9, 1, 10)
    max_fileter_times= np.linspace(0.1, 1, 10)
    setting=[]


    for threshold in thresholds:
        for judger in judgers:
            for max_filter_time in max_fileter_times:
                setting.append((threshold, judger, max_filter_time))
    index_range = np.linspace(0, len(setting), split_num+1).astype('int')
    split = split_idx
    start = index_range[split]
    end   = index_range[split+1]
    target_path = f"debug/split/{start:08d}_{end:08d}.json"
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    if os.path.exists(target_path):exit()
    status_pred       = np.load('debug/status_pred.npy')
    p_arrival_samples = np.load('debug/p_arrival_samples.npy')

    results = {}
    for i in tqdm(range(start ,end)):
        name = setting[i]
        result =  process_setting(name, status_pred, p_arrival_samples)
        name = ",".join([str(t) for t in name])
        results[name] = result

    with open(target_path, 'w') as f:
        json.dump(results, f)
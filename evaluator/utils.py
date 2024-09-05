
import torch
import numpy as np
from geopy.distance import geodesic
import pandas as pd
from obspy.signal.trigger import trigger_onset
from scipy.ndimage import convolve1d
from collections import defaultdict
def findAllPS(pred_status,ps_win, expansion, judger=0.8, timetype='all'):
    """
        window_size = 7
        kernel      = np.ones(window_size)
        a = np.arange(12)
        b = convolve1d(a, kernel, mode='nearest',origin=-(window_size//2))
        print(a)
        print(b)
    """
    pred_status_is2 = (pred_status==2).astype('int')
    pred_status_is1 = (pred_status==1).astype('int')   
    window_size = int(ps_win)
    kernel      = np.ones(window_size)
    ##use orgin to make sure the left hand padding
    if window_size == 1: # skip conv if kernel is 1
        pred_status_is2_status_pool = {'realtime':pred_status_is2==2}
        pred_status_is1_status_pool = {'realtime':pred_status_is1==1}
    else:
        pred_status_is2_status_pool = {}
        pred_status_is1_status_pool = {}
        if timetype in ['all','realtime']:
            pred_status_is2_status_pool['realtime'] = convolve1d(pred_status_is2, kernel, mode='nearest',origin=-(window_size//2)) > (window_size*judger) # <-- larger than 0.5 means more than half is 2
            pred_status_is1_status_pool['realtime'] = convolve1d(pred_status_is1, kernel, mode='nearest',origin=-(window_size//2)) > (window_size*judger) # <-- larger than 0.5 means more than half is 1
        if timetype in ['all','posttime']:
            pred_status_is2_status_pool['posttime'] = convolve1d(pred_status_is2, kernel, mode='nearest',origin=+(window_size - 1)//2) > (window_size*judger) # <-- larger than 0.5 means more than half is 2
            pred_status_is1_status_pool['posttime'] = convolve1d(pred_status_is1, kernel, mode='nearest',origin=+(window_size - 1)//2) > (window_size*judger) # <-- larger than 0.5 means more than half is 1
        
    s_position_map_pool = {}
    p_position_map_pool = {}
    for key in pred_status_is2_status_pool.keys():
        # pred_status_new = pred_status.copy()
        # pred_status_new[pred_status_is1_status]=1
        # pred_status_new[pred_status_is2_status]=2
        pred_status_is2_status = pred_status_is2_status_pool[key]
        pred_status_is1_status = pred_status_is1_status_pool[key]
        pred_status_is2_jumper = np.logical_and(~pred_status_is2_status[...,:-1],pred_status_is2_status[...,1:]) #[right]
        pred_status_is1_jumper = np.logical_and(~pred_status_is1_status[...,:-1],pred_status_is1_status[...,1:]) #[right]

        p_position_map = defaultdict(set)
        rows, cols  = np.where(pred_status_is1_jumper)
        for row, col in tqdm(zip(rows, cols * expansion), total=len(rows), leave=False):
            p_position_map[row].add(col)
        p_position_map = dict(p_position_map)
        
        rows, cols  = np.where(pred_status_is2_jumper)
        s_position_map = defaultdict(set)
        for row, col in tqdm(zip(rows, cols * expansion), total=len(rows), leave=False):
            s_position_map[row].add(col)
        s_position_map = dict(s_position_map)
            
        s_position_map_pool[key] = s_position_map
        p_position_map_pool[key] = p_position_map
    return p_position_map_pool, s_position_map_pool

def picking(pred, tri_th_h, tri_th_l):
    """
    tri_th_h max_value
    tri_th_l min_value
    """
    trigs = trigger_onset(pred, tri_th_h, tri_th_l)
    picks = []
    for tri_on, tri_off in trigs:
        if tri_on == tri_off:
            continue
        pick = np.argmax(pred[tri_on:tri_off+1]) + tri_on
        picks.append(pick)
    return picks

from tqdm.auto import tqdm
def findAllP_Peak(batchdata, tri_th_h, tri_th_l,expansion, output_one=False,offset=0):
    ppick = {}
    for row, data in tqdm(enumerate(batchdata), total=len(batchdata), leave=False):
        p_picking = picking(data,tri_th_h, tri_th_l )
        if len(p_picking)>0:
            if output_one:p_picking = [p_picking[np.argmax(data[p_picking])]]
            p_picking = [t*expansion + offset for t in p_picking]
            ppick[row] = set(p_picking)
    
    return ppick
        
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


def calc_loc_stats(loc_pred: torch.Tensor, event_metadata: pd.DataFrame, pos_offset: list):
    coord_keys = detect_location_keys(event_metadata.columns)
    true_coords = torch.Tensor(event_metadata[coord_keys].values)

    mean_coords = torch.sum(loc_pred[:, :, :1] * loc_pred[:, :, 1:4], dim=1)

    mean_coords *= 100
    D2KM = 1.0  # Please replace this with your actual util.D2KM value
    mean_coords[:, :2] /= D2KM
    mean_coords[:, 0] += pos_offset[0]
    mean_coords[:, 1] += pos_offset[1]

    dist_epi = torch.zeros(len(mean_coords))
    dist_hypo = torch.zeros(len(mean_coords))
    for i, (pred_coord, true_coord) in enumerate(zip(mean_coords, true_coords)):
        dist_epi[i] = geodesic(pred_coord[:2].numpy(),
                               true_coord[:2].numpy()).km
        dist_hypo[i] = np.sqrt(dist_epi[i].item() ** 2 +
                               (pred_coord[2] - true_coord[2]).item() ** 2)

    rmse_epi = torch.sqrt(torch.mean(dist_epi ** 2))
    mae_epi = torch.mean(torch.abs(dist_epi))

    rmse_hypo = torch.sqrt(torch.mean(dist_hypo ** 2))
    mae_hypo = torch.mean(dist_hypo)

    return rmse_hypo, mae_hypo, rmse_epi, mae_epi

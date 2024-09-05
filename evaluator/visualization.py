import matplotlib.ticker as mtick
from matplotlib.ticker import FuncFormatter
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from scipy import stats
from mltool.visualization import *
import os
from sklearn.linear_model import LogisticRegression
import wandb
import scipy
from .utils import findAllPS

def compare_the_distribution(pred_result, should_result, root_path = "debug", epoch=0):
    # magnitude part 
    mag_pred = pred_result['magnitude']
    magnitude_predic = np.sum(mag_pred[:, :, 0] * mag_pred[:, :, 1], axis=1)
    magnitude_should = should_result['magnitude'][:,0,0]
    fig = plt.figure()
    smoothhist(magnitude_predic,color='g',label='pred')
    smoothhist(magnitude_should,color='r',label='groud')
    plt.legend()
    save_dir = os.path.join(root_path, 'trace', 'magnitude_distribution')
    if not os.path.exists(save_dir):os.makedirs(save_dir)

    plt.savefig(os.path.join(save_dir,f"epoch-{epoch:03d}.png"))
    fig.clear()

    location_predic = pred_result['location']
    # location_predic = loc_pred[..., 1:4]  # <------- notice this
    # if len(location_predic.shape) == 3:location_predic = location_predic.sum(1)  # <------- notice this
    
    location_should = should_result['location'][...,0] #(B,3,1)->(B,3)
    
    for i in range(3):
        fig = plt.figure()
        predic = location_predic[...,i]
        should = location_should[..., i]
        smoothhist(predic,color='g',label='pred')
        smoothhist(should,color='r',label='groud')
        plt.legend()
        save_dir = os.path.join(root_path, 'trace', f'coord_{i}')
        if not os.path.exists(save_dir):os.makedirs(save_dir)
        plt.xlim([-1,1])
        plt.savefig(os.path.join(save_dir,f"epoch-{epoch:03d}.png"))
        fig.clear()
    plt.clf()

import scipy.stats as stats
from scipy.stats import gaussian_kde
def gauss_mimic_smoothhist(data,ax=None,**kargs):
    density = gaussian_kde(data)
    xs = np.linspace(min(data),max(data),200)
    density.covariance_factor = lambda : .25
    density._compute_covariance()
    x = xs
    y = density(xs)
    y = y/y.max()
    mean = data.mean()
    std  = data.std()
    fake_y = stats.norm.pdf(x, mean, std)
    fake_y = fake_y/fake_y.max()
    handle = ax if ax is not None else plt
    handle.plot(x,y,**kargs)
    handle.plot(x,y,'b',label='data')
    handle.plot(x,fake_y,'r',label='gauss')
    handle.legend()
    mean_std_text = f'Mean: {mean:.2f}, Std: {std:.2f}'
    handle.text(0.05, 0.95, mean_std_text, transform=handle.transAxes, fontsize=12, verticalalignment='top')

def topepi(dist_epi,root_path,epoch   = 0,**kargs):
    fig,axes = plt.subplots(2,3, figsize=(12,6))
    axes = axes.flatten()
    dist_epi = np.sort(dist_epi)
    toplist = [500,1000,5000,10000,50000,100000]
    for ax,top in zip(axes,toplist):
        data = dist_epi[:top]
        smoothhist(data,color='r',label='pred',ax=ax)
        ax.legend()
        mae   = np.mean(abs(data))
        max_v = np.max(abs(data))
        min_v = np.min(abs(data))
        mean_std_text = f'Top{top}==> \nMAE: {mae:.2f}, \nMAX: {max_v:.2f}, \nMIN:{min_v:.2f}'
        ax.text(0.1, 0.95, mean_std_text, transform=ax.transAxes,fontsize=12, verticalalignment='top')
    key     = "epi"
    channel = "test"
    save_dir = os.path.join(root_path, 'trace', f'{key}_{channel}')
    if not os.path.exists(save_dir):os.makedirs(save_dir)
    plt.suptitle(f'{key}_{channel}')
    fig.savefig(os.path.join(save_dir, f"epoch-{epoch:03d}.png"))
    fig.clear()
    plt.clf()      


def deltaepi(dist_epi, should_coords, predic_coords, root_path, epoch=0, **kargs):
    fig,axes = plt.subplots(2,3, figsize=(12,6))
    limits = [50,100,200]
    for i,limit in enumerate(limits):
        for coord in [0,1]:
            ax = axes[coord,i]
            order = np.argsort(should_coords[dist_epi>limit][:,coord])
            ax.plot(should_coords[dist_epi>limit][:,coord][order],label='should')
            ax.plot(predic_coords[dist_epi>limit][:,coord][order],label='predic')
            
            lab = 'lat' if coord ==0 else 'lon'
            mean_std_text = f'epi big than {limit} for [{lab}]'
            ax.text(0.1, 0.95, mean_std_text, transform=ax.transAxes,fontsize=12, verticalalignment='top')
            ax.legend()
    key     = "epi_delta"
    channel = "test"
    save_dir = os.path.join(root_path, 'trace', f'{key}_{channel}')
    if not os.path.exists(save_dir):os.makedirs(save_dir)
    plt.suptitle(f'{key}_{channel}')
    fig.savefig(os.path.join(save_dir, f"epoch-{epoch:03d}.png"))
    fig.clear()
    plt.clf()      


def plot_order_by_ascent(target, preded, ax=None,
                         linewidth1=2, color1='r',
                         linewidth2=0.5, color2='b'):
    if ax is None:
        fig, ax = plt.subplot()
    if np.any(target):
        order = np.argsort(target)
    else:
        order = np.arange(len(target))
    target = target[order]
    preded = preded[order]
    error = np.abs(target - preded).mean()
    ax.plot(preded, linewidth=linewidth2, color=color2)
    ax.plot(target, linewidth=linewidth1, color=color1)
    ax.set_xticks([])
    ax.text(0.03, 0.95, f"mae={error:.3f}", verticalalignment='top', horizontalalignment='left',
            transform=ax.transAxes, color='black', fontsize=10)


def metric_evaluate_plot(target, preded, name, save_path = None):
    target = target.squeeze()
    preded = preded.squeeze()
    
    


    

    if len(preded.shape)==2: #(B, L)
        fig = plt.figure()
        if len(target.shape) ==1:
            delta = np.abs(preded - target[:, None])
        else:
            delta = np.abs(preded - target)
        ax1 = plt.subplot2grid((2, 3), (0, 0))
        ax2 = plt.subplot2grid((2, 3), (0, 1))
        ax3 = plt.subplot2grid((2, 3), (0, 2))
        ax4 = plt.subplot2grid((2, 3), (1, 0), colspan=3)

        ax1.set_ylabel('target(red)-pred(blue)')

        length = preded.shape[-1]
        for select_index, ax in zip([0, length//2, length-1], 
                                    [ax1,ax2,ax3]):
            t = target if len(target.shape)==1 or target.shape[-1]==1 else target[:,select_index]
            p = preded[:, select_index]
            plot_order_by_ascent(t, p, ax)
            ax.set_title(f'get {(1+ select_index/100):.2f}s')
        
        
        means = delta.mean(0)
        stds = delta.std(0)
        x_axis = np.arange(len(means))
        ax4.errorbar(x_axis, means, yerr=stds, alpha=.4, linewidth=0.1)
        ax4.plot(x_axis, means, linewidth=2)
        ax4.set_xlabel('time window (0.01s)')
        ax4.set_ylabel('abs-error (mean/std)')
        ax4.text(0.03, 0.95, f"mae={delta.mean():.3f}", verticalalignment='top', horizontalalignment='left',
                transform=ax4.transAxes, color='black', fontsize=10)
        plt.suptitle(name)
        fig.tight_layout()
    elif len(preded.shape)==1: #(B, ):
        fig,(ax1,ax2) = plt.subplots(2,1)
        delta = preded - target
        means = delta.mean()
        stds  = delta.std()
        sampled = np.random.choice(range(len(delta)), min(2000,len(delta)), replace=False)
        smoothhist(delta[sampled], color='r', ax=ax1)
        ax1.set_title(f"delta_mean={means:.3f} delta_std={stds:.3f}")
        
        delta = np.abs(preded - target)
        means = delta.mean()
        stds = delta.std()
        sampled = np.random.choice(range(len(delta)), min(2000, len(delta)), replace=False)
        smoothhist(delta[sampled], color='r', ax=ax2)
        ax2.set_title(f"abs_error:=> mean={means:.3f} delta_std={stds:.3f}")
        fig.tight_layout()

    else:
        raise NotImplementedError
    if save_path is not None:
        fig.savefig(save_path)
        fig.clear()
        plt.clf()      
    return fig


def accu_matrix_plot(accu_pool, name, save_path=None):

    fig = plt.figure()
    ax1 = plt.subplot2grid((3, 2), (0, 0))
    ax2 = plt.subplot2grid((3, 2), (0, 1))
    ax3 = plt.subplot2grid((3, 2), (1, 0))
    ax4 = plt.subplot2grid((3, 2), (1, 1))
    ax5 = plt.subplot2grid((3, 2), (2, 0), colspan=2)

    ax1.set_ylabel('100%')

    keys = [k for k in accu_pool.keys() if k != 'acc']
    for key, ax in zip(keys, [ax1, ax2, ax3, ax4]):

        ax.plot(accu_pool[key])
        ax.set_title(key)

    key = 'acc'
    ax5.plot(accu_pool[key], linewidth=2)
    ax5.set_title(key)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)
        fig.clear()
        plt.clf()
    return fig


def metric_evaluate_slide_plot(target, preded, name, p_arrive_position, unit, 
                               warning_position, save_path=None, wandb_key=None):
    target = target.squeeze()
    preded = preded.squeeze()

    if len(preded.shape) != 2:
        print(f"bad pred shape = {preded.shape}")
        return  # (B, L)
    fig = plt.figure()
    if len(target.shape) == 1:
        delta = np.abs(preded - target[:, None])
    else:
        delta = np.abs(preded - target)
    ax1 = plt.subplot2grid((2, 3), (0, 0))
    ax2 = plt.subplot2grid((2, 3), (0, 1))
    ax3 = plt.subplot2grid((2, 3), (0, 2))
    ax4 = plt.subplot2grid((2, 3), (1, 0), colspan=3)

    ax1.set_ylabel('target(red)-pred(blue)')

    length = preded.shape[-1]
    for select_index, ax in zip([0, length//2, length-1],
                                [ax1, ax2, ax3]):
        p = preded[:, select_index]
        if (target == 0).all():
            smoothhist(p,ax=ax)
            ax.text(0.1, 0.95, f"mae={p.mean():.3f}", verticalalignment='top', horizontalalignment='left',
                transform=ax.transAxes, color='black', fontsize=10)
        else:
            t = target if len(target.shape) == 1 or target.shape[-1] == 1 else target[:, select_index]  
            plot_order_by_ascent(t, p, ax)
        ax.set_title(f'@ p=> {((select_index-p_arrive_position))*unit:.2f}s')

    means = delta.mean(0)
    stds = delta.std(0)
    x_axis = np.arange(len(means)) - p_arrive_position
    if wandb_key is not None:
        for sample_id, val in zip(x_axis, means):
            wandb.log({f"report/{wandb_key}": val,
                        "report/time_base_p(second)": sample_id*unit})
        target_time   = 2 # 2 second after p 
        target_sample = p_arrive_position + int(target_time/unit)
        if target_sample < len(means):wandb.log({f"number_eval/{wandb_key}_after_p_{target_time}s": means[target_sample],f"number":0})

        target_time   = 15 # 2 second after p 
        target_sample = p_arrive_position + int(target_time/unit)
        if target_sample < len(means):wandb.log({f"number_eval/{wandb_key}_after_p_{target_time}s": means[target_sample],f"number":0})

        target_time   = 28 # 2 second after p 
        target_sample = p_arrive_position + int(target_time/unit)
        if target_sample < len(means):wandb.log({f"number_eval/{wandb_key}_after_p_{target_time}s": means[target_sample],f"number":0})

        target_time   = -1 # 2 second after p 
        target_sample = p_arrive_position + int(target_time/unit)
        if target_sample < len(means) and target_sample > 0:wandb.log({f"number_eval/{wandb_key}_before_p_{-target_time}s": means[target_sample],f"number":0})

        target_time   = -2 # 2 second after p 
        target_sample = p_arrive_position + int(target_time/unit)
        if target_sample < len(means) and target_sample > 0:wandb.log({f"number_eval/{wandb_key}_before_p_{-target_time}s": means[target_sample],f"number":0})


    ax4.errorbar(x_axis, means, yerr=stds, alpha=.4, linewidth=0.1)
    ax4.plot(x_axis, means, linewidth=2)
    ax4.axvline(x=0, color='r', linestyle='--')
    if warning_position>0:
        ax4.axvline(x=-warning_position, color='g', linestyle='--')
    
    ax4.set_xlabel(f'window start time (P_arrive is Red) (Train Range is Green-Red) (unit={unit})')
    ax4.set_ylabel('abs-error (mean/std)')
    ax4.text(0.03, 0.95, f"mae={delta.mean():.3f}", verticalalignment='top', horizontalalignment='left',
                transform=ax4.transAxes, color='black', fontsize=10)
    plt.suptitle(name)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)
        fig.clear()
        plt.clf()
    return fig


def error_dependence_diagram(df, error_key, depend_key, save_path):
    import seaborn as sns
    fig = plt.figure()
    sns.kdeplot(data=df, x=depend_key, y=error_key,
                cmap='Blues', fill=True)

    # Calculate the 1st and 99th percentiles of the data
    x_min, x_max = np.percentile(df[depend_key], [0,  90])
    y_min, y_max = np.percentile(df[error_key],  [0,  90])

    # Set the limits of the x and y axes
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel(f"value_of_{depend_key}")
    plt.ylabel(f"error_of_{error_key}")
    plt.title(f'{error_key}_error_vs_{depend_key}')
    assert save_path is not None
    fig.savefig(save_path)
    fig.clear()
    plt.clf()

def scatter_error_vs_para_diagram(error, para, name, save_path=None):
    fig = plt.figure()
    plt.scatter(para, error, s=10, alpha=0.2, marker='o')
    plt.xlabel('Parameter')
    plt.ylabel('Error')
    plt.yscale('log')
    plt.title(f'[{name}][Scatter] Error vs Parameter')
    if save_path is not None:
        fig.savefig(save_path)
        fig.clear()
        plt.clf()
    return fig

def scatter_real_vs_pred_diagram(real_data, pred_data, name, save_path=None):
    fig = plt.figure()
    plt.scatter(real_data, pred_data, s=10, alpha=0.2, marker='o')
    limits = [np.min([plt.xlim(), plt.ylim()]),  # find the minimum limit between x and y
              np.max([plt.xlim(), plt.ylim()])]  # find the maximum limit between x and y
    plt.plot(limits, limits, 'r')  # plot a red line with 45-degree

    plt.xlim(limits)
    plt.ylim(limits)

    plt.xlabel('Real')
    plt.ylabel('Predicted')
    plt.title(f'[{name}][Scatter] Real vs Predicted')
    if save_path is not None:
        fig.savefig(save_path)
        fig.clear()
        plt.clf()
    return fig

def error_intervel_hist_diagram(df, error_key, depend_key, save_path):
    if depend_key in df and error_key in df:
        x_ranges = np.linspace(df[depend_key].min(), df[depend_key].max(), 5).astype('int')

        fig, axs = plt.subplots(1, len(x_ranges)-1, figsize=(10, 5), sharey=True)
        y_min, y_max = np.percentile(df[error_key],  [0,  99])
        for i, (x_min, x_max) in enumerate(zip(x_ranges[:-1], x_ranges[1:])):
            # Subset the data
            subset = df[(df[depend_key] >= x_min) & (df[depend_key] < x_max)]
            if len(subset) ==0:continue
            # Create a histogram of y for the subset
            smoothhist(subset[error_key], axs[i], color='blue', alpha=0.7)
            axs[i].set_title(f'{depend_key} in ({x_min}, {x_max}) ')
            axs[i].set_xlabel(f'{error_key} error')
            axs[i].set_ylabel('Histgram (Max normlized)')
            axs[i].set_xlim([y_min, y_max])
            sample_count = len(subset)
            axs[i].text(0.35, 0.95, f'Samples: {sample_count}\nMean:{subset[error_key].mean():.3f}\nStd:{subset[error_key].std():.2f}',
                        transform=axs[i].transAxes, verticalalignment='top')
        fig.savefig(save_path)
        fig.clear()
        plt.clf()

from tqdm.auto import tqdm
def get_bingo_and_accu_in_the_window(expanded_pred, real, p_index, sign, width_list):
    must_have = []
    accu = []
    for width in tqdm(width_list, desc='iterating different width', position=0, leave=False):
        col_indices = []
        for offset in range(2*width):
            col_indices.append(p_index - width + offset)
        col_indices = np.stack(col_indices, -1)
        row_indices = np.repeat(np.arange(real.shape[0])[:, None], 2*width, axis=-1)
        max_length  = expanded_pred.shape[1]
        mask        = col_indices.max(-1)<max_length
        row_indices = row_indices[mask]
        col_indices = col_indices[mask]
        should_data = expanded_pred[row_indices, col_indices]
        ground_data = real[row_indices, col_indices]
        must_have.append([(should_data == sign).any(axis=-1),mask])
        accu.append((should_data == ground_data).mean())
    #must_have = np.stack(must_have, -1).astype('int8')
    #accu      = np.stack(accu,-1).astype('int8')
    return must_have, accu


def plot_LogisticBingo(p_must_have, s_must_have, snr_db, magnitude, width_list, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    for i, (status, data_source_with_mask) in enumerate(zip(['p', 's'], [p_must_have, s_must_have])):
        for j, (tag, depend_data) in enumerate(zip(['SNR', 'MAG'], [snr_db, magnitude])):
            ax = axes[i][j]
            for k, (must_have, mask) in enumerate(data_source_with_mask):
                X = depend_data[mask].reshape(-1, 1)
                Y = must_have
                model = LogisticRegression()
                model.fit(X, Y)
                orderX = np.sort(X, axis=0)
                Prob_Y_given_X = model.predict_proba(orderX)[:, 1]
                ax.plot(orderX, Prob_Y_given_X, alpha=1,
                        label=f"±{width_list[k]}")
            ax.legend()
            ax.set_xlabel(tag)
            ax.set_ylabel(f'E[{status}|{tag}]')
            ax.set_title(f'Expected {status}-Bingo of Bingo given {tag}')
    if save_path is not None:
        fig.savefig(save_path)
        fig.clear()
        plt.clf()
    return fig


def plot_accurancy_relaxation(p_must_have, p_accu, s_must_have, s_accu, width_list, unit, save_path=None, wandb_key=None):
    fig, axes = plt.subplots(2)

    for idx, (ax, status, (must_have, accu)) in enumerate(zip(axes, ['p', 's'], [[p_must_have, p_accu], [s_must_have, s_accu]])):
        ax.set_ylabel(f'{status}_predict')
        if wandb_key is not None: 
            for (w, a, b) in zip(width_list, accu, must_have):
                wandb.log({f"report_status/{wandb_key}/accu_{status}": a,
                           f"report_status/{wandb_key}/bingo_{status}": b,
                         "report_status/relaxtion(second)": w*unit})
        ax.plot(accu, label='accu')
        ax.plot(must_have, label='bingo')

        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax*1.18)  # Extend the upper limit by 20%
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        ax.set_xlim(-1, 9)
        for i, (a, b) in enumerate(zip(accu, must_have)):
            ax.annotate(f'{a:.0%}', (i, a), textcoords="offset points", xytext=(0, -20), ha='center', arrowprops=dict(arrowstyle='->'))
            ax.annotate(f'{b:.0%}', (i, b), textcoords="offset points", xytext=(20, +15), ha='center', arrowprops=dict(arrowstyle='->'))
        ax.set_xticks(np.arange(len(width_list)))
        ax.set_xticklabels([f"±{t}" for t in width_list])
        ax.legend(loc='upper left')
        if idx == 1:
            ax.set_xlabel(f'half-window width (unit={unit:.2f}s [may change for different dataset])')
        if idx == 0:
            ax.set_title('prediction_accurancy_with_relaxation interval')
    if save_path is not None:
        fig.savefig(save_path)
        fig.clear()
        plt.clf()
    return fig

def max_count_filter(signal, window_size=7):
    windows = np.lib.stride_tricks.sliding_window_view(signal, window_size)
    modes, counts = stats.mode(windows, axis=1, keepdims=True)
    result = modes.flatten()
    return result


def phase_prediction_fourline_diagram(real, pred,sample_idxes, metadata_handle,waveform_handle,sample=3, window_size=7, save_path = None, wave_padding=None):
    expand = int(np.round(1.0*real.shape[1] / pred.shape[1]))
    bingo = real[:,::expand][:,:pred.shape[1]] == pred
    bingo = bingo.shape[1] - bingo.sum(1)
    order = np.argsort(bingo)


    
    worst_case_indexes = order[-sample:]
    middle = len(order)//3*2
    bad_case_indexes  = order[middle:middle+sample]
    middle = len(order)//3
    good_case_indexes = order[middle:middle+sample]
    best_case_indexes = order[:sample]
    # Create a gridspec object with height ratios [1, 2, 2, 2]
    gs = gridspec.GridSpec(4, sample+1, width_ratios=[1]+ [2]*sample)

    # Define the figure
    fig = plt.figure(figsize=((2*sample+1)*2, 8))

    # Define the first subplot that spans all 4 rows
    ax0 = fig.add_subplot(gs[:, 0])  # spans all rows, first column
    data = bingo
    density = gaussian_kde(data)
    xs = np.linspace(min(data), max(data), 200)
    density.covariance_factor = lambda : .25
    density._compute_covariance()
    y_density = density(xs)
    y_density = y_density / y_density.max()
    ax0.fill_betweenx(xs, y_density, alpha=0.5)  # Vertical filled curve
    ax0.invert_yaxis() 
    ax0.set_ylabel('error_count')
    ax0.set_xticks([])
    mean_count= np.mean(bingo)
    totl_count= pred.shape[1]
    acc = 100*(1-mean_count/totl_count)
    ax0.text(0,0.9*max(bingo),f'mean error:{mean_count:.0f}\ntotal slots:{totl_count:d}\nacc:{acc:.0f}%')

    for i,ind_row in enumerate([best_case_indexes,good_case_indexes, bad_case_indexes,worst_case_indexes]):
        for j, ind in enumerate(ind_row):
            ax = fig.add_subplot(gs[i, j+1])
            pred_sequence = pred[ind]
            filte_sequence= max_count_filter(pred[ind],window_size)
            real_sequence = real[ind]

            real_index = sample_idxes[ind]
            wave_name = metadata_handle(real_index)['trace_name']
            mag       = metadata_handle(real_index)['source_magnitude']
            wave = waveform_handle(real_index)[...,0]
            if wave_padding is not None:
                wave = np.pad(wave,wave_padding,'constant',constant_values=0)
            wave      = np.abs(wave)

            maxwave = wave.max()
            ax.plot(wave/wave.max()*2,'r',alpha=0.3)
            ax.plot(np.repeat(filte_sequence,expand),'b',linewidth=3)
            ax.plot(np.repeat(pred_sequence,expand),'g',alpha=0.5,linewidth=3)
            ax.plot(real_sequence,'r')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{wave_name}|H:{maxwave:.0f}|M:{mag:.1f}",fontsize=10)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
        fig.clear()
        plt.clf()   
    return fig


    
    expand = int(np.round(1.0*real.shape[1] / pred.shape[1]))
    bingo = real[:,::expand][:,:pred.shape[1]] == pred
    bingo = bingo.shape[1] - bingo.sum(1)
    order = np.argsort(bingo)


    
    worst_case_indexes = order[-sample:]
    middle = len(order)//3*2
    bad_case_indexes  = order[middle:middle+sample]
    middle = len(order)//3
    good_case_indexes = order[middle:middle+sample]
    best_case_indexes = order[:sample]
    # Create a gridspec object with height ratios [1, 2, 2, 2]
    gs = gridspec.GridSpec(4, sample+1, width_ratios=[1]+ [2]*sample)

    # Define the figure
    fig = plt.figure(figsize=((2*sample+1)*2, 8))

    # Define the first subplot that spans all 4 rows
    ax0 = fig.add_subplot(gs[:, 0])  # spans all rows, first column
    data = bingo
    density = gaussian_kde(data)
    xs = np.linspace(min(data), max(data), 200)
    density.covariance_factor = lambda : .25
    density._compute_covariance()
    y_density = density(xs)
    y_density = y_density / y_density.max()
    ax0.fill_betweenx(xs, y_density, alpha=0.5)  # Vertical filled curve
    ax0.invert_yaxis() 
    ax0.set_ylabel('error_count')
    ax0.set_xticks([])
    mean_count= np.mean(bingo)
    totl_count= pred.shape[1]
    acc = 100*(1-mean_count/totl_count)
    ax0.text(0,0.9*max(bingo),f'mean error:{mean_count:.0f}\ntotal slots:{totl_count:d}\nacc:{acc:.0f}%')

    for i,ind_row in enumerate([best_case_indexes,good_case_indexes, bad_case_indexes,worst_case_indexes]):
        for j, ind in enumerate(ind_row):
            ax = fig.add_subplot(gs[i, j+1])
            pred_sequence = pred[ind]
            filte_sequence= max_count_filter(pred[ind],window_size)
            real_sequence = real[ind]

            real_index = sample_idxes[ind]
            wave_name = df.iloc[real_index]['trace_name']
            mag       = df.iloc[real_index]['source_magnitude']
            wave      = np.abs(waveform[real_index][...,0])

            maxwave = wave.max()
            ax.plot(wave/wave.max()*2,'r',alpha=0.3)
            ax.plot(np.repeat(filte_sequence,expand),'b',linewidth=3)
            ax.plot(np.repeat(pred_sequence,expand),'g',alpha=0.5,linewidth=3)
            ax.plot(real_sequence,'r')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{wave_name}|H:{maxwave:.0f}|M:{mag:.1f}",fontsize=10)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
        fig.clear()
        plt.clf()   
    return fig




#######################################################################
#                        Metrix for P/S picking
#######################################################################


# def findPS(signal,window_size=7):
#     # find the first 1 and remain window size
#     ppicks = -1
#     spicks = -1
#     for i in range(signal.shape[0]):
#         # find the first 1
#         if signal[i] == 1 and ppicks==-1:
#             # the 1 remain large than window_size
#             # if np.sum(signal[i:i+window_size] == 1) == window_size and signal[i-1] == 0:
#             if np.sum(signal[i:i+window_size] == 1) == window_size:
#                 ppicks = i
#         if signal[i] == 2 and spicks==-1:
#             if np.sum(signal[i:i+window_size] == 2) == window_size:
#                 spicks = i
#         if ppicks!=-1 and spicks !=-1:
#             return ppicks,spicks
#     return ppicks,spicks

# from p_tqdm import p_map
# def findAllPS(pred_status,status_win=7,ps_win = 7):
#     window_sizes = np.ones(pred_status.shape[0])*status_win
#     window_sizes = window_sizes.astype(int)
#     # get the filte status
#     filte_status = p_map(max_count_filter,pred_status,window_sizes)
#     filte_status = np.array(filte_status)
    
#     # get the sample position of the P/S
#     ppicks,spicks = [],[]
#     for ind in tqdm(range(pred_status.shape[0])):
#         # filte_sequence= max_count_filter(pred_status[ind,:],window_size=window_size)
#         filte_sequence = filte_status[ind,:]
#         ppick,spick = findPS(filte_sequence,window_size=ps_win)
#         ppicks.append(ppick)
#         spicks.append(spick)
#     return ppicks,spicks


def compute_accurancy_metric(TP, FP, TN, FN, APP, APL, eps=1e-10):
    # TPR = TP / (TP + FN + eps)# True  Positive rate (FPR)
    # FPR = FP / (FP + TN + eps)# False Positive rate (FPR)
    # TNR = TN / (TN + FP + eps)# True  Negative Rate (TNR)
    # FNR = FN / (FN + TP + eps)# False Negative Rate (FNR)
    
    # precision = TP/(TP+FP+ eps) # precision = Tp/(Tp+Fp)
    # recall    = TP/(TP+FN+ eps) # recall = Tp/(Fn + Tp)
    precision = TP/(APP+ eps)
    recall    = TP/(APL+ eps)
    f1        = 2*precision*recall/(precision+recall+eps)# F1 = 2*precision*recall/(precision+recall)
    return {'precision':precision, 
            'recall':recall, 
            'f1':f1}

import pandas as pd 
def metrix_PS(pred,target, freq,max_length,flag, wandb_key=None,save_path=None,verbose=True,
              metric_types=['alpha','beta','gamma']):
    # 统计前后1s内 precision
    t_ranges = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

    # ======================================================
    # True positive (TP): we predicted a positive result and it was positive
    # False negative (FN): we predicted a negative result and it was positive
    # ======================================================
    # False positive (FP): we predicted a positive result and it was negative
    # True negative (TN): we predicted a negative result and it was negative
    ## <--- we only look at the case that target in our prediction domain, thus FP=0
    # all positive precition (APP)
    # all positive labels    (APL)

    TP_alpha = {t_range:0 for t_range in t_ranges }
    FN_alpha = {t_range:0 for t_range in t_ranges }
    FP_alpha = {t_range:0 for t_range in t_ranges }
    TN_alpha = {t_range:0 for t_range in t_ranges }
    APP_alpha= {t_range:0 for t_range in t_ranges }
    APL_alpha= {t_range:0 for t_range in t_ranges }

    TP_beta  = {t_range:0 for t_range in t_ranges }
    FN_beta  = {t_range:0 for t_range in t_ranges }
    FP_beta  = {t_range:0 for t_range in t_ranges }
    TN_beta  = {t_range:0 for t_range in t_ranges }
    APP_beta = {t_range:0 for t_range in t_ranges }
    APL_beta = {t_range:0 for t_range in t_ranges }


    TP_gamma  = {t_range:0 for t_range in t_ranges }
    FN_gamma  = {t_range:0 for t_range in t_ranges }
    FP_gamma  = {t_range:0 for t_range in t_ranges }
    TN_gamma  = {t_range:0 for t_range in t_ranges }
    APP_gamma = {t_range:0 for t_range in t_ranges }
    APL_gamma = {t_range:0 for t_range in t_ranges }


    for row, real_position in tqdm(enumerate(target),total=len(target),leave=False):
        for t_range in t_ranges:
            if (real_position < 0) or (real_position > max_length): 
                ## <--- we only look at the case that target in our prediction domain, thus FP=0
                continue
            #if row in pred and len(pred[row])>1:continue  # <---- this line is for the case that we only predict one peak
            APL_beta[t_range]+=1
            APL_alpha[t_range]+=1
            if row in pred and len(pred[row])==1:APL_gamma[t_range]+=1
            if row not in pred:
                FN_alpha[t_range]+=1
                continue
            if row in pred and len(pred[row])==1:APP_gamma[t_range]+=1
            
            APP_beta[t_range]+=1
            hasPositive = False
            hasNegitive = False
            for pred_position in pred[row]:
                if pred_position<0: continue
                APP_alpha[t_range]+=1
                ae  = np.abs(pred_position - real_position)  
                if ae < t_range*freq:
                    TP_alpha[t_range] +=1
                    hasPositive = True
                else:
                    FN_alpha[t_range]+=1
                    hasNegitive = True
            TP_beta[t_range] += int(hasPositive)
            FN_beta[t_range] += int(hasNegitive)   
            if row in pred and len(pred[row])==1:
                TP_gamma[t_range] += int(hasPositive)
                FN_gamma[t_range] += int(hasNegitive)   

    metric_pool_all = {}

    if 'alpha' in metric_types:
        for t_range in t_ranges:
            metric_pool = compute_accurancy_metric(TP_alpha[t_range], 
                                                FP_alpha[t_range], 
                                                TN_alpha[t_range], 
                                                FN_alpha[t_range],
                                                APP_alpha[t_range],
                                                APL_alpha[t_range])
            for key, val in metric_pool.items():
                key = f"{flag}/{key}"
                if key not in metric_pool_all:metric_pool_all[key] = {}
                metric_pool_all[key][t_range] = val

        if verbose: print(f"==================== {flag} alpha counting table ====================")
        df = pd.DataFrame({n:v for n,v in zip(["TP","FP","TN","FN","APP","APL"], [TP_alpha, FP_alpha, TN_alpha, FN_alpha,APP_alpha,APL_alpha])})
        if verbose: print(df)   

    if 'beta' in metric_types:
        for t_range in t_ranges:
            metric_pool = compute_accurancy_metric(TP_beta[t_range], 
                                                FP_beta[t_range], 
                                                TN_beta[t_range], 
                                                FN_beta[t_range],
                                                APP_beta[t_range],
                                                APL_beta[t_range])
            for key, val in metric_pool.items():
                key = f"{flag}/{key}_beta"
                if key not in metric_pool_all:metric_pool_all[key] = {}
                metric_pool_all[key][t_range] = val
        
        if verbose: print(f"==================== {flag} beta counting table ====================")
        df = pd.DataFrame({n:v for n,v in zip(["TP","FP","TN","FN","APP","APL"], [TP_beta, FP_beta, TN_beta, FN_beta,APP_beta,APL_beta])})
        if verbose: print(df)

    if 'gamma' in metric_types:
        for t_range in t_ranges:
            metric_pool = compute_accurancy_metric( TP_gamma[t_range], 
                                                    FP_gamma[t_range], 
                                                    TN_gamma[t_range], 
                                                    FN_gamma[t_range],
                                                APP_gamma[t_range],
                                                APL_gamma[t_range])
            for key, val in metric_pool.items():
                key = f"{flag}/{key}_gamma"
                if key not in metric_pool_all:metric_pool_all[key] = {}
                metric_pool_all[key][t_range] = val

        if verbose: print(f"==================== {flag} gamma counting table ====================")
        df = pd.DataFrame({n:v for n,v in zip(["TP","FP","TN","FN","APP","APL"], [TP_gamma, FP_gamma, TN_gamma, FN_gamma,APP_gamma,APL_gamma])})
        if verbose: print(df)

    if verbose: print(f"------------------------------------------------------------------")

    if verbose: print(f"==================== {flag} metric table ====================")
    df = pd.DataFrame(metric_pool_all)
    df.round(2)
    if verbose: print(df)

    ########## Lets plot the curve ############# 
    plt_cols  = 3
    plt_rows  = int(np.ceil(len(metric_pool_all)/plt_cols))
    fig, axes = plt.subplots(plt_rows, plt_cols, figsize=(3*plt_cols, 3*plt_rows))
    axes = axes.flatten()
    for ax, (key, metric) in zip(axes, metric_pool_all.items()):
        t_ranges = list(metric.keys())
        values   = [metric[t] for t in t_ranges]
        ax.scatter(t_ranges,values,c='r')
        ax.set_title(key)
        ax.set_xlabel("Tolerance threshold",fontsize=14)
        #print(f"metric: {key} => {values}")
    plt.tight_layout()


    if save_path:
        fig.savefig(save_path,bbox_inches="tight",dpi=300)
    plt.clf()    
    plt.close()

    pool= {f"{key}.at{0.5}": val[0.5] for key, val in metric_pool_all.items() if (key.split('/')[-1] in ['precision','recall','recall_beta'])}
    
    if wandb_key is not None:
        #wandb.log(pool|{"report/time_range(second)":0.5})
        for key, metric in metric_pool_all.items():
            for t_range, value in metric.items():
                wandb.log({f"report/{wandb_key}/{key}": value,"report/time_range(second)": t_range})
        
        t_range = t_ranges[-1]
        wandb.log({f"report/{wandb_key}/{flag}/APL": APL_alpha[t_range],
                   f"report/{wandb_key}/{flag}/APP": APP_alpha[t_range],
                   f"report/{wandb_key}/{flag}/APP_beta": APP_beta[t_range],
                   f"report/{wandb_key}/{flag}/APP_gamma": APP_gamma[t_range],
                   f"report/{wandb_key}/{flag}/APL_gamma": APL_gamma[t_range],
                   f"report/{wandb_key}/{flag}/DatasetCo": len(target),
                   "report/time_range(second)": t_range}
                )
        # for key, val in zip(["TP","FP","TN","FN","APP","APL"], [TP_alpha, FP_alpha, TN_alpha, FN_alpha,APP_alpha,APL_alpha]):
        #     key = f"{flag}_{key}_alpha"
        #     for t_range in metric.keys():
        #         wandb.log({f"report/{wandb_key}/{key}": val[t_range],"report/time_range(second)": t_range})
        # for key, val in zip(["TP","FP","TN","FN","APP","APL"],
        #                      [TP_beta, FP_beta, TN_beta, FN_beta,APP_beta,APL_beta]):
        #     key = f"{flag}_{key}_beta"
        #     for t_range in metric.keys():
        #         wandb.log({f"report/{wandb_key}/{key}": val[t_range],"report/time_range(second)": t_range})
    
    return pool 

def angle_depend_error2real(real_line_angle, error_line_angle, ax=None, save_path=None, wandb_key=None):
# Create a DataFrame from your points
    df = pd.DataFrame({
        'x': real_line_angle,  # Replace this with your actual continuous x values
        'y': error_line_angle  # Replace with your y values
    })

    # Define bin edges, or use np.linspace to create a specific number of bins
    bin_edges = np.linspace(df['x'].min(), df['x'].max(), num=100)  # Adjust 'num' for the number of bins

    # Assign each x value to a bin
    df['x_bin'] = pd.cut(df['x'], bins=bin_edges, include_lowest=True, right=True)

    # Group by the bins and calculate mean and std dev
    bin_stats = df.groupby('x_bin')['y'].agg(['mean', 'std']).reset_index()

    # For plotting, we need to convert the bin intervals to a single x value (e.g., bin center)
    bin_centers = [interval.mid for interval in bin_stats['x_bin']]

    # Plotting
    if ax is None:
        fig, ax = plt.subplots(1,figsize=(10, 5))
    
    # Plot the mean
    ax.errorbar(bin_centers, bin_stats['mean'], yerr=bin_stats['std'], fmt='o', 
                 ecolor='lightgray', elinewidth=3, capsize=0, label='Mean and Std Dev')

    # Adding titles and labels
    ax.set_title('Mean and Standard Deviation of y within x intervals')
    ax.set_xlabel('real angle')
    ax.set_ylabel('angle error')
    ax.legend()

    if save_path is not None:
        fig.savefig(save_path)
        fig.clear()
        plt.clf()
    
    if wandb_key is not None:
        for x, y in zip(bin_centers, bin_stats['mean']):
            wandb.log({f"report/{wandb_key}/angle_error": y,"report/real_angle": x})

from matplotlib.collections import LineCollection
def long_term_findP_plot(alldata,dataset, slide_feature_window_size,slide_stride_in_training,real_time_for_one_stamp = 0.02, save_path=None, expansion=3):
    slots= [0,1,2,3,4,5,6,7,8]
    ### lets plot
    fig, axes = plt.subplots(3, 3,figsize=(24,15))
    axes = axes.flatten()
    unit = real_time_for_one_stamp*expansion
    for ax, slot in zip(axes, slots):
        origin_curve = dataset[slot]['waveform_seq']
        if 'trend_seq' in dataset[slot]:
            origin_curve+=dataset[slot]['trend_seq']
        origin_curve = origin_curve[:,:3]
        origin_curve = np.abs(origin_curve[::expansion,0])
        origin_curve = origin_curve/origin_curve.max()*2
        x = unit*np.arange(len(origin_curve))
        ax.plot(x,origin_curve, 'gray', linewidth=1,alpha=0.5)
        ax.plot(x,dataset[slot]['status_seq'][::expansion], 'black', linewidth=5)
        locations = []
        start_points = []
        prediction=alldata['findP']['pred'][slot]
        for i,position in enumerate(prediction):
            start_in_stamp = slide_stride_in_training*i
            if position >0:
                location = position + start_in_stamp
                locations.append(location)
                start_points.append(start_in_stamp+slide_feature_window_size)
        # Create start and end points for x and y
        x_starts = unit*np.array(start_points)
        y_starts = np.zeros_like(start_points)
        x_ends   = unit*np.array(locations)
        y_ends   = np.ones_like(locations)

        # Prepare line segments
        segments = [((x_start, y_start), (x_end, y_end)) for x_start, y_start, x_end, y_end in zip(x_starts, y_starts, x_ends, y_ends)]
        # Create a line collection from the segments
        lc = LineCollection(segments, colors='red', linewidths=.8, alpha=.5)
        ax.add_collection(lc)
        # Auto-scale the view limits based on the line collection
        ax.autoscale()
        ax.set_yticks([])
    fig.supxlabel('Leading Time/s', fontsize=16)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
        fig.clear()
        plt.clf()

def binary_search_next_smallest(S, value):
    left, right = 0, len(S) - 1
    result = -1
    while left <= right:
        mid = left + (right - left) // 2
        if S[mid] > value:
            result = mid
            right = mid - 1
        else:
            left = mid + 1
    return result

def factor_filter(P, S):
    P = sorted(P)
    S = sorted(S)
    R = set()

    for p in P:
        index = binary_search_next_smallest(S, p)
        if index != -1:
            R.add(S[index])

    return R

def long_term_phasepicking(args, p_position_map_pool, s_position_map_pool,dataset,slots= [0,1,2,3,4,5,6,7,8], 
                                         windows_size_for_the_start_point=7, save_path=None):
    expansion=args.model.Embedding.resolution
    one_single_trace_length = dataset.config.Resource.resource_length
    invervel_length = dataset.intervel_length
    waveform_start = -dataset.early_warning
    assert dataset.start_point_sampling_strategy == 'ahead_L_to_the_sequence'
    num_of_events = (dataset.max_length - dataset.early_warning)//(invervel_length+one_single_trace_length)+1
    real_start_location=[]
    for i in range(num_of_events):
        real_start_location.append( int((one_single_trace_length+invervel_length)*(i)  - waveform_start ))
        
    unit = 1.0/args.DataLoader.Dataset.Resource.sampling_frequence*expansion
    with plt.style.context('fast'):
        ### lets plot
        fig, axes = plt.subplots(3, 3,figsize=(12,8))
        axes = axes.flatten()
        for ax, slot in zip(axes, slots):
            ax.set_facecolor('grey')
            for start_pos in real_start_location:
                ax.axvspan(start_pos//expansion, start_pos//expansion + one_single_trace_length//expansion, color='white', alpha=1)
            origin_curve = dataset[slot]['waveform_seq']
            if 'trend_seq' in dataset[slot]:
                origin_curve+=dataset[slot]['trend_seq']
            origin_curve = origin_curve[:,:expansion]
            origin_curve = origin_curve[::expansion,0]
            origin_curve = np.abs(origin_curve)
            origin_curve = origin_curve/origin_curve.max()*2
            x = unit*np.arange(len(origin_curve))
            ax.plot(origin_curve, 'gray', linewidth=1)
            ax.plot(dataset[slot]['status_seq'][::expansion], 'black', linewidth=5)
            P = p_position_map_pool[slot]
            real_start_pos = args.task.sampling_strategy.valid_sampling_strategy.early_warning + np.arange(5)*(args.DataLoader.Dataset.Resource.basemaxlength+args.DataLoader.Dataset.component_intervel_length)
            real_start_intervel_left = real_start_pos//expansion - windows_size_for_the_start_point
            real_start_intervel_rigt = real_start_pos//expansion + windows_size_for_the_start_point
            if len(P)>2:
                P = [p for p in P if not np.any(np.logical_and(real_start_intervel_left <p, p<real_start_intervel_rigt))]
            for a in P:ax.vlines(a//expansion, 0, 1, linewidth=2)
            S = s_position_map_pool[slot]
            S = factor_filter(P, S)
            for a in S:ax.vlines(a//expansion, 0, 2, linewidth=2,color='g')
            ax.set_yticks([])
        fig.supxlabel(f'Leading Time unit={unit}s', fontsize=16)
        plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
        fig.clear()
        plt.clf()

import matplotlib.cm as cm
def kvflow_monitor_plot(args, dataset, monitor_data,save_path=None):
    expansion=args.model.Embedding.resolution
    one_single_trace_length = dataset.config.Resource.resource_length
    invervel_length = dataset.intervel_length
    waveform_start = -dataset.early_warning
    assert dataset.start_point_sampling_strategy == 'ahead_L_to_the_sequence'
    num_of_events = (dataset.max_length - dataset.early_warning)//(invervel_length+one_single_trace_length)+1
    real_start_location=[]
    for i in range(num_of_events):
        real_start_location.append( int((one_single_trace_length+invervel_length)*(i)  - waveform_start ))
    Deepth = monitor_data.shape[-1]
    Heads = monitor_data.shape[-2]
    colors = cm.tab20(np.linspace(0, 1, Heads))  # Choose a colormap and generate 16 colors

    fig, axes = plt.subplots(Deepth,1, figsize=(10, 10))

    for iii, (ax, layer_num) in enumerate(zip(axes, range(Deepth))):
        for start_pos in real_start_location:
            ax.axvspan(start_pos//expansion, start_pos//expansion + one_single_trace_length//expansion, color='white', alpha=1)
        
        max_value =0
        for head_num, color in zip(range(16), colors):
            data = monitor_data[:,head_num,layer_num]/100
            max_value = max(max_value, max(data))
            ax.plot(data,label=f"H{head_num:02d}", color=color)   
        
        ax.text(0,max_value//2,f'layer_{layer_num}')
        if iii != 3:ax.set_xticks([])
    fig.subplots_adjust(right=0.8)  # Adjust the right space to make room for the legend
    axes[0].legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
        fig.clear()
        plt.clf()

def long_term_error_plot(args, dataset, error_pool, real_start_location,save_path=None, wandb_key=None ):
    expansion=args.model.Embedding.resolution
    one_single_trace_length = dataset.config.Resource.resource_length
    invervel_length = dataset.intervel_length
    waveform_start = -dataset.early_warning
    assert dataset.start_point_sampling_strategy == 'ahead_L_to_the_sequence'
    num_of_events = (dataset.max_length - dataset.early_warning)//(invervel_length+one_single_trace_length)+1
    real_start_location=[]
    for i in range(num_of_events):
        real_start_location.append( int((one_single_trace_length+invervel_length)*(i)  - waveform_start ))
    unit=1/dataset.sampling_frequence*expansion

    fig, axeses = plt.subplots(4,1,figsize=(15,9))
    for iii, (ax,key) in enumerate(zip(axeses,error_pool.keys())):
        delta  = error_pool[key]
        
        means  = delta.mean(0)
        x_axis = np.arange(len(means)) #- dataset.early_warning # <--- let the p start position is 0
        if wandb_key is not None:
            for sample_id, val in zip(x_axis, means):
                wandb.log({f"longterm/report/{wandb_key}/{key}": val,
                            "longterm/report/time_in_real(second)": sample_id*unit})
        stds   = delta.std(0)
        minpos = np.argmin(means)
        
        ax.errorbar(x_axis, means, yerr=stds, alpha=.4, linewidth=0.1)
        ax.plot(x_axis, means, linewidth=2, label = f"min={means[minpos]:.3f}")
        min_y = means[minpos] - stds[minpos]
        max_y = max(means) 
        ax.set_ylim([min_y, max_y+ stds[minpos]])
        ax.set_ylabel(f'e_{key}')
        for start_pos in real_start_location:
            start = start_pos//expansion
            end   = (start_pos+one_single_trace_length)//expansion
            middle= (start + end)//2
            if start>=end:continue  ###
            if start>len(means):continue  ### The second quake is not in the domain
            ax.text(middle,max_y,f"min={min(means[start:end]):.3f}")
    #plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
        fig.clear()
        plt.clf()

def compute_normal_metric(TP,FP,TN,FN,eps=1e-10):
    TPR = TP / (TP + FN + eps)# True  Positive rate (FPR)
    FPR = FP / (FP + TN + eps)# False Positive rate (FPR)
    TNR = TN / (TN + FP + eps)# True  Negative Rate (TNR)
    FNR = FN / (FN + TP + eps)# False Negative Rate (FNR)
    
    accuracy = (TP + TN) / (TP + TN + FP + FN + eps)
    precision = TP / (TP + FP+ eps)
    recall = TP / (TP + FN+ eps)
    f1_score = 2 * (precision * recall) / (precision + recall+ eps)
    specificity = TN / (TN + FP+ eps)
    false_positive_rate = FP / (FP + TN+ eps)
    negative_predictive_value = TN / (TN + FN+ eps)
    false_discovery_rate = FP / (FP + TP+ eps)
    false_negative_rate = FN / (FN + TP+ eps)

    return {
            "accuracy":accuracy,
            "precision":precision,
            "recall":recall,
            "f1_score":f1_score,
            # "specificity":specificity,
            # "false_positive_rate":false_positive_rate,
            # "negative_predictive_value":negative_predictive_value,
            "false_discovery_rate":false_discovery_rate,
            "false_negative_rate":false_negative_rate,
    
    }
  
def metrix_PS_for_findPS(  pred:np.ndarray, target:np.ndarray, freq, max_length, t_range=0.5, wandb_key=None,save_path=None,verbose=True):
    #t_ranges   = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    eps = 1e-10
    resolution = t_range*freq ## 0.5s --> 50
    # ======================================================
    # True positive (TP): we predicted a positive result and it was positive
    # False negative (FN): we predicted a negative result and it was positive
    # ======================================================
    # False positive (FP): we predicted a positive result and it was negative
    # True negative (TN): we predicted a negative result and it was negative
    ## <--- we only look at the case that target in our prediction domain, thus FP=0
    # all positive precition (APP)
    # all positive labels    (APL)


    negative_pred_part   = np.logical_or(pred < 0, pred > max_length)
    activate_pred_part   = ~negative_pred_part
    negative_target_part = np.logical_or(target < 0, target > max_length)
    activate_target_part = ~negative_target_part
    #### 
    TP_select= np.logical_and(activate_target_part, activate_pred_part)
    TP_pred  =   pred[TP_select]
    TP_target= target[TP_select]
    TP       = np.sum(np.logical_and(TP_pred>TP_target - resolution , TP_pred < TP_target + resolution))
    ####
    FN = np.sum(activate_target_part) - TP
    ####
    TN = np.sum(np.logical_and(negative_pred_part  , negative_target_part))
    #FP = np.sum(np.logical_and(activate_pred_part  , negative_target_part))
    FP = np.sum(activate_pred_part) - TP  #### <--- which one is the correct FP????????????
    #return {'TP':TP,  'FN':FN,  'TN':TN, 'FP':FP}
    #print(f"TP={TP},FP={FP},TN={TN},FN={FN}")
    return compute_normal_metric(TP,FP,TN,FN)

from typing import Dict, Set
def metrix_PS_for_set_set(pred:Dict[int,Set], target:Dict[int,Set], 
              freq,max_length,flag, wandb_key=None,save_path=None,verbose=True,
              metric_types=['alpha','beta','gamma'],t_ranges=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]):
    # ======================================================
    # True positive (TP): we predicted a positive result and it was positive
    # False negative (FN): we predicted a negative result and it was positive
    # False positive (FP): we predicted a positive result and it was negative
    # True negative (TN): we predicted a negative result and it was negative
    ## <--- we only look at the case that target in our prediction domain, thus FP=0
    # all positive precition (APP)
    # all positive labels    (APL)

    TP  = {t_range:0 for t_range in t_ranges }
    FN  = {t_range:0 for t_range in t_ranges }
    FP  = {t_range:0 for t_range in t_ranges }
    TN  = {t_range:0 for t_range in t_ranges }

    for t_range in t_ranges:
        for row, pred_set in pred.items():
            real_set = target[row]
            countTP = 0
            for real_position in real_set:
                isTP = 0
                for pred_position in pred[row]:
                    ae  = np.abs(pred_position - real_position)  
                    if ae < t_range*freq:
                        isTP=1
                        break
                countTP+=isTP
            FN[t_range] += len(real_set) - countTP
            FP[t_range] += len(pred[row]) - countTP
            TP[t_range] += countTP
        for row, real_set in target.items():
            if row in pred:continue
            for pred_position in real_set:
                if pred_position<0 or pred_position>=max_length:continue
                FN[t_range] += 1
    metrixs = []
    for t_range in t_ranges:
        metrixs.append({'t_range(second)':t_range}|compute_normal_metric(
            TP[t_range], 
            FP[t_range], 
            TN[t_range], 
            FN[t_range], 
        ))

    if wandb_key is not None:
        #wandb.log(pool|{"report/time_range(second)":0.5})
        for metric in metrixs:
            for key, value in metric.items():
                wandb.log({f"report/{wandb_key}/{key}": value})
    
    #t_range = 0.5
    #print(f"TP={TP[t_range]},FP={FP[t_range]},TN={TN[t_range]},FN={FN[t_range]}" )
    return pd.DataFrame(metrixs)

from scipy.cluster.hierarchy import fclusterdata
def cluster_timestamps_scipy(timestamps, max_distance=1000, aggreate='mean'):
    # Convert timestamps to a 2D array
    timestamps = list(timestamps)
    timestamps_2d = [[t] for t in timestamps]

    # Perform hierarchical clustering
    clusters = fclusterdata(timestamps_2d, t=max_distance, criterion='distance')

    # Convert cluster labels to a list of clusters
    cluster_dict = {}
    for i, label in enumerate(clusters):
        if label not in cluster_dict:
            cluster_dict[label] = []
        cluster_dict[label].append(timestamps[i])

    current_cluster= list(cluster_dict.values())
    if aggreate == 'min':
        return [np.min(t).astype('int') for t in current_cluster]
    elif aggreate == 'mean':
        return [np.mean(t).astype('int') for t in current_cluster]

def long_term_target_plot(args, preded_tensor_pool,
                                target_tensor_pool, 
                                dataset,
                                save_path=None,slots=[0,1,2]):
    
    real_start_pos = args.task.sampling_strategy.valid_sampling_strategy.early_warning + np.arange(5)*(args.DataLoader.Dataset.Resource.basemaxlength+args.DataLoader.Dataset.component_intervel_length) 
    real_start_pos = real_start_pos//3
    expansion = args.model.Embedding.resolution
    unit = 1.0/args.DataLoader.Dataset.Resource.sampling_frequence*expansion
    fig, axeses = plt.subplots(len(slots),len(preded_tensor_pool),figsize=(6*len(preded_tensor_pool),3*len(slots)))
    itemnamelist =  preded_tensor_pool.keys()
    for iii, (slot, axes) in enumerate(zip(slots, axeses)):
        real_item_in_this_row = [target_tensor_pool[itemname][slot] for itemname in itemnamelist]
        pred_item_in_this_row = [preded_tensor_pool[itemname][slot] for itemname in itemnamelist]
        
        for jjj,(name,pred_in_this_row, real_in_this_row, ax) in enumerate(zip(preded_tensor_pool.keys(),pred_item_in_this_row, real_item_in_this_row, axes)):
            ax.set_facecolor('grey')
            real_start_pos = [t for t in real_start_pos if t < len(pred_in_this_row)]
            target_value_sequence = np.zeros(len(pred_in_this_row))
            for value, start in zip(real_in_this_row, real_start_pos):
                target_value_sequence[start:start+args.DataLoader.Dataset.Resource.basemaxlength//expansion] = value
            # Add white rectangles for the assigned regions
            for start_pos in real_start_pos:
                ax.axvspan(start_pos, start_pos + args.DataLoader.Dataset.Resource.basemaxlength//expansion, color='white', alpha=1)
            maxvalue = max(max(target_value_sequence), max(pred_in_this_row))
            origin_curve = dataset[slot]['waveform_seq']
            if 'trend_seq' in dataset[slot]:
                origin_curve+=dataset[slot]['trend_seq']
            origin_curve = origin_curve[:,:expansion]
            origin_curve = origin_curve[::expansion,0]
            origin_curve = origin_curve
            origin_curve = origin_curve/origin_curve.max()*maxvalue
            ax.plot(origin_curve, 'gray', linewidth=1,alpha=0.5)

            ax.plot(pred_in_this_row)
            ax.plot(target_value_sequence,color='r')
            ax.spines['bottom'].set_color('black')
            ax.spines['bottom'].set_linewidth(2)
            ax.spines['left'].set_color('black')
            ax.spines['left'].set_linewidth(2)
            ax.spines['top'].set_color('black')
            ax.spines['top'].set_linewidth(2)
            ax.spines['right'].set_color('black')
            ax.spines['right'].set_linewidth(2)
            if iii==0: ax.set_title(name)
            if jjj==0: ax.set_ylabel(f'sample_{slot}')
    fig.supxlabel(f'Leading Time unit={unit}s', fontsize=16)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
        fig.clear()
        plt.clf()
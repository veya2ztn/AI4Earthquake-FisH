import os
import numpy as np
from scipy.stats import gaussian_kde
from scipy.optimize import root
from mltool.visualization import *
from scipy.optimize import minimize
import scipy.stats as stats
def gauss_mimic_plot(x,y, savepath=None):
    
    plt.figure()
    plt.plot(x,y,'b',label='data')
    fake_y = stats.norm.pdf(x, 0, 1)
    fake_y = fake_y/fake_y.max()
    plt.plot(x,fake_y,'r',label='gauss')
    plt.legend()
    if savepath is not None:
        plt.savefig(savepath)# f'dataset/figures/source_deepth_density_view.png')


def compute_the_meanstds_of_target(metadatapool,root_path=None):
    if root_path is None:
        root_path = "debug/figures"
    if os.path.isfile(root_path):
        root_path = os.path.dirname(root_path)
    for key, full_data in metadatapool.items():
        ## we will compute the good mean and std of the location.
        print("we now calculate the mean/std of the location, notice, you should get this value for training dataset")
        print(f"===============> {key}{full_data.shape} <================")
        if len(full_data.shape) == 1:full_data=full_data[:,None]
        for i in range(full_data.shape[-1]):
            data = full_data[...,i]
            assert len(data.shape)==1, f"{key} data has shape {data.shape} must be 1"
            z_data = data
            z_data = np.squeeze((z_data - z_data.mean())/z_data.std())
            density = gaussian_kde(z_data)
            xs = np.linspace(min(z_data), max(z_data), 2000)
            density.covariance_factor = lambda: .25
            density._compute_covariance()
            x = xs
            y = density(xs)
            y = y/y.max()
            gauss_mimic_plot(x, y, f'{root_path}/source_{key}.{i}_view_with_gauss.png' if root_path is not None else None)
            mean  = data.mean()
            std   = data.std()
            _max  = data.max()
            _min  = data.min()
            print(f"{key}.{i} ==>  mean:{mean:.3f} std:{std:.3f} max:{_max:.3f} min:{_min:.3f}")

    for key in ['magnitude', 'Magnitude']:  # ,'deepth','Depth/Km'
        
        if key not in metadatapool: continue
        data  = metadatapool[key]
        print(f"===============> {key}{data.shape} <================")
        assert (data<=0).any(), f"""
            {key} must big than zero, which means it is underground, 
            [{(data<0).sum()}] slots has negtive z_location which means the quake above the ground.
            we should manually set all the negtive data into 0 for data collection"""
    
        density = gaussian_kde(data)
        xs = np.linspace(min(data), max(data), 2000)
        density.covariance_factor = lambda : .25
        density._compute_covariance()
        x = xs
        y = density(xs)
        max_position_x = x[np.argmax(y)]
        max_position_x = minimize(lambda x:-density(x),max_position_x).x[0]
        y_max = density(max_position_x)
        print("")
        min_x = data.min()
        max_x = data.max()
        
        y = y/y_max
        plt.figure()
        plt.plot(x,y)
        if root_path is not None:
            plt.savefig(f'{root_path}/source_{key}_view.png')

        print(f"The z_offset is set[{max_position_x}]")
        scale = 1
        while True:
            zdata  = np.log((data - max_position_x)/scale + 1)
            if (not np.isnan(zdata).any()) and (not np.isinf(zdata).any()) :break
            print(f"fail for scale={scale}, retry:{scale+1}")
            scale+=1
            if scale > 50: raise
        
        print(f"""The range of z_data is from [{min_x}] to [{max_x}] and we will later use [log((z-z_offset)/{scale}+1)] as the label
            The z_offset is set [{max_position_x}] and notice the output of network is better use a clamp for [log(z_min-z_offset+1)]
            Then we need rescale the value of the log(z) to make it looks like a normal distribution, it can be realized by make the half-bridge
            value be sqrt(2ln(2))=1.177 
            """)

        density = gaussian_kde(zdata)
        xs      = np.linspace(min(zdata),max(zdata),2000)
        density.covariance_factor = lambda : .25
        density._compute_covariance()
        x       = xs
        y       = density(xs)
        max_position_x = x[np.argmax(y)]
        max_position_x = minimize(lambda x: -density(x), max_position_x).x[0]
        y_max = density(max_position_x)
        half_bridge = np.abs( root(lambda x: density(x) - y_max/2, max_position_x).x)[0]
        scaler = abs(max_position_x - half_bridge)/1.177
        print(f"""the maximum position of log(x) at {max_position_x}, should around zero, we still do a shift put the peak at around zero 
                the half bridge is {half_bridge} and the distarnce between {max_position_x}-{half_bridge} should be scale it to 1.177 
                which is {scaler} """
            )
        
        y = y/y_max
        x = (x - max_position_x)/scaler
        gauss_mimic_plot(x, y, f'{root_path}/source_deepth_view_for_{max_position_x}_{half_bridge}_scale_{scale}.png' if root_path is not None else None)
    # ---------------------------------   
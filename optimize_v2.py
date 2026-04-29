from helper_funs import (prof_normalization, check_flip, get_num_rups, gen_params_and_bounds2, 
                         post_process_params, post_process_params_v4,
                          calc_uncertainties, plot_uncert_with_lines)
#from slip_profiles_torch_v4_lbfgs import NN_optimize
from slip_profiles_torch_v4_noclamp import NN_optimize
#from slip_profiles_torch_v2 import NN_optimize
#from slip_profiles_torch_v3_WLoss import NN_optimize
#from slip_profiles_torch_v3_diff_FF import NN_optimize
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt
from copy import deepcopy
   

class Data:
    def __init__(self, data, prof_id, init_p, learn_rate=1e-5, n_epochs=None):
        self.data = data
        self.prof_id = prof_id
        self.init_p = init_p
        self.x, self.y, self.scale_shift = self._normalize()
        self.flip = self._flip()
        self.n_dim = self.y.shape[1]
        self.n_rup = None

        self.lr = learn_rate
        self.n_epochs = n_epochs
        
        self.param_0, self.param_bounds = None, None

    def _normalize(self):
        # Normaliztion of smoothed data
        data_norm, shift, scale = prof_normalization(self.data)
        scale_shift = (scale, shift)
        data_norm_loc  = data_norm[:,0]
        data_norm_disp = data_norm[:,1:]
        data_norm_disp = data_norm_disp[:,np.newaxis] if len(data_norm_disp.shape) == 1 else data_norm_disp

        return data_norm_loc, data_norm_disp, scale_shift
    
    def _flip(self):
        flip = check_flip(self.y[:,0])
        if flip < 0:
            self.y *= flip
        
        return flip
    
    def _rup(self, peaks):
        # if len(peaks) < 4 and len(peaks) > 0: 
        #     n_rup = 2
        #     peaks = peaks[:2]
        # else:
        #     n_rup = 1

        n_rup = len(peaks) if 0 < len(peaks) < 4 else 1
        if self.init_p is not None and len(self.init_p[0]['loc']) != n_rup:
            self.init_p = None

        #return n_rup, peaks
        return 1, peaks
    
    def _gen_params(self, peaks, rand):
        self.param_0, self.param_bounds = self.init_p if self.init_p is not None else gen_params_and_bounds2(self.x, self.y, peaks, rand=rand)


def fit_data(input):
    data_obj, peaks1, peaks2, rand, history = input
    data_obj.n_rup, peaks = data_obj._rup(peaks1)
    data_obj._gen_params(peaks, rand)
    model, losses = NN_optimize(data_obj, history)

    # Secondary fit with tighter peak threshold (lesser # of peaks) if len(peaks1) >= 4
    if peaks2:
        n_rup = data_obj._rup(peaks2)
        data_obj._gen_params(peaks2, rand)
        model2, losses2 = NN_optimize(data_obj, history)

        if losses2['total_loss'][-1] < losses['total_loss'][-1]:
            model, losses = model2, losses2
            data_obj.n_rup = n_rup

    return model, losses, data_obj


def run_optimization(data, orig_data, prof_num, sigma, rand=False, uncert=False, coords=None, win_bounds=None, init_p=None, history=False):
    # Normaliztion of non-smoothed original data
    data_norm_orig, _, _ = prof_normalization(orig_data)
    data_norm_loc_orig  = data_norm_orig[:,0]
    data_norm_disp_orig = data_norm_orig[:,1:]
    data_norm_disp_orig = data_norm_disp_orig[:,np.newaxis] if len(data_norm_disp_orig.shape) == 1 else data_norm_disp_orig

    data_obj = Data(data, prof_num, init_p)
    _, peaks1, peaks2 = get_num_rups(data_obj.y[:,0], sigma)
    
    # Fit with standard initial parameters based on the data
    if not rand:
        model, losses, data_obj = fit_data([data_obj, peaks1, peaks2, rand, history])     
        
    # Fit with 10 sets of random inital parameters to find an optimal fit
    else:
        min_loss = np.inf
        inputs = [[deepcopy(data_obj), peaks1, peaks2, rand, history] for _ in range(10)]
        pool = Pool()
        out = pool.map(fit_data, inputs)
        pool.close()
        pool.join()

        for o in out:
            if o[1]['total_loss'][-1] < min_loss:
                model, losses, data_obj = o
                min_loss = o[1]['total_loss'][-1]

    if history:
        return model, losses, data_obj.x, data_obj.y, data_obj.scale_shift

    table, norm_vals, scaled_vals, lin_seg, _, org_p = post_process_params_v4(data_obj, model, coords)
    data_obj.init_p = (org_p, data_obj.param_bounds) if init_p else None

    if uncert:
        i = 0
        uncertainties = []
        data = np.hstack([(data_obj.x/data_obj.scale_shift[0]).reshape(-1,1), data_obj.y])

        for loc in org_p['loc']:
            u, p1,  p99 = calc_uncertainties(data[lin_seg[i][0]:lin_seg[i][1]], data[lin_seg[i+1][0]:lin_seg[i+1][1]], loc.item(), win_bounds, 1000)
            u_fig = plot_uncert_with_lines(u, data_obj.x/data_obj.scale_shift[0], data_obj.y, scaled_vals.detach().numpy(), p1, p99)
            uncertainties.append(u_fig)
            i += 2
    else:
        uncertainties = None

    if data_obj.n_dim > 1:
        fig, ax = plt.subplots(figsize=(25,15), nrows=1, ncols=2)

        # ax[0].plot(data_norm_orig[:,0], data_norm_orig[:,1], 'o')
        # ax[0].plot(data_norm_orig[:,0], data_obj.flip*norm_vals[:,0], '-')
        # ax[0].set_title("First Dimension, Normalized")
        # ax[0].set_xlabel("Horizontal Distance")
        # ax[0].set_ylabel("Displacement")

        ax[0].plot(orig_data[:,0]*3, orig_data[:,1], 'o')
        ax[0].plot(orig_data[:,0]*3, scaled_vals[:,0], '-', linewidth=3.5)
        ax[0].set_title("First Dimension (Parallel)", fontdict={'size': 25})
        ax[0].set_xlabel("Distance Along Profile (m)", fontdict={'size': 35})
        ax[0].set_ylabel("Fault Displacement (m)", fontdict={'size': 35})
        ax[0].grid(True, linewidth=1.8)
        ax[0].tick_params(axis='both', labelsize=15)

        # ax[1].plot(data_norm_orig[:,0], data_norm_orig[:,2], 'o')
        # ax[1].plot(data_norm_orig[:,0], data_obj.flip*norm_vals[:,1], '-')
        # ax[1].set_title("Second Dimension, Normalized")
        # ax[1].set_xlabel("Horizontal Distance")
        # ax[1].set_ylabel("Displacement")

        ax[1].plot(orig_data[:,0]*3, orig_data[:,2], 'o')
        ax[1].plot(orig_data[:,0]*3, scaled_vals[:,1], '-', linewidth=3.5)
        ax[1].set_title("Second Dimension (Perpendicular)",fontdict={'size': 25})
        ax[1].set_xlabel("Distance Along Profile (m)", fontdict={'size': 35})
        ax[1].set_ylabel("Fault Displacement (m)", fontdict={'size': 35})
        ax[1].grid(True, linewidth=1.8)
        ax[1].tick_params(axis='both', labelsize=15)
    else:
        fig, ax = plt.subplots(figsize=(20,15), nrows=1, ncols=2)

        ax[0].plot(data_norm_orig[:,0], data_norm_orig[:,1], 'o')
        ax[0].plot(np.linspace(data_norm_orig[0,0], data_norm_orig[-1,0], len(norm_vals)), data_obj.flip*norm_vals, '-',linewidth=3)
        # ax.plot(data_obj.x, data_obj.y, 'o')
        # ax.plot(np.linspace(data_obj.x[0], data_obj.x[-1], len(norm_vals)), data_obj.flip*norm_vals, '-')
        ax[0].grid(True, linewidth=1.8)
        ax[0].tick_params(axis='both', labelsize=20)
        ax[0].set_title("Normalized")
        ax[0].set_xlabel("Normalized Distance Along Profile", fontdict={'size': 33})
        ax[0].set_ylabel("Displacement", fontdict={'size': 33})

        # ax[1].plot(data[:,0], data[:,1], 'o')
        # ax[1].plot(np.linspace(data[0,0], data[-1,0], len(scaled_vals)), scaled_vals, '-')
        ax[1].plot(orig_data[:,0], orig_data[:,1], 'o')
        ax[1].plot(np.arange(len(scaled_vals)), scaled_vals, '-', linewidth=3)
        #ax[1].plot(np.arange(len(norm_vals)), norm_vals, '-', linewidth=3)
        #ax.plot(np.linspace(data[0,0], data[-1,0], len(scaled_vals)), scaled_vals, '-')
        ax[1].set_title("Rescaled")
        ax[1].grid(True, linewidth=1.8)
        ax[1].tick_params(axis='both', labelsize=20)
        ax[1].set_xlabel("Distance Along Profile", fontdict={'size': 33})
        ax[1].set_ylabel("Displacement", fontdict={'size': 33})

    
    return table, fig, model, uncertainties, losses, data_obj.init_p
from helper_funs import (prof_normalization, gen_params_and_bounds2, 
                         post_process_params, post_process_params_v4,
                          calc_uncertainties, plot_uncert_with_lines)
#from slip_profiles_torch_v4_lbfgs import NN_optimize
from slip_profiles_torch_v4_noclamp import NN_optimize, SlipProfileNN
#from slip_profiles_torch_v2 import NN_optimize
#from slip_profiles_torch_v3_WLoss import NN_optimize
#from slip_profiles_torch_v3_diff_FF import NN_optimize
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt
from copy import deepcopy
import torch
from scipy.signal import find_peaks
from numpy.lib.stride_tricks import sliding_window_view

s_0 = 0.0
c_0 = np.array([-1])
s_r = np.array([0.5])
d_r = np.array([[1]])
ramp = np.array([0])
c_r = np.vstack([0])

# a_r = [30, 70, 120, 170, 250]
a_r = [30, 130, 240]

rup_param_bounds = {
    'origin': [-1.1, 1.1],
    'loc' : [0.1, 0.9],
    'width' : [15., 300.],
    'disp' : [0.1, 2.2],
    'ramp': [-30, 30],
    'slope': [-30, 30]
}

model_for_rup_det = SlipProfileNN(1,1,seed_origin=c_0,
                    seed_ramp=ramp, seed_loc=s_r,
                    seed_width=np.array([15]),
                    seed_disp=d_r, seed_slope=c_r,
                    bounds=rup_param_bounds)

def map_peaks(peaks, data_range):
    groups = []
    
    for loc in peaks:
        mapped_value = round((loc)*99/data_range) # map from 0 - len(data_disp) to 0 - 99
        for g in range(len(groups)):
            group = groups[g]
            if abs(group[1]/group[0] - mapped_value) <= 5: # threshold for grouping is 5
                groups[g] = (group[0] + 1, group[1] + mapped_value)
                break
        else:
            groups.append((1, mapped_value))

    groups = [int((group[1]/group[0])*data_range/99) for group in groups if group[0] >= 2]

    return len(groups), groups

def _normalize(data):
    # Normaliztion of smoothed data
    data_norm, shift, scale = prof_normalization(data)
    scale_shift = (scale, shift)
    data_norm_loc  = data_norm[:,0]
    data_norm_disp = data_norm[:,1:]
    data_norm_disp = data_norm_disp[:,np.newaxis] if len(data_norm_disp.shape) == 1 else data_norm_disp

    return data_norm_loc, data_norm_disp, scale_shift

def _flip(y):
    num_points = int(0.25*len(y))
    if np.mean(y[:num_points]) > np.mean(y[len(y)-num_points:]):
        return -1
    return 1

def _rup(y):
    window_size_factor = [0.1, 0.2, 0.25]
    all_peaks = []
    A_LB = 15
    A_UB = 300
    N = len(y)

    for w in window_size_factor:
        window_size = int(w*N)
        window_x = torch.linspace(0, 1, window_size)[:, None]
        scores = []

        windows = sliding_window_view(y, window_size)
        windows = windows[:-1]
        windows_avg = windows - windows.mean(axis=1, keepdims=True)
        profiles = []
        

        for a in a_r:
            transf_width = np.clip((a - A_LB) / (A_UB - A_LB), 1e-8, 1-1e-8)
            transf_width = np.log(transf_width/(1-transf_width))

            with torch.no_grad():
                model_for_rup_det.prof.r0.width.copy_(torch.tensor(transf_width, dtype=torch.float32))
                profile = model_for_rup_det(window_x).detach().numpy().flatten()
            profile_avg = profile -  np.mean(profile)
            profiles.append(profile_avg)

        profiles = np.asarray(profiles, dtype=np.float32)
        corr_matrix = windows_avg @ profiles.T
        scores = corr_matrix.max(axis=1)            
        scores = (scores - min(scores)) / (max(scores) - min(scores))

        peaks,_ = find_peaks(scores)
        peaks = [p+(window_size//2) for p in peaks if scores[p] > 0.55]
        #peaks = [p for p in peaks if scores[p] > 0.55]
        all_peaks.extend(peaks)

        # fig, ax = plt.subplots(nrows=2, ncols=1)
        # ax[0].plot(y)
        # ax[1].plot(scores, '-')
        # ax[1].plot(peaks, scores[peaks], 'x')
        # plt.show()

    return map_peaks(all_peaks, N)
   

class Data:
    def __init__(self, data, prof_id, init_p, rand=False, learn_rate=1e-5, n_epochs=None):
        self.data = data
        self.prof_id = prof_id
        self.init_p = init_p
        self.x, self.y, self.scale_shift = _normalize(self.data)

        self.flip = _flip(self.y[:,0])
        self.y *= self.flip

        self.n_dim = self.y.shape[1]
        self.n_rup, self.rup_locs = _rup(self.y[:,0])
        #self.n_rup, self.rup_locs = 1, []
        self.param_0, self.param_bounds = init_p if init_p is not None else gen_params_and_bounds2(self.x,
                                                                                                   self.y,
                                                                                                   self.rup_locs,
                                                                                                   rand=rand)

        self.lr = learn_rate
        self.n_epochs = n_epochs


def fit_data(input):
    data_obj, history = input
    model, losses = NN_optimize(data_obj, history)

    return model, losses, data_obj


def run_optimization(data, orig_data, prof_num, sigma, rand=False, uncert=False, coords=None, win_bounds=None, init_p=None, history=False):
    # Normaliztion of non-smoothed original data
    data_norm_orig, _, _ = prof_normalization(orig_data)
    data_norm_loc_orig  = data_norm_orig[:,0]
    data_norm_disp_orig = data_norm_orig[:,1:]
    data_norm_disp_orig = data_norm_disp_orig[:,np.newaxis] if len(data_norm_disp_orig.shape) == 1 else data_norm_disp_orig

    data_obj = Data(data, prof_num, init_p, rand)

    # Fit with standard initial parameters based on the data
    if not rand:
        model, losses, data_obj = fit_data([data_obj, history])     
        
    # Fit with 10 sets of random inital parameters to find an optimal fit
    else:
        min_loss = np.inf
        inputs = [[deepcopy(data_obj), history] for _ in range(10)]
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

        ax[0].plot(data_norm_orig[:,0], data_norm_orig[:,1], 'o')
        ax[0].plot(data_norm_orig[:,0], data_obj.flip*norm_vals[:,0], '-')
        ax[0].set_title("First Dimension, Normalized")
        ax[0].set_xlabel("Horizontal Distance")
        ax[0].set_ylabel("Displacement")

        # ax[0].plot(orig_data[:,0]*3, orig_data[:,1], 'o')
        # ax[0].plot(orig_data[:,0]*3, scaled_vals[:,0], '-', linewidth=3.5)
        # ax[0].set_title("First Dimension (Parallel)", fontdict={'size': 25})
        # ax[0].set_xlabel("Distance Along Profile (m)", fontdict={'size': 35})
        # ax[0].set_ylabel("Fault Displacement (m)", fontdict={'size': 35})
        # ax[0].grid(True, linewidth=1.8)
        # ax[0].tick_params(axis='both', labelsize=15)

        ax[1].plot(data_norm_orig[:,0], data_norm_orig[:,2], 'o')
        ax[1].plot(data_norm_orig[:,0], data_obj.flip*norm_vals[:,1], '-')
        ax[1].set_title("Second Dimension, Normalized")
        ax[1].set_xlabel("Horizontal Distance")
        ax[1].set_ylabel("Displacement")

        # ax[1].plot(orig_data[:,0]*3, orig_data[:,2], 'o')
        # ax[1].plot(orig_data[:,0]*3, scaled_vals[:,1], '-', linewidth=3.5)
        # ax[1].set_title("Second Dimension (Perpendicular)",fontdict={'size': 25})
        # ax[1].set_xlabel("Distance Along Profile (m)", fontdict={'size': 35})
        # ax[1].set_ylabel("Fault Displacement (m)", fontdict={'size': 35})
        # ax[1].grid(True, linewidth=1.8)
        # ax[1].tick_params(axis='both', labelsize=15)
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
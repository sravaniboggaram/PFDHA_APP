import h5py
import numpy as np
from pylib_general import gaussian_convolution_nonuniform
from optimize_v3 import run_optimization
import csv
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from slip_profiles_torch_v4_noclamp import SlipProfileNN, param_act_to_transf, param_transf_to_act
import torch
from numpy.lib.stride_tricks import sliding_window_view
import tracemalloc
from helper_funs import split_profile


def create_profile_family():
    s_0 = 0.0
    c_0 = np.array([-1])
    s_r = np.array([0.5])
    d_r = np.array([[1]])
    ramp = np.array([0])
    c_r = np.vstack([0])

    a_r = [15, 50, 120, 170, 250]
    models = []

    for a in a_r:
        a_v = np.array([a])
        model = SlipProfileNN(1,1,seed_origin=c_0,
                    seed_ramp=ramp, seed_loc=s_r,
                    seed_width=a_v, seed_disp=d_r, seed_slope=c_r)
        models.append(model)

    return models

def map_peaks2(peaks, data_range):
    tracemalloc.start()
    groups = []

    for loc in peaks:
        mapped_value = round((loc)*99/data_range)
        for group in groups:
            center = np.mean(group)
            if abs(center - mapped_value) <= 5:
                group.append(mapped_value)
                break
        else:
            groups.append([mapped_value])

    final_peak_locs = []
    n_peaks = 0
    print("GROUPS", groups)
    for group in groups:
        if len(group) >= 2:
            n_peaks += 1
            final_peak_locs.append(int(np.mean(group)*data_range/99))

    current, peak_m = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("FINAL", final_peak_locs)
    print("CURRENT MEM ", current)
    print("PEAK ", peak_m)

    return n_peaks, final_peak_locs

def map_peaks3(peaks, data_range):
    tracemalloc.start()
    groups = []
    
    for loc in peaks:
        mapped_value = round((loc)*99/data_range)
        for g in range(len(groups)):
            group = groups[g]
            if abs(group[1]/group[0] - mapped_value) <= 5:
                groups[g] = (group[0] + 1, group[1] + mapped_value)
                break
        else:
            groups.append((1, mapped_value))

    #final_peak_locs = []
    print("GROUPS", groups)
    groups = [int((group[1]/group[0])*data_range/99) for group in groups if group[0] >= 2]
    
    current, peak_m = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("FINAL ", groups)
    print("CURRENT MEM ", current)
    print("PEAK ", peak_m)
    return len(groups), groups
    

def map_peaks(peaks, final_range):
    groups = []
    for range_len, peak_locs in peaks.items():
        for peak in peak_locs:
            mapped_value = round((peak)*99/range_len)
            for group in groups:
                center = np.mean(group)
                if abs(center - mapped_value) <= 5:
                    group.append(mapped_value)
                    break
            else:
                groups.append([mapped_value])

    print("Groups", groups)
    final_peak_locs = []
    n_peaks = 0
    for group in groups:
        if len(group) >= 2:
            n_peaks += 1
            final_peak_locs.append(int(np.mean(group)*final_range/99))

    print("FINAL", final_peak_locs)

    return n_peaks, final_peak_locs
    
    # return sum(len(group) >= 2 for group in groups)
    


def create_noise_profiles(x):
    noise = 3*np.pow(x,2)
    return noise - np.mean(noise)


def combine_parallel_perp_files():
    file_paths = ['eaf_parallel_profiles_6273_0-999.h5', 'eaf_perp_6273_0-999.h5']
    output_path = 'combined_turkey_0-999.h5'

    with h5py.File(output_path, 'w') as out_f:
        for fp in file_paths:
            with h5py.File(fp, 'r') as in_f:
                # Create a group named after the file (or a custom name)
                group_name = fp.split('.')[0] 
                dest_group = out_f.create_group(group_name)
                
                # Copy all content recursively
                for key in in_f.keys():
                    in_f.copy(key, dest_group)

def reformat_parallel_file():
    file_path = "eaf_parallel_profiles_6273_0-999.h5"
    ouput_path = "single_group_eaf_parallel_0_999.h5"

    with h5py.File(ouput_path,'w') as out_f:
        with h5py.File(file_path, 'r') as in_f:
            group = out_f.create_group('parallel')

            for key in in_f.keys():
                in_f.copy(key, group)



def two_dim():
    path = "combined_turkey_0-999.h5"
    f = h5py.File(path, 'r')
    dirs = list(f.keys())
    parallel_group, perp_group = f[dirs[0]], f[dirs[1]]

    parallel_keys = list(parallel_group.keys())
    perp_keys = list(perp_group.keys())

    ds = np.stack([np.array(parallel_group[parallel_keys[482]]),
                   np.array(perp_group[perp_keys[482]])],
                   axis=0)
    print(ds.shape)
    return

    profiles = range(481,483)

    for i in range(1):
    #for id in profiles:
        ds1 = parallel_group[parallel_keys[482]]
        parallel_ds = ds1[:,np.any(~np.isnan(ds1), axis=0)]

        ds2 = perp_group[perp_keys[482]]
        perp_ds = ds2[:,np.any(~np.isnan(ds2), axis=0)]

        y1 = np.nanmean(parallel_ds, axis=0)
        y2 = np.nanmean(perp_ds, axis=0)
        x = np.arange(len(y1))

        data = np.array([x, y1, y2]).T

        smooth_y1 = gaussian_convolution_nonuniform(x, y1, sigma_x=20)
        smooth_y2 = gaussian_convolution_nonuniform(x, y2, sigma_x=20)
        smooth_data = np.array([x, smooth_y1, smooth_y2]).T
        print(smooth_data.shape)

        # table, fig, model, u, losses, init_p = run_optimization(smooth_data, 
        #                                                         data, 
        #                                                         0, 
        #                                                         None, 
        #                                                         False, 
        #                                                         None, 
        #                                                         None, 
        #                                                         None)
        
        # plt.show()
        #fig.savefig(f"prof 482 2_dim_test.png")

#two_dim()

def check_for_dips(data):
    n = len(data)
    start = data[:int(0.33*n)]
    end = data[-int(0.33*n):]

    start_diff = np.gradient(start)
    end_diff = np.gradient(end)

    fig, ax = plt.subplots(nrows=3, ncols=1)

    ax[0].plot(data)
    ax[1].plot(start_diff)
    ax[2].plot(end_diff)

    plt.show()

def two_rup(p):
    path = "eaf_parallel_profiles_6273_0-999.h5"
    f = h5py.File(path, 'r')
    keys = list(f.keys())

    s_0 = 0.0
    c_0 = np.array([-1])
    s_r = np.array([0.5])
    d_r = np.array([[1]])
    ramp = np.array([0])
    c_r = np.vstack([0])

    # a_r = [30, 70, 120, 170, 250]
    a_r = [30, 130, 240]

    param_bounds = {
        'origin': [-1.1, 1.1],
        'loc' : [0.1, 0.9],
        'width' : [15., 300.],
        'disp' : [0.1, 2.2],
        'ramp': [-30, 30],
        'slope': [-30, 30]
    }

    model = SlipProfileNN(1,1,seed_origin=c_0,
                          seed_ramp=ramp, seed_loc=s_r,
                          seed_width=np.array([15]),
                          seed_disp=d_r, seed_slope=c_r,
                          bounds=param_bounds)


    for i in range(1):
    #for id in profiles:
        key = keys[400]
        print("KEY ", key, int(key[key.rfind("_") + 1:]))
        ds1 = f[key]
        ds1 = ds1[:,np.any(~np.isnan(ds1), axis=0)]

        y1 = np.nanmean(ds1, axis=0)
        n_points = len(y1)
        x = np.arange(n_points)

        if n_points < 450:
            x_interp = np.linspace(0, n_points, 1000)
            y_interp = np.interp(x_interp, x, y1)
            data = np.array([x_interp, y_interp]).T

            x,y1 = x_interp, y_interp
        else:
            data = np.array([x, y1]).T

        smooth_y1 = gaussian_convolution_nonuniform(x, y1, sigma_x=25)
        if np.mean(smooth_y1[:50]) > np.mean(smooth_y1[-50:]):
            smooth_y1 *= -1

        window_size_factor = [0.1, 0.2, 0.25]
        all_peaks = {}
        A_LB = 15
        A_UB = 300
        test_peaks = []

        for w in window_size_factor:
            window_size = int(w*len(smooth_y1))
            window_x = torch.linspace(0, 1, window_size)[:, None]
            scores = []

            windows = sliding_window_view(smooth_y1, window_size)
            windows = windows[:-1]
            windows_avg = windows - windows.mean(axis=1, keepdims=True)
            profiles = []
            

            for a in a_r:
                transf_width = np.clip((a - A_LB) / (A_UB - A_LB), 1e-8, 1-1e-8)
                transf_width = np.log(transf_width/(1-transf_width))

                with torch.no_grad():
                    model.prof.r0.width.copy_(torch.tensor(transf_width, dtype=torch.float32))
                    profile = model(window_x).detach().numpy().flatten()
                profile_avg = profile -  np.mean(profile)
                profiles.append(profile_avg)

            profiles = np.asarray(profiles, dtype=np.float32)
            corr_matrix = windows_avg @ profiles.T
            scores = corr_matrix.max(axis=1)            
            scores = (scores - min(scores)) / (max(scores) - min(scores))

            peaks,_ = find_peaks(scores)
            peaks = [p+(window_size//2) for p in peaks if scores[p] > 0.55]
            #peaks = [p for p in peaks if scores[p] > 0.55]
            test_peaks.extend(peaks)
            all_peaks[len(scores)] = peaks

            fig, ax = plt.subplots(nrows=2, ncols=1)
            ax[0].plot(smooth_y1)
            ax[1].plot(scores, '-')
            ax[1].plot(peaks, scores[peaks], 'x')
            plt.show()

        n_peaks, final_peaks_locs = map_peaks3(test_peaks, len(smooth_y1))
        print("NPEAKS", n_peaks)
        print("FINAL PEAK LOCS ", final_peaks_locs)

        plt.plot(smooth_y1)
        plt.plot(final_peaks_locs, smooth_y1[final_peaks_locs], 'x', markersize=10)
        plt.show()

        #smooth_data = np.array([x, smooth_y1]).T



        # table, fig, model, u, losses, init_p = run_optimization(smooth_data, 
        #                                                         data, 
        #                                                         int(key[key.rfind("_") + 1:]), 
        #                                                         None, 
        #                                                         False, 
        #                                                         None, 
        #                                                         None, 
        #                                                         None)
        
        # fig.savefig(f"prof 158 burma 2_rup_test.png")

#two_rup(1)


def create_prof_ids_csv():
    path = "eaf_parallel_profiles_6273_3590-6272.h5"

    profile_ints = []
    file = h5py.File(path, 'r')

    print(file.keys())

    for key in file.keys():
        profile_ints.append(int(key[key.rfind("_") + 1:]))

    with open("prof_ids.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows([[prof_id] for prof_id in profile_ints])

    f.close()


def test_single_prof():
    path = "eaf_parallel_profiles_6273_0-999.h5"
    f = h5py.File(path, 'r')
    keys = list(f.keys())
    key = keys[401]
    ys = []

    ds1 = f[key]
    ds2 = None

    ds1 = ds1[:,np.any(~np.isnan(ds1), axis=0)]
    ys.append(np.nanmean(ds1, axis=0))
    if ds2 is not None:
        ds2 = ds2[:,np.any(~np.isnan(ds2), axis=0)]
        ys.append(np.nanmean(ds2, axis=0))
   
    n_points = len(ys[0])
    print("LEN ", len(ys[0]))

    # #FIX!!!!!!!!
    sigma_x = {}
    x = np.arange(n_points)

    if n_points < 450:
        x_interp = np.linspace(0, n_points-1, 1000)
        ys = [np.interp(x_interp, x, y) for y in ys]
        x = x_interp
    
    smooth_ys = [gaussian_convolution_nonuniform(x, y, sigma_x=20)
                 for y in ys]
    
    data = np.column_stack((x, *ys))
    smooth_data = np.column_stack((x, *smooth_ys))

    table, fig, model, u, losses, init_p = run_optimization(smooth_data, 
                                                                data, 
                                                                int(key[key.rfind("_") + 1:]), 
                                                                None, 
                                                                False, 
                                                                None, 
                                                                None, 
                                                                None)
    
    for n, p in model.named_parameters():
        print(n, p.item())

    plt.show()

test_single_prof()

def test_rescaling():
    # prof.r0.loc
    # prof.r0.width
    # prof.r0.prof.d0.disp
    # prof.r0.prof.d0.slope
    # base.d0.ramp
    # base.d0.origin

    scale = 1/1680

    c_0 = np.array([-1])
    s_r = np.array([0.4])
    d_r = np.array([[1]])
    ramp = np.array([0])
    c_r = np.vstack([0])
    a_r = np.array([100])

    param_bounds = {
        'origin': [-1.1, 1.1],
        'loc' : [0.1, 0.9],
        'width' : [15., 300.],
        'disp' : [0.1, 2.2],
        'ramp': [-30, 30],
        'slope': [-30, 30]
    }

    rescale = {
        'width': lambda v: v * scale,
        'loc': lambda v: v / scale,
        'disp': lambda v: v,
        'ramp': lambda v: v *scale,
    }

    model = SlipProfileNN(1,1,seed_origin=c_0,
                          seed_ramp=ramp, seed_loc=s_r,
                          seed_width=a_r,
                          seed_disp=d_r, seed_slope=c_r,
                          bounds=param_bounds)
    
    print("Transf loc ", model.prof.r0.loc)
    raw_norm_loc = param_transf_to_act(model.prof.r0.loc.item(), param_bounds['loc'])
    print("Raw norm loc ", raw_norm_loc)

    raw_scaled_loc = rescale['loc'](raw_norm_loc)
    print("Raw scaled loc: ", raw_scaled_loc)
    raw_scaled_bounds = [0., 500.]

    transf_scaled_loc = param_act_to_transf(raw_scaled_loc, raw_scaled_bounds)
    print("Transf scaled loc ", transf_scaled_loc)


# from slip_profiles import fun_slip_profile
# import numpy as np
# import matplotlib.pyplot as plt
# import h5py
# from slip_profiles_torch_v2 import SlipProfileNN
# import torch


# # No clamp 223 params

# #rupture reference point
# s_0 = 0.0
# c_0 = np.array([-3.983063658325618])
# #rupture location
# s_r = np.array([834.7894842549067])
# #displacement values (n_rup x n_dim)
# d_r = np.array([[2.926184568707388]])
# #curvature
# a_r = np.array([0.03335785983591074])
# #slopes
# ramp = np.array([0.0014085104248143796])
# c_r = np.vstack([-3.1283368389730686e-05])

# # v2 223 parameters

# #rupture reference point
# # s_0 = 0.0
# # c_0 = np.array([-4.349568588082891])
# # #rupture location
# # s_r = np.array([0.4484162704911325])
# # #displacement values (n_rup x n_dim)
# # d_r = np.array([[6.9593367594253195]])
# # #curvature
# # a_r = np.array([28.859404044963004])
# # #slopes
# # ramp = np.array([-1.9244564830659918])
# # c_r = np.vstack([-1.3013880812419574])

# def rmse(y_pred, y_act):
#     return torch.sqrt(torch.mean((y_pred - y_act)**2)).detach().numpy()

# def L1(y_pred, y_act):
#     return torch.mean(torch.abs(y_pred - y_act)).detach().numpy()


# model = SlipProfileNN(1,1,seed_origin=c_0,
#                       seed_ramp=ramp, seed_loc=s_r,
#                       seed_width=a_r, seed_disp=d_r, seed_slope=c_r)


# #print(y)

# path = "eaf_parallel_profiles_6273_1000-1999.h5"
# f = h5py.File(path, 'r')
# keys = list(f.keys())
# ds = f[keys[290]]

# ds1 = ds[:,np.any(~np.isnan(ds), axis=0)]
# y1 = np.nanmean(ds1, axis=0)

# s_arr = np.arange(len(y1))
# #y = fun_slip_profile(s_arr, s_0, c_0, s_r, a_r, d_r, c_r, 1, 1)

# model_y = model(torch.tensor(s_arr, dtype=torch.float64).unsqueeze(0).T)

# print("LOSS", rmse(model_y, torch.tensor(y1, dtype=torch.float64)))

# # ramp_region = np.linspace(0,0.2,100)
# # ramp_region_y = -1.7306958825538956*ramp_region - 1.8735719724266264


# plt.plot(s_arr, model_y.detach().numpy(), label='v2 model')
# plt.plot(s_arr, y1, label='actual data')
# #plt.plot(ramp_region, ramp_region_y, label='ramp')
# plt.legend()
# plt.show()

import h5py
import numpy as np
from pylib_general import gaussian_convolution_nonuniform
from optimize_v2 import run_optimization
import csv
import matplotlib.pyplot as plt

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


def two_dim():
    path = "combined_turkey_0-999.h5"
    f = h5py.File(path, 'r')
    dirs = list(f.keys())
    print(f.keys())
    parallel_group, perp_group = f[dirs[0]], f[dirs[1]]

    parallel_keys = list(parallel_group.keys())
    perp_keys = list(perp_group.keys())

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

        table, fig, model, u, losses, init_p = run_optimization(smooth_data, 
                                                                data, 
                                                                0, 
                                                                None, 
                                                                False, 
                                                                None, 
                                                                None, 
                                                                None)
        
        fig.savefig(f"prof 482 2_dim_test.png")

def two_rup():
    path = "burma_20250407_parallel_profiles_483_0-483.h5"
    f = h5py.File(path, 'r')
    keys = list(f.keys())

    for i in range(1):
    #for id in profiles:
        key = keys[158]
        ds1 = f[key]
        ds1 = ds1[:,np.any(~np.isnan(ds1), axis=0)]

        y1 = np.nanmean(ds1, axis=0)
        x = np.arange(len(y1))

        data = np.array([x, y1]).T

        smooth_y1 = gaussian_convolution_nonuniform(x, y1, sigma_x=20)
        smooth_data = np.array([x, smooth_y1]).T

        table, fig, model, u, losses, init_p = run_optimization(smooth_data, 
                                                                data, 
                                                                int(key[key.rfind("_") + 1:]), 
                                                                None, 
                                                                False, 
                                                                None, 
                                                                None, 
                                                                None)
        
        fig.savefig(f"prof 158 burma 2_rup_test.png")

two_rup()

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


# path = "eaf_parallel_profiles_6273_2000-2992.h5"
# file = h5py.File(path)
# keys = list(file.keys())

# print(file[keys[3]])

# ds1 = file[keys[430]]
# ds1 = ds1[:,np.any(~np.isnan(ds1), axis=0)]
# y1 = np.nanmean(ds1, axis=0)
# x = np.arange(len(y1))*3

# plt.plot(x,y1)
# plt.xlabel("Distance Along Profile (m)")
# plt.ylabel("Fault Displacement (m)")
# plt.show()
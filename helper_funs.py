import torch
import numpy as np
from scipy import stats
import h5py
from scipy.signal import find_peaks
from scipy.signal import peak_prominences, peak_widths
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from pylib_general import gaussian_convolution_nonuniform
from scipy.signal import find_peaks_cwt
from numpy.random import uniform
import random
import pandas as pd
import re
import utm
import matplotlib.pyplot as plt
import pathlib
  


def gen_params_and_bounds2(data_loc, data_disp, peak_locs, rand=False):
    # data_loc is the normalized 0-1 x range
    n_rup = len(peak_locs) if len(peak_locs) > 0 else 1

    n_pt, n_dim = data_disp.shape

    i_str = np.argmin(data_loc)
    i_end = np.argmax(data_loc)

    s_str = data_loc[i_str]
    s_end = data_loc[i_end]

    b = 10

    if n_dim == 1:
        # 1D Case
        d_par_min   = np.min(data_disp, axis=0)[0]
        d_nor_min   = np.inf
        d_par_max   = np.max(data_disp, axis=0)[0]
        d_nor_max   = -np.inf
        d_par_width = d_par_max - d_par_min
        d_nor_width = -np.inf

        d_par_str = data_disp[i_str][0]
        d_nor_str = None

        ramp_par_val = (data_loc[b] - d_par_str)/b
        ramp_nor_val = None
        slope_par_val = (data_disp[-b] - data_disp[i_end])/(data_loc[-b] - s_end)
        slope_nor_val = None
        
    if n_dim == 2:
        # 2D Case
        d_par_min = np.min(data_disp[:,0])
        d_par_max = np.max(data_disp[:,0])
        d_nor_min = np.min(data_disp[:,1])
        d_nor_max = np.max(data_disp[:,1])
        d_par_width = d_par_max - d_par_min
        d_nor_width = d_nor_max - d_nor_min

        print(d_par_max, d_par_min)
        
        d_par_str = data_disp[i_str][0]
        d_nor_str = data_disp[i_str][1]

        ramp_par_val = stats.linregress(data_loc[:b], data_disp[:,0][:b]).slope
        ramp_nor_val = stats.linregress(data_loc[:b], data_disp[:,1][:b]).slope

        slope_par_val = stats.linregress(data_loc[-b:], data_disp[:,0][-b:]).slope
        slope_nor_val = stats.linregress(data_loc[-b:], data_disp[:,1][-b:]).slope


    c0_lb = min(d_par_min,d_nor_min)
    s_lb = 0.1
    #d_lb = 0.2*max(d_par_width,d_nor_width)
    d_lb = 1e-6
    a_lb = 15.
    ramp_lb = -30
    slope_lb = -30

    c0_ub = max(d_par_max,d_nor_max)
    s_ub = 0.9
    d_ub = 1.*max(d_par_width,d_nor_width)
    a_ub = 300.
    ramp_ub = 30
    slope_ub = 30

    d_fact = 0.2

    rand = int(rand)
    r1, r2 = 1-rand, rand

    res = {}
    #res["loc"]    = [r1*i*(s_end-s_str)/(n_rup+1) + r2*uniform(s_str, s_end) for i in range(1, n_rup+1)]
    if len(peak_locs) > 0 and len(peak_locs) < 4: # filter prominent peaks
        res["loc"]    = [r1*data_loc[loc] + r2*uniform(s_str, s_end) for loc in peak_locs]
    else:
        res["loc"]    = [r1*i*(s_end-s_str)/(n_rup+1) + r2*uniform(s_str, s_end) for i in range(1, n_rup+1)]
    res["width"]  = [r1*50 + r2*uniform(10,300) for i in range(n_rup)]
    res["origin"] = [r1*d_par_str + r2*uniform(d_par_min, d_par_max)]
    res["ramp"]   = [r1*0 + r2*uniform(ramp_lb, ramp_ub)]
    res["disp"]   = [[r1*d_fact*d_par_width + r2*uniform(d_lb, d_ub)]]
    res["slope"]  = [[0. + r2*uniform(slope_lb, slope_ub)]]

    if n_dim == 2:
        res["origin"].append(r1*d_nor_str + r2*uniform(d_nor_min, d_nor_max))
        res["ramp"].append(r1*ramp_nor_val + r2*uniform(ramp_lb, ramp_ub))
        res["disp"][0].append(r1*d_fact*d_nor_width + r2*uniform(d_lb, d_ub))
        res["slope"][0].append(0. + r2*uniform(slope_lb, slope_ub))

    for i in range(n_rup-1):
        d1,s1  = [], []
        for j in range(len(res["disp"][0])):
            d1.append(r1*res["disp"][0][j] + r2*uniform(d_lb, d_ub))
            s1.append(r1*res["slope"][0][j] + r2*uniform(slope_lb, slope_ub))
        res["disp"].append(d1)
        res["slope"].append(s1)
    
    param_bounds = {
        'origin': [c0_lb, c0_ub],
        'loc' : [s_lb, s_ub],
        'width' : [a_lb, a_ub],
        'disp' : [d_lb, d_ub],
        'ramp': [ramp_lb, ramp_ub],
        'slope': [slope_lb, slope_ub]
    }

    return res, param_bounds


def prof_normalization(data):
    #
    data_nrom = data.copy()
    
    #scale factor
    offset = data[:,0].min() 
    scale = 1/(data[:,0].max() - data[:,0].min())

    #x axis shift
    data_nrom[:,0] -= offset
    #x axis scaling
    data_nrom[:,0] *= scale

    return data_nrom, offset, scale


def param_scaling(scale, flip, param):
    scaled_param = {}
    #position arguments
    for key in param.keys():
        temp = []
        for name, val in param[key]:
            if key == "width" or key == "ramp" or key == "slope":
                temp.append((name, val*scale))
            elif key == "loc":
                temp.append((name, val/scale))
            elif key == "origin":
                temp.append((name, val))
            else:
                temp.append((name, flip*val))
        scaled_param[key] = temp
    return scaled_param


# def plot_uncert(u):
#     fig, ax = plt.subplots(figsize=(6, 4))
#     ax.hist(u.ravel(), bins=20, alpha=0.75, color='steelblue', edgecolor='black')
#     ax.set_title("Uncertainty Distribution")
#     ax.set_xlabel("Uncertainty at loc")
#     ax.set_ylabel("Frequency")
#     ax.grid(True)

#     return fig


def plot_uncert_with_lines(u, x, y, fit_y, p1_data, p99_data):
    p1, (lp1, rp1) = p1_data
    p99, (lp99, rp99) = p99_data

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].hist(u.ravel(), bins=30, alpha=0.75, color='steelblue', edgecolor='black')
    ax[0].axvline(p1, color='red', linestyle='--', label="1st percentile")
    ax[0].axvline(p99, color='green', linestyle='--', label="99th percentile")
    ax[0].set_title("Uncertainty Distribution")
    ax[0].set_xlabel("Projected Displacement at Profile Centerpoint")
    ax[0].set_ylabel("Frequency")
    ax[0].grid(True)
    ax[0].legend()

    ax[1].plot(x, lp1[0]*x + lp1[1], 'r-', label="Left (1st pct)")
    ax[1].plot(x, rp1[0]*x + rp1[1], 'r--', label="Right (1st pct)")

    ax[1].plot(x, lp99[0]*x + lp99[1], 'g-', label="Left (99th pct)")
    ax[1].plot(x, rp99[0]*x + rp99[1], 'g--', label="Right (99th pct)")

    ax[1].plot(x, y, color='blue', marker='o')
    ax[1].plot(x, fit_y, color='orange')
    
    ax[1].set_ylim(min(y)-1, max(y)+1)

    ax[1].set_title("Regression Lines at Percentiles")
    ax[1].set_xlabel("Distance Along Profile")
    ax[1].set_ylabel("Displacement")
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    return fig


def linear_regression(X, y):
    XtX = X.T @ X
    Xty = X.T @ y
    return np.linalg.pinv(XtX) @ Xty

def calc_uncertainties(seg1, seg2, loc, wind_bounds=None, n_iter=1000):
    (min_w, max_w) = (wind_bounds[0], wind_bounds[1]) if wind_bounds is not None else (20, 50)
    left_params, right_params = [], []
    #window units should be length

    for _ in range(n_iter):
        wL = np.random.randint(min_w, max_w + 1)
        startL = np.random.randint(0, len(seg1) - wL + 1)
        winL = seg1[startL:startL + wL]
        XL = np.vstack([winL[:, 0], np.ones(wL)]).T
        yL = winL[:, 1]
        slopeL, interceptL = np.linalg.pinv(XL.T @ XL) @ (XL.T @ yL)
        left_params.append((slopeL, interceptL))

        wR = np.random.randint(min_w, max_w + 1)
        startR = np.random.randint(0, len(seg2) - wR + 1)
        winR = seg2[startR:startR + wR]
        XR = np.vstack([winR[:, 0], np.ones(wR)]).T
        yR = winR[:, 1]
        slopeR, interceptR = np.linalg.pinv(XR.T @ XR) @ (XR.T @ yR)
        right_params.append((slopeR, interceptR))

    left_params = np.array(left_params)
    right_params = np.array(right_params)

    left_preds = left_params[:, 0] * loc + left_params[:, 1]
    right_preds = right_params[:, 0] * loc + right_params[:, 1]

    uncertainties = np.abs(left_preds[:, None] - right_preds[None, :])

    u_flat = uncertainties.ravel()
    p1, p99 = np.percentile(u_flat, [1, 99])

    idx1 = np.unravel_index(np.argmin(np.abs(uncertainties - p1)), uncertainties.shape)
    idx99 = np.unravel_index(np.argmin(np.abs(uncertainties - p99)), uncertainties.shape)

    pair1 = (left_params[idx1[0]], right_params[idx1[1]])
    pair99 = (left_params[idx99[0]], right_params[idx99[1]])

    return uncertainties, (p1, pair1), (p99, pair99)

def post_process_params(data_obj, model, coords):
    scale, shift = data_obj.scale_shift
    norm_vals = model(torch.tensor(data_obj.x, dtype=torch.float32).unsqueeze(0).T)

    col_names = ["Profile ID", "Dimension", "Rupture", "origin", "rescaled origin", "loc",
                 "rescaled loc", "width", "rescaled width", "actual width", "ramp", 
                 "rescaled ramp", "disp", "rescaled disp", "slope", "rescaled slope",
                 "latitude", "longitude", "old lat", "old lon"]
    df = pd.DataFrame(np.zeros((data_obj.n_rup*data_obj.n_dim, len(col_names))), columns=col_names)

    dim_col, rup_col = [], []

    for d in range(1, data_obj.n_dim+1):
        dim_col.extend(data_obj.n_rup*[d])
        rup_col.extend(range(1, data_obj.n_rup+1))

    df['Dimension'], df["Rupture"] = dim_col, rup_col
    ramp_p = None
    org_params = {'loc': [],
                  'width': [],
                  'disp': [],
                  'slope': [],
                  'origin': [],
                  'ramp': []}

    rescale = {
        'width': lambda v: v * scale,
        'loc': lambda v: v / scale + shift,
        'disp': lambda v: v * data_obj.flip,
        'ramp': lambda v: v * data_obj.flip * scale,
        'origin': lambda v: v * data_obj.flip - ramp_p.item()*shift,
        'slope': lambda v: v * data_obj.flip * scale,
    }


    for full_name, p in model.named_parameters():

        ind = full_name.rfind(".") + 1
        name = full_name[ind:]
        if name == "ramp":
            ramp_p = p

        org_params[name].append(p.item())
        
        loc_encountered = False
        first_row = None

        r_num = re.search(r'r\d{1,2}', full_name)
        d_num = re.search(r'd\d{1,2}', full_name)

        r_num = int(r_num.group()[1:]) + 1 if r_num else None
        d_num = int(d_num.group()[1:]) + 1 if d_num else None

        if d_num and r_num:
            rows = df.index[(df["Dimension"] == d_num) & (df["Rupture"] == r_num)].to_list()
        else:
            # populate first row of params that are the same across dims or rups
            rows = df.index[df["Dimension"] == d_num].to_list() if d_num else df.index[df["Rupture"] == r_num].to_list()
            row = rows[0]
            df.at[row, name] = p.item()
            df.at[row, "rescaled "+name] = rescale[name](p.item())
            p.data = rescale[name](p.data)
            first_row = row
            rows = rows[1:]
        
        for row in rows:

            df.at[row, "Profile ID"] = data_obj.prof_id
            if name == "disp" or name == "slope":
                df.at[row, name] = p.item()
                df.at[row, "rescaled "+name] = rescale[name](p.item())
                p.data = rescale[name](p.data)
                continue

            df.at[row, name] = df.at[first_row, name]
            df.at[row, "rescaled "+name] = df.at[first_row, "rescaled "+name]
        
            if name == 'loc' and not loc_encountered and coords != None:
                new_coords = convert_location(p.item(), coords)['LATLON']
                df.at[row, "latitude"] = new_coords[0]
                df.at[row, "longitude"] = new_coords[1]
                df.at[row, "old lat"] = coords[0]
                df.at[row, "old lon"] = coords[1]
                loc_encountered = True


    scaled_vals = model(torch.tensor((data_obj.x/scale)+shift, dtype=float).unsqueeze(0).T)

    lin_seg, curve_seg, act_widths = split_profile(df["rescaled loc"], df['rescaled width'], scale=1)

    df['actual width'] = act_widths

    org_params["slope"] = [[org_params["slope"][:2]], [org_params["slope"][2:]]]
    org_params["disp"] = [[org_params["disp"][:2]], [org_params["disp"][2:]]]

    return df, norm_vals.detach().numpy(), scaled_vals.detach().numpy(), lin_seg, curve_seg, org_params


def post_process_params_v4(data_obj, model, coords):
    scale, shift = data_obj.scale_shift
    norm_vals = model(torch.tensor(data_obj.x, dtype=torch.float32).unsqueeze(0).T)

    col_names = ["Profile ID", "Dimension", "Rupture", "origin", "rescaled origin", "loc",
                 "rescaled loc", "width", "rescaled width", "actual width", "ramp", 
                 "rescaled ramp", "disp", "rescaled disp", "slope", "rescaled slope",
                 "latitude", "longitude", "old lat", "old lon"]
    df = pd.DataFrame(np.zeros((data_obj.n_rup*data_obj.n_dim, len(col_names))), columns=col_names)

    dim_col, rup_col = [], []

    for d in range(1, data_obj.n_dim+1):
        dim_col.extend(data_obj.n_rup*[d])
        rup_col.extend(range(1, data_obj.n_rup+1))

    df['Dimension'], df["Rupture"] = dim_col, rup_col
    ramp_p = None
    org_params = {'loc': [],
                  'width': [],
                  'disp': [],
                  'slope': [],
                  'origin': [],
                  'ramp': []}

    rescale = {
        'width': lambda v: v * scale,
        'loc': lambda v: v / scale + shift,
        'disp': lambda v: v * data_obj.flip,
        'ramp': lambda v: v * data_obj.flip * scale,
        'origin': lambda v: v * data_obj.flip - ramp_p*shift,
        'slope': lambda v: v * data_obj.flip * scale,
    }

    model.rescaled = True
    for full_name, p in model.named_parameters():

        ind = full_name.rfind(".") + 1
        name = full_name[ind:]

        p_lb, p_ub = data_obj.param_bounds[name]
        real_norm_val = (p_lb + (p_ub - p_lb)*torch.sigmoid(p)).item()  #actual normalized param value
        real_rescaled_val = rescale[name](real_norm_val)

        if name == "ramp":
            ramp_p = real_norm_val

        org_params[name].append(real_norm_val)
        
        loc_encountered = False
        first_row = None

        r_num = re.search(r'r\d{1,2}', full_name)
        d_num = re.search(r'd\d{1,2}', full_name)

        r_num = int(r_num.group()[1:]) + 1 if r_num else None
        d_num = int(d_num.group()[1:]) + 1 if d_num else None

        if d_num and r_num:
            rows = df.index[(df["Dimension"] == d_num) & (df["Rupture"] == r_num)].to_list()
        else:
            # populate first row of params that are the same across dims or rups
            rows = df.index[df["Dimension"] == d_num].to_list() if d_num else df.index[df["Rupture"] == r_num].to_list()
            first_row = rows[0]
            df.at[first_row, name] = real_norm_val
            df.at[first_row, "rescaled "+name] = real_rescaled_val

            with torch.no_grad():
                p.copy_(torch.tensor(real_rescaled_val, dtype=torch.float64).reshape_as(p))

            rows = rows[1:]

        for row in rows:
            df.at[row, "Profile ID"] = data_obj.prof_id
            if name == "disp" or name == "slope":
                df.at[row, name] = real_norm_val
                df.at[row, "rescaled " + name] = real_rescaled_val
                with torch.no_grad():
                    p.copy_(torch.tensor(real_rescaled_val, dtype=torch.float64).reshape_as(p))
                continue

            df.at[row, name] = df.at[first_row, name]
            df.at[row, "rescaled "+name] = df.at[first_row, "rescaled "+name]
        
            if name == 'loc' and not loc_encountered and coords != None:
                new_coords = convert_location(real_norm_val, coords)['LATLON']
                df.at[row, "latitude"] = new_coords[0]
                df.at[row, "longitude"] = new_coords[1]
                df.at[row, "old lat"] = coords[0]
                df.at[row, "old lon"] = coords[1]
                loc_encountered = True


    scaled_vals = model(torch.tensor((data_obj.x/scale)+shift, dtype=torch.float32).unsqueeze(0).T)

    lin_seg, curve_seg, act_widths = split_profile(df["rescaled loc"], df['rescaled width'], scale=1)

    df['actual width'] = act_widths

    org_params["slope"] = [[org_params["slope"][:2]], [org_params["slope"][2:]]]
    org_params["disp"] = [[org_params["disp"][:2]], [org_params["disp"][2:]]]

    return df, norm_vals.detach().numpy(), scaled_vals.detach().numpy(), lin_seg, curve_seg, org_params


def sigmoid_percentile(q, s_r, a_r):
    """
    Compute the s values for which a sigmoid function reaches the given percentiles.
    
    The sigmoid function is defined as:
        sigmoid(s) = 1 / (1 + np.exp(-a_r * (s - s_r)))
    and its inverse is:
        s = s_r + (1 / a_r) * np.log(p / (1 - p))
    where p is the percentile value.
    
    Parameters:
    -----------
    q   : list or numpy array
        Percentiles at which to calculate s values. Each value must be in the open interval (0, 1).
    s_r : float
        The reference (midpoint) of the sigmoid function.
    a_r : float
        The scaling factor of the sigmoid function.
        
    Returns:
    --------
    s_values : numpy array
        The s values such that sigmoid(s) = p for each provided percentile p.
    """
    q = np.asarray(q)
    
    #ensure percentiles are between 0 and 1
    if np.any(q <= 0) or np.any(q >= 1):
        raise ValueError("All percentiles must be in the interval (0, 1).")
    
    #ensure that a_r is nonzero
    if a_r == 0:
        raise ValueError("The scaling factor a_r must be nonzero.")
    
    #compute the inverse of the sigmoid function
    s_q = s_r + np.log(q / (1 - q)) / a_r
    return s_q
    
def sigmoid_width(q_pair, s_r, a_r):
    """
    Compute the sigmoid width corresponding to two percentiles.
    
    Parameters:
    -----------
    q_pair : list or numpy array of two elements
        Two percentile values (each in the open interval (0, 1)).
    s_r : float
        The reference (midpoint) value of the sigmoid function.
    a_r : float
        The scaling factor of the sigmoid function. Must be nonzero.
        
    Returns:
    --------
    distance : float
        The absolute difference between the two s values corresponding to the input percentiles.
    """
    q_pair = np.asarray(q_pair)
    
    # Ensure exactly two percentiles are provided
    if q_pair.size != 2:
        raise ValueError("Input q_pair must contain exactly two elements.")
    
    #compute location for given percentiles
    s_values = sigmoid_percentile(q_pair, s_r, a_r)
    
    #return the absolute difference between the two s values
    return abs(s_values[1] - s_values[0])


def check_flip(data_disp):
    num_points = int(0.25*len(data_disp))
    if np.mean(data_disp[:num_points]) > np.mean(data_disp[len(data_disp)-num_points:]):
        return -1
    return 1


def plot_disp_graph(disp):
    x_axis = range(1, len(disp)+1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x_axis, disp, 'o')
    #ax.set_ylim()
    ax.set_title("Displacement Graph")
    ax.set_xlabel("Profile")
    ax.set_ylabel("Displacement")
    ax.grid(True)

    return fig


def convert_location(loc, coords, input_format='LATLON'):
    if input_format == 'LATLON':
        origin_lat, origin_long, azimuth = coords
        easting, northing, zone_num, zone_let = utm.from_latlon(origin_lat, origin_long)
    else:
        easting, northing, azimuth, zone_num, zone_let = coords

    northern_bool = zone_let.upper() in {'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'NORTHERN'}
    
    azimuth = np.deg2rad(azimuth + 90)
    #azimuth = np.deg2rad(180 - azimuth)
    #loc += (254*3)
    loc -= 6000

    x = loc*np.sin(azimuth)
    y = loc*np.cos(azimuth)

    loc_easting = easting + x
    loc_northing = northing + y

    loc_lat, loc_long = utm.to_latlon(loc_easting, loc_northing, zone_num, northern=northern_bool)

    # return {"UTM": [loc_easting, loc_northing, zone_num, zone_let], "LATLON": [loc_lat, loc_long]}
    return {"UTM": {"easting": loc_easting, "northing": loc_northing, "zone": f"{zone_num}{zone_let}"},
            "LATLON": {"latitude": loc_lat, "longitude": loc_long}}


def split_profile(locs, width_factors, scale=1):
    widths = len(width_factors)*[0]
    for w in range(len(width_factors)):
        widths[w] = sigmoid_width([0.02, 0.98], locs[w], width_factors[w])


    curr_ind = 0
    linear_segments = []
    sigmoid_segments = []
    for i in range(len(locs)):
        linear_segments.append((curr_ind,round((locs[i]-widths[i]/2)/scale)))
        sigmoid_segments.append((round((locs[i]-widths[i]/2)/scale),round((locs[i]+widths[i]/2)/scale)))
        curr_ind = round((locs[i]+widths[i]/2)/scale)
    linear_segments.append((curr_ind, -1))

    return linear_segments, sigmoid_segments, widths

import torch
from torch import nn
from torch import Tensor
from torch.nn import functional
from fit_animation import FitHistoryWidget
from scipy.stats import linregress
from helper_funs import *


# Slip Profile Layers
# ---   ---   ---   ---   ---
class LayerBase1Dim(nn.Module):
    def __init__(self, seed_origin, seed_ramp) -> None:
        super().__init__()

        self.origin, self.origin_bounds = seed_origin
        self.ramp, self.ramp_bounds = seed_ramp

        transf_orig = (self.origin - self.origin_bounds[0]) / (self.origin_bounds[1] - self.origin_bounds[0])
        transf_orig = torch.clamp(torch.tensor(transf_orig, dtype=torch.float32), 1e-8, 1 - 1e-8)
        transf_orig = torch.log(transf_orig / (1-transf_orig))

        transf_ramp = (self.ramp - self.ramp_bounds[0]) / (self.ramp_bounds[1] - self.ramp_bounds[0])
        transf_ramp = torch.clamp(torch.tensor(transf_ramp, dtype=torch.float32), 1e-8, 1 - 1e-8)
        transf_ramp = torch.log(transf_ramp / (1-transf_ramp))

        #optimizable parameters
        self.ramp   = nn.Parameter(torch.tensor([[transf_ramp]], dtype=torch.float32), requires_grad=True) #slope of base profile
        self.origin = nn.Parameter(torch.tensor([transf_orig], dtype=torch.float32), requires_grad=True) #displacment at origin

    def forward(self, s:Tensor, rescaled: bool) -> Tensor:
        '''Compute displacement of base profile (origin and slope)'''
        if rescaled:
            ramp = self.ramp
            origin = self.origin
        else:
            ramp = self.ramp_bounds[0] + (self.ramp_bounds[1] - self.ramp_bounds[0])*torch.sigmoid(self.ramp)
            origin = self.origin_bounds[0] + (self.origin_bounds[1] - self.origin_bounds[0])*torch.sigmoid(self.origin)

        #transform profile axis
        d = functional.linear(s, ramp, origin)
        
        return d

class LayerSingleRup1Dim(nn.Module):
    '''Single Rupture Layer (1 Dimensional)'''
    def __init__(self, seed_disp, seed_slope) -> None:
        super().__init__()

        self.disp, self.disp_bounds = seed_disp
        self.slope, self.slope_bounds = seed_slope
        #self.width = seed_width

        transf_disp = (self.disp - self.disp_bounds[0]) / (self.disp_bounds[1] - self.disp_bounds[0])
        transf_disp = torch.clamp(torch.tensor(transf_disp, dtype=torch.float32), 1e-8, 1 - 1e-8)
        transf_disp = torch.log(transf_disp / (1-transf_disp))

        transf_slope = (self.slope - self.slope_bounds[0]) / (self.slope_bounds[1] - self.slope_bounds[0])
        transf_slope = torch.clamp(torch.tensor(transf_slope, dtype=torch.float32), 1e-8, 1 - 1e-8)
        transf_slope = torch.log(transf_slope / (1-transf_slope))

        #optimizable parameters
        self.disp  = nn.Parameter(torch.tensor([transf_disp], dtype=torch.float32), requires_grad=True)  #displacement
        self.slope = nn.Parameter(torch.tensor([transf_slope], dtype=torch.float32), requires_grad=True) #slope
        
    #def forward(self, s_sig:Tensor, s_hinge: Tensor) -> Tensor:
    def forward(self, s: Tensor, rescaled: bool) -> Tensor:
        '''Compute displacement from single rupture 
           (zero displacement at origin, rupture location at s=0)'''

        if rescaled:
            disp = self.disp
            slope = self.slope
        else:
            disp = self.disp_bounds[0] + (self.disp_bounds[1] - self.disp_bounds[0])*torch.sigmoid(self.disp)
            slope = self.slope_bounds[0] + (self.slope_bounds[1] - self.slope_bounds[0])*torch.sigmoid(self.slope)

        d = disp * functional.sigmoid(s) + slope * functional.softplus(s)   #no scale
        #d = disp * functional.sigmoid(s_sig) + slope * functional.softplus(s_hinge) # scale with width as par

        return d

class LayerSingleRupMDim(nn.Module):
    def __init__(self, ndim:int, 
                 seed_loc, seed_width,
                 seed_disp, seed_slope) -> None:
        super().__init__()
        
        #initialize seed if unspecified
        if seed_disp is None:  seed_disp  = [1.0] * ndim
        if seed_slope is None: seed_slope = [0.0] * ndim

        seed_disp, disp_bounds = seed_disp
        seed_slope, slope_bounds = seed_slope

        #fixed parameters
        self.ndim = ndim
        #optimizable parameters
        self.loc, self.loc_bounds = seed_loc
        self.width, self.width_bounds = seed_width

        transf_loc = (self.loc - self.loc_bounds[0]) / (self.loc_bounds[1] - self.loc_bounds[0])
        transf_loc = torch.clamp(torch.tensor(transf_loc, dtype=torch.float32), 1e-8, 1 - 1e-8)
        transf_loc = torch.log(transf_loc / (1-transf_loc))

        transf_width = (self.width - self.width_bounds[0]) / (self.width_bounds[1] - self.width_bounds[0])
        transf_width = torch.clamp(torch.tensor(transf_width, dtype=torch.float32), 1e-8, 1 - 1e-8)
        transf_width = torch.log(transf_width / (1-transf_width))

        self.loc   = nn.Parameter(torch.tensor([transf_loc], dtype=torch.float32), requires_grad=True)     #rupture location
        self.width = nn.Parameter(torch.tensor([[transf_width]], dtype=torch.float32), requires_grad=True) #rupture width

        #building block layers
        self.prof = nn.ModuleDict([[self.key_dim(j), LayerSingleRup1Dim((seed_disp[j], disp_bounds), (seed_slope[j], slope_bounds))] 
                                   for j in range(self.ndim)])
    
    def key_dim(self, j:int) -> str:
        
        return 'd%i'%j
    
    def forward(self, s:Tensor, rescaled: bool) -> Tensor:
        '''Compute displacement from single rupture, multiple dimensions
           (zero displacement at origin, rupture location from linear layer)'''
        
        if rescaled:
            loc = self.loc
            width = self.width
        else:
            loc = self.loc_bounds[0] + (self.loc_bounds[1] - self.loc_bounds[0])*torch.sigmoid(self.loc)
            width = self.width_bounds[0] + (self.width_bounds[1] - self.width_bounds[0])*torch.sigmoid(self.width)
        
        #transform profile axis
        s = functional.linear(s, width, -loc*width)
        # s_shift = s - loc
        # s_sig = width*s_shift
        # s_hinge = s_shift
        
        #compute displacement multiple dimenstions
        #d = torch.cat([self.prof[self.key_dim(j)](s_sig, s_hinge) for j in range(self.ndim)], dim=1)
        d = torch.cat([self.prof[self.key_dim(j)](s, rescaled) for j in range(self.ndim)], dim=1)

        return d

# Slip Profile Neural Network
# ---   ---   ---   ---   ---
class SlipProfileNN(nn.Module):
    def __init__(self, ndim:int=1, nrup:int=1,
                 seed_origin = None, seed_ramp = None, 
                 seed_loc = None, seed_width = None,
                 seed_disp = None, seed_slope = None,
                 bounds = None,
                 rescaled=False) -> None:
        super().__init__()

        self.bounds = bounds
        #initialize seed for base profile parameters if unspecified
        if seed_origin is None: seed_origin = [0.0] * ndim
        if seed_ramp is None: seed_ramp  = [0.0] * ndim
        #initialize seed for slip profile parameters if unspecified
        if seed_loc   is None: seed_loc   = [(l+0.5)/nrup for l in range(nrup)]
        if seed_width is None: seed_width = [100.] * nrup
        if seed_disp  is None: seed_disp  = [None] * nrup
        if seed_slope is None: seed_slope = [None] * nrup

        #fixed parameters
        self.nrup = nrup #number of ruptures
        self.ndim = ndim #number of dimensions

        self.rescaled = rescaled

        #building block layers
        self.prof = nn.ModuleDict([[self.key_rup(l), LayerSingleRupMDim(ndim, (seed_loc[l],bounds['loc']), (seed_width[l], bounds['width']), 
                                                                              (seed_disp[l], bounds['disp']), (seed_slope[l], bounds['slope']))] 
                                   for l in range(self.nrup)])
        self.base = nn.ModuleDict([[self.key_dim(j), LayerBase1Dim((seed_origin[j], bounds['origin']), (seed_ramp[j], bounds['ramp']))] 
                                   for j in range(self.ndim)])
    
    def key_rup(self, l:int) -> str:
        
        return 'r%i'%l

    def key_dim(self, j:int) -> str:    

        #inherit key_dim method from LayerSingleRupMDim
        return self.prof[self.key_rup(0)].key_dim(j)
    
    def forward(self, s:Tensor) -> Tensor:
        '''Compute displacement from multiple ruptres'''
        #base profile (origin and linear slope)
        d = torch.cat([self.base[self.key_dim(j)](s, self.rescaled) for j in range(self.ndim)], dim=1)

        #add displacement of each rupture
        for l in range(self.nrup):
            d += self.prof[self.key_rup(l)](s, self.rescaled)
        
        return d
    

def set_trainable(model_params, vars):
    if vars[0] == "all_true":
        for _, param in model_params:
            param.requires_grad = True
    else:
        for full_name, param in model_params:
            param.requires_grad = any(var_name in full_name for var_name in vars)
        

def rmse(y_pred, y_act):
    #return torch.sqrt(torch.mean((y_pred - y_act)**2)).detach().numpy()
    return torch.sqrt(torch.mean((y_pred - y_act)**2))

def L1(y_pred, y_act):
    #return torch.mean(torch.abs(y_pred - y_act)).detach().numpy()
    return torch.mean(torch.abs(y_pred - y_act))


# def loc_loss(y_pred, y_act, loc_params):
#     if len(loc_params) == 1:
#         return torch.nn.MSELoss()(y_pred, y_act)

#     loc_penalty = torch.tensor(1, dtype=torch.float32)
#     for i in range(1, len(loc_params)):
#         loc_penalty = torch.add(loc_penalty,
#                                 (loc_params[i] - loc_params[i-1]) - torch.tensor(0.1, dtype=torch.float32),
#                                 alpha=3)
    
#     return torch.subtract(torch.nn.MSELoss()(y_pred, y_act), loc_penalty)

def get_model_split_params(model):
    """
    Return current model rupture locs and widths in their original,
    bounded parameter space as ordinary Python lists.

    The returned values are detached from autograd because they are
    used only to define loss weights.
    """
    locs = []
    widths = []

    for i in range(model.nrup):
        rup = model.prof[model.key_rup(i)]

        loc_raw = rup.loc.detach().item()
        width_raw = rup.width.detach().item()

        loc = param_transf_to_act(
            loc_raw,
            rup.loc_bounds,
        )

        width = param_transf_to_act(
            width_raw,
            rup.width_bounds,
        )

        locs.append(float(loc))
        widths.append(float(width))

    return locs, widths

def weighted_line_fit(x, y, weights):
    w_sum = np.sum(weights)

    if w_sum <= 0:
        return 0.0, float(np.mean(y))

    x_mean = np.sum(weights * x) / w_sum
    y_mean = np.sum(weights * y) / w_sum

    x_centered = x - x_mean
    y_centered = y - y_mean

    denom = np.sum(
        weights * x_centered**2
    )

    if denom < 1e-12:
        return 0.0, y_mean

    slope = (
        np.sum(
            weights
            * x_centered
            * y_centered
        )
        / denom
    )

    intercept = (
        y_mean
        - slope * x_mean
    )

    return slope, intercept


def robust_linear_fit(
    y,
    start_idx,
    end_idx,
    c=2.5,
    min_weight=0.01,
    n_iter=4,
):
    """
    Robustly fit one linear segment.

    Returns
    -------
    full_weights : np.ndarray
        Length len(y). Robust weights are present inside the requested
        segment; values outside the segment are 1.

    slope : float | None
        Robust slope of the segment.

    Notes
    -----
    Uses fast closed-form weighted linear regression rather than
    np.linalg.lstsq().
    """

    y = np.asarray(y, dtype=float)
    n = len(y)

    full_weights = np.ones(
        n,
        dtype=np.float32,
    )

    start_idx = max(
        0,
        int(start_idx),
    )

    end_idx = min(
        n - 1,
        int(end_idx),
    )

    if end_idx - start_idx < 4:
        return full_weights, None

    y_seg = y[
        start_idx:end_idx + 1
    ]

    x_seg = np.arange(
        start_idx,
        end_idx + 1,
        dtype=float,
    )

    finite = np.isfinite(y_seg)

    if finite.sum() < 5:
        return full_weights, None

    x_fit = x_seg[finite]
    y_fit = y_seg[finite]

    fit_weights = np.ones(
        len(y_fit),
        dtype=float,
    )

    slope = 0.0

    for _ in range(n_iter):

        # Fast weighted straight-line fit.
        slope, intercept = weighted_line_fit(
            x_fit,
            y_fit,
            fit_weights,
        )

        fitted = (
            slope * x_fit
            + intercept
        )

        residuals = (
            y_fit - fitted
        )

        # Robustly center residuals.
        residual_center = np.median(
            residuals
        )

        centered = (
            residuals
            - residual_center
        )

        # Robust residual scale.
        mad = np.median(
            np.abs(centered)
        )

        sigma = (
            1.4826 * mad
        )

        if (
            not np.isfinite(sigma)
            or sigma < 1e-8
        ):
            break

        # Tukey bisquare weighting.
        u = centered / (
            c * sigma
        )

        new_weights = np.zeros_like(
            fit_weights
        )

        inside = (
            np.abs(u) < 1.0
        )

        new_weights[inside] = (
            1.0
            - u[inside]**2
        )**2

        new_weights = np.clip(
            new_weights,
            min_weight,
            1.0,
        )

        # Stop early if robust weights have converged.
        if np.max(
            np.abs(
                new_weights
                - fit_weights
            )
        ) < 1e-3:
            fit_weights = new_weights
            break

        fit_weights = new_weights

    # One final line fit using the FINAL robust weights.
    #
    # This is important because otherwise slope corresponds to the
    # weights from the previous IRLS iteration.
    slope, _ = weighted_line_fit(
        x_fit,
        y_fit,
        fit_weights,
    )

    segment_weights = np.full(
        len(y_seg),
        min_weight,
        dtype=np.float32,
    )

    segment_weights[finite] = (
        fit_weights.astype(
            np.float32
        )
    )

    full_weights[
        start_idx:end_idx + 1
    ] = segment_weights

    return full_weights, float(slope)


def outer_linear_weights(
    n_points,
    model,
    y_true,
    n_dim=1,
    scale=1,
    edge_weight=0.5,
    transition_weight=2.0,
    middle_weight=1.0,
    power=1.5,
    robust_c=2.5,
    robust_min_weight=0.01,
):
    """
    Build all information associated with the first and last
    linear profile segments.

    Returns
    -------
    weights : torch.Tensor
        Pointwise loss weights, flattened in the same order as y_true.

    slope_info : dict
        Segment boundaries and robust target slopes.
    """

    # ==========================================================
    # FIND FIRST/LAST LINEAR SEGMENTS ONCE
    # ==========================================================

    locs, width_factors = (
        get_model_split_params(model)
    )

    linear_segments, _, _ = split_profile(
        locs,
        width_factors,
        scale=scale,
    )

    first_start, first_end = linear_segments[0]
    last_start, last_end = linear_segments[-1]

    first_start = max(0, int(first_start))
    first_end = min(n_points - 1, int(first_end))
    last_start = max(0, int(last_start))

    if last_end == -1:
        last_end = n_points - 1
    else:
        last_end = min(n_points - 1, int(last_end))

    device = y_true.device
    dtype = y_true.dtype

    y_np = y_true.detach().reshape(-1).cpu().numpy()

    dimension_weights = []
    target_slopes = []

    # ==========================================================
    # ONE PASS PER DIMENSION
    # ==========================================================

    for dim_idx in range(n_dim):

        begin = dim_idx * n_points
        end = begin + n_points

        y_dim = y_np[begin:end]

        # ======================================================
        # POSITIONAL WEIGHTS
        # ======================================================

        weights = torch.full(
            (n_points,),
            float(middle_weight),
            device=device,
            dtype=dtype,
        )

        # ------------------------------------------------------
        # First linear segment
        #
        # profile edge -> sigmoid transition
        # low weight   -> high weight
        # ------------------------------------------------------

        if first_end >= first_start:

            n_first = first_end - first_start + 1

            t = torch.linspace(
                0.0,
                1.0,
                n_first,
                device=device,
                dtype=dtype,
            )

            t = t ** power

            weights[first_start:first_end + 1] = (
                edge_weight + (transition_weight - edge_weight) * t
            )

        # ------------------------------------------------------
        # Last linear segment
        #
        # sigmoid transition -> profile edge
        # high weight        -> low weight
        # ------------------------------------------------------

        if last_end >= last_start:

            n_last = last_end - last_start + 1

            t = torch.linspace(
                1.0,
                0.0,
                n_last,
                device=device,
                dtype=dtype,
            )

            t = t ** power

            weights[last_start:last_end + 1] = (
                edge_weight + (transition_weight - edge_weight) * t
            )

        # ======================================================
        # ROBUST FITS
        #
        # Each segment is fitted exactly ONCE.
        # Each call provides:
        #     robust point weights
        #     robust target slope
        # ======================================================

        robust_weights = np.ones(
            n_points,
            dtype=np.float32,
        )

        first_target_slope = None
        last_target_slope = None

        if first_end > first_start:

            (
                first_robust,
                first_target_slope,
            ) = robust_linear_fit(
                y=y_dim,
                start_idx=first_start,
                end_idx=first_end,
                c=robust_c,
                min_weight=robust_min_weight,
            )

            robust_weights[
                first_start:first_end + 1
            ] = first_robust[
                first_start:first_end + 1
            ]

        if last_end > last_start:

            (
                last_robust,
                last_target_slope,
            ) = robust_linear_fit(
                y=y_dim,
                start_idx=last_start,
                end_idx=last_end,
                c=robust_c,
                min_weight=robust_min_weight,
            )

            robust_weights[
                last_start:last_end + 1
            ] = last_robust[
                last_start:last_end + 1
            ]

        robust_weights = torch.as_tensor(
            robust_weights,
            device=device,
            dtype=dtype,
        )

        # Positional importance × robust outlier handling.
        weights = (
            weights
            * robust_weights
        )

        dimension_weights.append(
            weights
        )

        target_slopes.append(
            (
                first_target_slope,
                last_target_slope,
            )
        )

    return (
        torch.cat(dimension_weights),
        {
            "first_segment": (
                first_start,
                first_end,
            ),
            "last_segment": (
                last_start,
                last_end,
            ),
            "target_slopes": target_slopes,
        },
    )


def torch_segment_slope(y_segment):
    y_segment = y_segment.reshape(-1)

    n = y_segment.numel()

    if n < 2:
        return None

    x = torch.arange(
        n,
        device=y_segment.device,
        dtype=y_segment.dtype,
    )

    x = x - x.mean()

    y = (
        y_segment
        - y_segment.mean()
    )

    denominator = torch.sum(
        x ** 2
    )

    return (
        torch.sum(x * y)
        / denominator
    )


def outer_linear_slope_penalty(
    y_pred,
    slope_info,
    n_points,
    n_dim,
    penalty_weight=0.1,
):
    """
    Penalize disagreement between model and robust outer-linear
    trends.

    Rather than comparing raw slopes, compare the displacement
    change implied by those slopes across each segment.

    This puts the penalty on approximately the same numerical
    scale as the displacement data.
    """

    first_start, first_end = (
        slope_info["first_segment"]
    )

    last_start, last_end = (
        slope_info["last_segment"]
    )

    targets = (
        slope_info["target_slopes"]
    )

    penalty = torch.zeros(
        (),
        device=y_pred.device,
        dtype=y_pred.dtype,
    )

    n_valid = 0

    for dim_idx in range(n_dim):

        begin = dim_idx * n_points
        end = begin + n_points

        y_dim = y_pred[
            begin:end
        ]

        first_target, last_target = (
            targets[dim_idx]
        )

        # ======================================================
        # FIRST LINEAR SEGMENT
        # ======================================================

        if (
            first_target is not None
            and first_end > first_start
        ):

            model_slope = torch_segment_slope(
                y_dim[
                    first_start:
                    first_end + 1
                ]
            )

            segment_length = (
                first_end
                - first_start
            )

            model_change = (
                model_slope
                * segment_length
            )

            target_change = (
                first_target
                * segment_length
            )

            penalty = (
                penalty
                + (
                    model_change
                    - target_change
                ) ** 2
            )

            n_valid += 1

        # ======================================================
        # LAST LINEAR SEGMENT
        # ======================================================

        if (
            last_target is not None
            and last_end > last_start
        ):

            model_slope = torch_segment_slope(
                y_dim[
                    last_start:
                    last_end + 1
                ]
            )

            segment_length = (
                last_end
                - last_start
            )

            model_change = (
                model_slope
                * segment_length
            )

            target_change = (
                last_target
                * segment_length
            )

            penalty = (
                penalty
                + (
                    model_change
                    - target_change
                ) ** 2
            )

            n_valid += 1

    if n_valid > 0:
        penalty = (
            penalty / n_valid
        )

    return (
        penalty_weight
        * penalty
    )


def weighted_mse_loss(y_pred, y_true, weights):
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)

    error = (y_pred - y_true) ** 2

    return torch.sum(weights * error) / torch.sum(weights)

def weighted_l1_loss(y_pred, y_true, weights):
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)

    error = torch.abs(y_pred - y_true)

    return torch.sum(weights * error) / torch.sum(weights)

def param_act_to_transf(param, bounds):
    transf_p = (param - bounds[0]) / (bounds[1] - bounds[0])
    transf_p = np.clip(transf_p, 1e-8, 1-1e-8)
    return np.log(transf_p/(1-transf_p))
    # return torch.log(torch.tensor(transf_p / (1-transf_p)), dtype=torch.float32)

def param_transf_to_act(param, bounds):
    return bounds[0] + (bounds[1] - bounds[0])/(1+np.exp(-param))

def NN_optimize(data, collect_param_vals=False):
    device = torch.device(data.device)

    model = SlipProfileNN(ndim=data.n_dim, nrup=data.n_rup,
                          seed_origin=data.param_0['origin'],
                          seed_ramp=data.param_0['ramp'],
                          seed_loc=data.param_0['loc'],
                          seed_width=data.param_0['width'],
                          seed_disp=data.param_0['disp'],
                          seed_slope=data.param_0['slope'],
                          bounds=data.param_bounds).to(device)
    
    named_params = list(model.named_parameters())

    x, y = data.x, data.y
    scale = data.scale_shift[0]
    learn_rate, n_epoch = data.lr, data.n_epochs
    x_tensor = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0).T
    y_b = torch.tensor(y, dtype=torch.float32, device=device)
    if data.n_dim > 1:
        y_b = torch.concat((y_b[:,0], y_b[:,1]))
    
    width_p = []
    loc_p = []
    non_width_p = []
    losses = {'total_loss':[], 'states': []}

    for name, param in named_params:
        if collect_param_vals:
            losses[name] = []
        if name[-5:] == "width":
            width_p.append(param)
        else:
            if name[-3:] == "loc":
                loc_p.append(param)
            non_width_p.append(param)

    opt = torch.optim.Adam(model.parameters())            
        
    def training_loop(ratio_limit, iter_limit, vars, lr=1e-3):
        opt.param_groups[0]['lr'] = lr
        set_trainable(model.named_parameters(), vars)

        opt.zero_grad()
        ratio = 0.0
        iter_n = 0

        weights = None
        slope_info = None
        prev_loss = None

        while True:
            iter_n += 1
            if collect_param_vals:
                losses['states'].append(model.state_dict())
            
            y_pred = model(x_tensor)

            if data.n_dim > 1:
                y_pred = torch.concat((y_pred[:,0], y_pred[:,1]))

            if (iter_n-1) % 10 == 0 or weights is None:
                weights, slope_info = outer_linear_weights(n_points=x_tensor.shape[0],
                                                            model=model,
                                                            y_true=y_b,
                                                            n_dim=data.n_dim,
                                                            scale=scale,   
                                                            edge_weight=0.5,
                                                            transition_weight=2.0,
                                                            middle_weight=1.0,
                                                            power=1.5,   
                                                            robust_c=2.5,
                                                            robust_min_weight=0.01,
                                                            )
                    
            loss = loss_fn(y_pred, y_b, weights)
            loss += outer_linear_slope_penalty(
                                        y_pred=y_pred,
                                        slope_info=slope_info,
                                        n_points=x_tensor.shape[0],
                                        n_dim=data.n_dim,
                                        penalty_weight=0.1,
                                    )

            if prev_loss is None:
                prev_loss = loss.item()
            
            opt.zero_grad()
            loss.backward()
            opt.step()

            if iter_n%100 == 0:
                ratio = loss.item()/prev_loss
                prev_loss = loss.item()

            losses['total_loss'].append(loss.item())

            #if n_epoch is not None and total_n_epochs >= n_epoch:
                # break
            if ratio >= ratio_limit or iter_n >= iter_limit:
                # print("RATIO", ratio)
                # print("ITER_N", iter_n)
                break
        

    loss_fn = weighted_mse_loss
    training_loop(0.99, 20000, ["all_true"])

    loss_fn = weighted_l1_loss
    training_loop(0.999, 4000, ["loc", "width"], lr=1e-4)

    loss_fn = weighted_mse_loss
    training_loop(0.9999, 3000, ["ramp", "slope", "origin", "disp"], lr=1e-4)

    # ==============================================================
    # PLOT FINAL ROBUST LINEAR FIT SLOPES
    # ==============================================================

    # # Recalculate once using the FINAL model loc/width values
    # weights, slope_info = outer_linear_weights(
    #     n_points=x_tensor.shape[0],
    #     model=model,
    #     y_true=y_b,
    #     n_dim=data.n_dim,
    #     scale=scale,
    #     edge_weight=0.5,
    #     transition_weight=2.0,
    #     middle_weight=1.0,
    #     power=1.5,
    #     robust_c=2.5,
    #     robust_min_weight=0.01,
    # )

    # first_start, first_end = slope_info["first_segment"]
    # last_start, last_end = slope_info["last_segment"]

    # target_slopes = slope_info["target_slopes"]

    # y_np = y_b.detach().cpu().numpy()
    # x_np = np.arange(x_tensor.shape[0])

    # plt.figure()

    # for dim_idx in range(data.n_dim):

    #     begin = dim_idx * x_tensor.shape[0]
    #     end = begin + x_tensor.shape[0]

    #     y_dim = y_np[begin:end]

    #     first_slope, last_slope = target_slopes[dim_idx]

    #     # Plot original data
    #     plt.scatter(
    #         x_np,
    #         y_dim,
    #         s=5,
    #         label=f"Data dim {dim_idx}",
    #     )

    #     # ----------------------------------------------------------
    #     # FIRST ROBUST SLOPE
    #     # ----------------------------------------------------------

    #     if first_slope is not None:

    #         x_first = x_np[
    #             first_start:first_end + 1
    #         ]

    #         y_first = y_dim[
    #             first_start:first_end + 1
    #         ]

    #         # We already know the slope. Find an intercept only for
    #         # visualization by centering the line on the segment.
    #         x_mean = np.mean(x_first)
    #         y_mean = np.median(y_first)

    #         first_line = (
    #             y_mean
    #             + first_slope
    #             * (x_first - x_mean)
    #         )

    #         plt.plot(
    #             x_first,
    #             first_line,
    #             linewidth=2,
    #             label=(
    #                 f"First robust slope "
    #                 f"dim {dim_idx}: "
    #                 f"{first_slope:.6f}"
    #             ),
    #         )

    #     # ----------------------------------------------------------
    #     # LAST ROBUST SLOPE
    #     # ----------------------------------------------------------

    #     if last_slope is not None:

    #         x_last = x_np[
    #             last_start:last_end + 1
    #         ]

    #         y_last = y_dim[
    #             last_start:last_end + 1
    #         ]

    #         x_mean = np.mean(x_last)
    #         y_mean = np.median(y_last)

    #         last_line = (
    #             y_mean
    #             + last_slope
    #             * (x_last - x_mean)
    #         )

    #         plt.plot(
    #             x_last,
    #             last_line,
    #             linewidth=2,
    #             label=(
    #                 f"Last robust slope "
    #                 f"dim {dim_idx}: "
    #                 f"{last_slope:.6f}"
    #             ),
    #         )

    # plt.legend()
    # plt.show()
    
    # for name, p in model.named_parameters():
    #     var_name = name[name.rfind(".")+1:]
    #     low, high = data.param_bounds[var_name]
    #     sig = torch.sigmoid(p).item()
    #     bounded = low + (high - low) * torch.sigmoid(p)

    #     print("NAME:", name)
    #     print("VAR :", var_name)
    #     print("RAW :", p.item())
    #     print("BND :", low, high)
    #     print("SIG :", sig)
    #     print("OUT :", bounded.item())

    # print("OPTIMIZER LOSS: ", losses['total_loss'][-1])

    # print("Done! Profile "+str(data.prof_id))

    return model, losses

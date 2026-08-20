from helper_funs import (prof_normalization, gen_params_and_bounds2,
                         post_process_params_v4, calc_uncertainties,
                         plot_uncert_with_lines)

from slip_profiles_torch_v4_noclamp import NN_optimize, SlipProfileNN
import numpy as np
import pandas as pd
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
    y_par = y[:,0]
    num_par_points = int(0.25*len(y_par))
    
    # if np.mean(y[:num_points]) > np.mean(y[len(y)-num_points:]):
    #     return -1
    # return 1
    par_flip, perp_flip = 1, 1

    if np.mean(y_par[:num_par_points]) > np.mean(y_par[len(y_par)-num_par_points:]):
        par_flip = -1

    if y.shape[1] == 2:
        y_perp = y[:,1]
        num_perp_points = int(0.25*len(y_perp))

        if np.mean(y_perp[:num_perp_points]) > np.mean(y_perp[len(y_perp)-num_perp_points:]):
            perp_flip = -1

    return par_flip, perp_flip

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
    def __init__(
        self,
        data,
        prof_id,
        init_p,
        device,
        rand=False,
        learn_rate=1e-5,
        n_epochs=None,
        forced_n_rup=None,
        forced_rup_locs=None,
    ):
        self.data = data
        self.prof_id = prof_id
        self.init_p = init_p
        self.x, self.y, self.scale_shift = _normalize(self.data)

        self.par_flip, self.perp_flip = _flip(self.y)
        self.y[:, 0] *= self.par_flip
        if self.y.shape[1] == 2:
            self.y[:, 1] *= self.perp_flip

        self.n_dim = self.y.shape[1]

        if forced_n_rup is None:
            detected_n_rup, detected_rup_locs = _rup(self.y[:, 0])
            self.detected_n_rup = detected_n_rup
            self.detected_rup_locs = detected_rup_locs

            self.n_rup = detected_n_rup
            self.rup_locs = detected_rup_locs
        else:
            self.n_rup = int(forced_n_rup)
            self.rup_locs = [] #if forced_rup_locs is None else list(forced_rup_locs)

        # Forced model-selection refits deliberately do not reuse init_p from
        # another model structure because the rupture count may be different.
        self.param_0, self.param_bounds = (
            init_p
            if init_p is not None
            else gen_params_and_bounds2(
                self.x,
                self.y,
                self.n_rup,
                self.rup_locs,
                rand=rand,
            )
        )

        self.lr = learn_rate
        self.n_epochs = n_epochs
        self.device = device

def fit_data(input):
    data_obj, history = input
    model, losses, _ = NN_optimize(data_obj, history)

    return model, losses


def _reduce_rup_locs(rup_locs, n_rup, n_points):
    """
    Reduce detected rupture locations to n_rup seed locations.

    This only chooses starting locations for a lower-complexity refit.
    The optimizer can still move all locations.
    """
    n_rup = int(n_rup)
    locs = np.asarray(rup_locs, dtype=float)

    if n_rup <= 0:
        return []

    if len(locs) == 0:
        return np.linspace(
            0.2 * max(n_points - 1, 1),
            0.8 * max(n_points - 1, 1),
            n_rup,
        ).tolist()

    if len(locs) <= n_rup:
        return np.sort(locs).tolist()

    # Small 1-D k-means so clustered detections do not simply occupy all
    # lower-complexity seed slots.
    centers = np.quantile(
        locs,
        np.linspace(0.0, 1.0, n_rup + 2)[1:-1],
    )

    for _ in range(10):
        labels = np.argmin(
            np.abs(locs[:, None] - centers[None, :]),
            axis=1,
        )
        new_centers = centers.copy()

        for i in range(n_rup):
            members = locs[labels == i]
            if len(members):
                new_centers[i] = members.mean()

        if np.allclose(new_centers, centers):
            break

        centers = new_centers

    return np.sort(np.rint(centers).astype(int)).tolist()


def _fit_model(data_obj, history=False, rand=False):
    """
    Fit one Data object using the original standard/random-start behavior.
    """
    if not rand:
        return fit_data([data_obj, history])

    min_loss = np.inf
    best = None
    inputs = [[deepcopy(data_obj), history] for _ in range(10)]

    pool = Pool()
    try:
        candidates = pool.map(fit_data, inputs)
    finally:
        pool.close()
        pool.join()

    for candidate in candidates:
        final_loss = candidate[1]["total_loss"][-1]
        if final_loss < min_loss:
            best = candidate
            min_loss = final_loss

    return best


def _predict_at_x(model, x, device):
    """
    Predict at normalized x coordinates and return (n_points, n_dim) numpy data.
    """
    x_tensor = torch.tensor(
        np.asarray(x),
        dtype=torch.float32,
        device=torch.device(device),
    ).reshape(-1, 1)

    with torch.no_grad():
        pred = model(x_tensor)

    return pred.detach().cpu().numpy()


def _prepare_original_eval_data(orig_data, reference_data_obj):
    """
    Put the original, unsmoothed observations into the same normalized/flipped
    coordinate system as the fitted model.

    Using the original observations here is important:
      * BIC is based on ordinary residuals to observed data, not the weighted
        training objective.
      * the dimension screen estimates measurement noise before smoothing
        suppresses it.
    """
    data_norm_orig, _, _ = prof_normalization(orig_data)

    x_eval = np.asarray(
        data_norm_orig[:, 0],
        dtype=float,
    )

    y_eval = np.asarray(
        data_norm_orig[:, 1:],
        dtype=float,
    ).copy()

    if y_eval.ndim == 1:
        y_eval = y_eval[:, None]

    y_eval[:, 0] *= reference_data_obj.par_flip

    if y_eval.shape[1] > 1:
        y_eval[:, 1] *= reference_data_obj.perp_flip

    return x_eval, y_eval, data_norm_orig


def _estimate_noise(y):
    """
    Robust single-observation noise estimate from first differences.

    MAD / 0.67449 estimates the standard deviation of the differences;
    division by sqrt(2) converts difference noise back to one-sample noise.
    """
    y = np.asarray(y, dtype=float).reshape(-1)

    if len(y) < 3:
        return np.nan

    dy = np.diff(y)
    dy = dy[np.isfinite(dy)]

    if len(dy) < 2:
        return np.nan

    center = np.median(dy)
    mad = np.median(
        np.abs(dy - center)
    )

    return float(
        mad
        / 0.6744897501960817
        / np.sqrt(2.0)
    )


def _residual_score_summary(y_true, y_pred):
    """
    Compute RMSE, noise estimate, and RMSE/noise score for each dimension.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]

    summaries = []

    for dim in range(y_true.shape[1]):
        residual = y_true[:, dim] - y_pred[:, dim]
        finite = np.isfinite(residual)
        residual = residual[finite]

        rmse = (
            np.sqrt(np.mean(residual ** 2))
            if len(residual)
            else np.nan
        )

        noise = _estimate_noise(
            y_true[:, dim]
        )

        if (
            np.isfinite(noise)
            and noise > 1e-12
            and np.isfinite(rmse)
        ):
            score = rmse / noise
        elif np.isfinite(rmse) and rmse <= 1e-12:
            score = 0.0
        else:
            score = np.inf

        summaries.append(
            {
                "rmse": float(rmse),
                "noise": float(noise)
                if np.isfinite(noise)
                else np.nan,
                "score": float(score),
            }
        )

    return summaries


def _dimension_residual_screen(y_true, y_pred, n_boot=200, confidence=0.95, random_seed=0):
    """Stringent 2-D residual screen. It flags only a statistically worse dimension that is also above its noise floor."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if y_true.ndim != 2 or y_true.shape[1] != 2:
        return {"residual_problem": False, "reason": "not_2d"}

    residual = y_true - y_pred
    finite = np.all(np.isfinite(residual) & np.isfinite(y_true), axis=1)
    y_true, residual = y_true[finite], residual[finite]
    n = len(residual)
    if n < 10:
        return {"residual_problem": False, "reason": "too_few_points", "n_points": n}

    noise = np.asarray([_estimate_noise(y_true[:, 0]), _estimate_noise(y_true[:, 1])], dtype=float)
    rmse = np.sqrt(np.mean(residual ** 2, axis=0))
    scores = np.divide(rmse, noise, out=np.full(2, np.inf), where=np.isfinite(noise) & (noise > 1e-12))
    if not np.all(np.isfinite(scores)):
        return {"residual_problem": False, "reason": "noise_estimate_unavailable", "scores": scores.tolist(), "noise": noise.tolist(), "rmse": rmse.tolist()}

    block_len = max(2, min(n, int(round(n ** (1.0 / 3.0)))))
    n_blocks = int(np.ceil(n / block_len))
    rng = np.random.default_rng(random_seed)
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    indices = (starts[:, :, None] + np.arange(block_len)[None, None, :]) % n
    indices = indices.reshape(n_boot, -1)[:, :n]
    boot_rmse = np.sqrt(np.mean(residual[indices] ** 2, axis=1))
    boot_scores = boot_rmse / noise[None, :]
    boot_delta = boot_scores[:, 0] - boot_scores[:, 1]

    alpha = (1.0 - confidence) / 2.0
    score_ci = np.quantile(boot_scores, [alpha, 1.0 - alpha], axis=0)
    delta_ci = np.quantile(boot_delta, [alpha, 1.0 - alpha])
    difference_significant = bool(delta_ci[0] > 0.0 or delta_ci[1] < 0.0)
    worse_dim = int(np.argmax(scores))
    worse_dim_above_noise = bool(score_ci[0, worse_dim] > 1.0)
    residual_problem = bool(difference_significant and worse_dim_above_noise)

    reason = f"dimension_{worse_dim}_worse_and_above_noise" if residual_problem else "joint_fit_passed_residual_screen"
    return {
        "residual_problem": residual_problem, "reason": reason, "scores": scores.tolist(), "rmse": rmse.tolist(),
        "noise": noise.tolist(), "score_ci": score_ci.T.tolist(), "score_difference": float(scores[0] - scores[1]),
        "score_difference_ci": delta_ci.tolist(), "difference_significant": difference_significant,
        "worse_dimension": worse_dim, "block_length": block_len, "bootstrap_samples": int(n_boot),
        "confidence": float(confidence), "n_points": n,
    }


def _single_dimension_underfit_screen(y_true, y_pred, n_boot=200, confidence=0.95, random_seed=0):
    """Return True only when the residual score is demonstrably above the natural noise-floor reference of 1."""
    y_true, y_pred = np.asarray(y_true, dtype=float).reshape(-1), np.asarray(y_pred, dtype=float).reshape(-1)
    residual = y_true - y_pred
    finite = np.isfinite(y_true) & np.isfinite(residual)
    y_true, residual = y_true[finite], residual[finite]
    n = len(residual)
    if n < 10:
        return {"underfit": False, "reason": "too_few_points", "n_points": n}

    noise = _estimate_noise(y_true)
    if not np.isfinite(noise) or noise <= 1e-12:
        return {"underfit": False, "reason": "noise_estimate_unavailable", "noise": float(noise) if np.isfinite(noise) else np.nan}

    rmse = float(np.sqrt(np.mean(residual ** 2)))
    score = rmse / noise
    block_len = max(2, min(n, int(round(n ** (1.0 / 3.0)))))
    n_blocks = int(np.ceil(n / block_len))
    rng = np.random.default_rng(random_seed)
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    indices = (starts[:, :, None] + np.arange(block_len)[None, None, :]) % n
    indices = indices.reshape(n_boot, -1)[:, :n]
    boot_scores = np.sqrt(np.mean(residual[indices] ** 2, axis=1)) / noise

    alpha = (1.0 - confidence) / 2.0
    ci = np.quantile(boot_scores, [alpha, 1.0 - alpha])
    underfit = bool(ci[0] > 1.0)
    return {
        "underfit": underfit, "score": float(score), "score_ci": ci.tolist(), "rmse": rmse, "noise": float(noise),
        "block_length": block_len, "bootstrap_samples": int(n_boot), "confidence": float(confidence), "n_points": n,
    }


def _model_bic(y_true, y_pred, n_params):
    """BIC from ordinary residuals. For multi-D data each dimension gets its own residual variance term."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]

    total_n, likelihood_term, rss_by_dim = 0, 0.0, []
    eps = np.finfo(float).tiny
    for dim in range(y_true.shape[1]):
        residual = y_true[:, dim] - y_pred[:, dim]
        residual = residual[np.isfinite(residual)]
        n = len(residual)
        if n == 0:
            return np.inf, []
        rss = max(float(np.sum(residual ** 2)), eps)
        likelihood_term += n * np.log(rss / n)
        total_n += n
        rss_by_dim.append(rss)

    return float(likelihood_term + int(n_params) * np.log(total_n)), rss_by_dim


def _linear_prediction(x, y):
    """Closed-form no-rupture baseline y = a + bx."""
    x, y = np.asarray(x, dtype=float).reshape(-1), np.asarray(y, dtype=float).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y)
    pred = np.full_like(y, np.nan, dtype=float)
    if finite.sum() < 2:
        return pred
    A = np.column_stack((np.ones(finite.sum()), x[finite]))
    coef, _, _, _ = np.linalg.lstsq(A, y[finite], rcond=None)
    pred[finite] = coef[0] + coef[1] * x[finite]
    return pred


def _dimension_rupture_param_count(n_rup):
    """Effective parameter count for a one-dimension rupture model."""
    return 2 + 4 * int(n_rup)


def _effective_sample_size(residual):
    """Estimate correlation-adjusted sample size with an initial-positive-sequence autocorrelation time."""
    r = np.asarray(residual, dtype=float).reshape(-1)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 3:
        return float(max(n, 1)), 1.0

    r = r - np.mean(r)
    denom = float(np.dot(r, r))
    if denom <= np.finfo(float).eps:
        return float(n), 1.0

    rho_sum = 0.0
    for lag in range(1, n // 2 + 1):
        rho = float(np.dot(r[:-lag], r[lag:]) / denom)
        if not np.isfinite(rho) or rho <= 0.0:
            break
        rho_sum += rho

    tau = max(1.0, 1.0 + 2.0 * rho_sum)
    n_eff = max(2.0, min(float(n), float(n) / tau))
    return n_eff, tau


def _effective_bic(y_true, y_pred, n_params, n_eff):
    """Approximate BIC using an effective sample size so correlated profile points do not count as independent evidence."""
    y_true, y_pred = np.asarray(y_true, dtype=float).reshape(-1), np.asarray(y_pred, dtype=float).reshape(-1)
    residual = y_true - y_pred
    residual = residual[np.isfinite(residual)]
    if len(residual) == 0:
        return np.inf
    mse = max(float(np.mean(residual ** 2)), np.finfo(float).tiny)
    n_eff = max(float(n_eff), 2.0)
    return float(n_eff * np.log(mse) + int(n_params) * np.log(n_eff))


def _block_improvement_test(y_true, simple_pred, complex_pred, block_len, n_boot=200, confidence=0.95, random_seed=0):
    """Test whether the complex prediction gives a stable squared-error improvement over the simple prediction."""
    y = np.asarray(y_true, dtype=float).reshape(-1)
    simple = np.asarray(simple_pred, dtype=float).reshape(-1)
    complex_ = np.asarray(complex_pred, dtype=float).reshape(-1)
    finite = np.isfinite(y) & np.isfinite(simple) & np.isfinite(complex_)
    y, simple, complex_ = y[finite], simple[finite], complex_[finite]
    n = len(y)
    if n < 10:
        return {"supported": False, "reason": "too_few_points", "n_points": n}

    improvement = (y - simple) ** 2 - (y - complex_) ** 2
    block_len = max(2, min(n, int(round(block_len))))
    n_blocks = int(np.ceil(n / block_len))
    rng = np.random.default_rng(random_seed)
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block_len)[None, None, :]) % n
    idx = idx.reshape(n_boot, -1)[:, :n]
    boot_means = np.mean(improvement[idx], axis=1)
    alpha = (1.0 - confidence) / 2.0
    ci = np.quantile(boot_means, [alpha, 1.0 - alpha])
    return {
        "supported": bool(ci[0] > 0.0), "mean_mse_improvement": float(np.mean(improvement)),
        "improvement_ci": ci.tolist(), "block_length": block_len, "bootstrap_samples": int(n_boot),
        "confidence": float(confidence), "n_points": n,
    }


def _dimension_signal_test(x, y_true, joint_pred, n_rup, n_boot=200, confidence=0.95):
    """
    Original signal/no-signal criterion: compare the joint rupture prediction with
    a 2-parameter linear/no-rupture baseline using ordinary BIC.

    n_boot and confidence are kept in the signature so the run_optimization call
    does not need special handling, but they are intentionally not used here.
    """
    tests = []
    k_rup = _dimension_rupture_param_count(n_rup)

    for dim in range(y_true.shape[1]):
        linear_pred = _linear_prediction(x, y_true[:, dim])
        linear_bic, linear_rss = _model_bic(y_true[:, dim], linear_pred, 2)
        rupture_bic, rupture_rss = _model_bic(y_true[:, dim], joint_pred[:, dim], k_rup)

        tests.append({
            "dimension": dim,
            "has_signal": bool(rupture_bic < linear_bic),
            "linear_bic": linear_bic,
            "rupture_bic": rupture_bic,
            "delta_bic_rupture_minus_linear": rupture_bic - linear_bic,
            "linear_rss": linear_rss[0] if linear_rss else np.nan,
            "rupture_rss": rupture_rss[0] if rupture_rss else np.nan,
        })

    return tests


def _probe_shared_geometry(model, x_train, y_train, x_eval, y_eval, joint_pred_dim, dim, device,
                           max_iter=40, n_boot=200, confidence=0.95):
    """Cheap local probe: optimize only shared loc/width for one dimension, then test whether improvement is significant."""
    probe = deepcopy(model).to(torch.device(device))
    for p in probe.parameters():
        p.requires_grad_(False)

    shared = []
    for i in range(probe.nrup):
        rup = probe.prof[probe.key_rup(i)]
        rup.loc.requires_grad_(True)
        rup.width.requires_grad_(True)
        shared.extend([rup.loc, rup.width])
    if not shared:
        return {"useful": False, "reason": "no_shared_parameters"}

    x_t = torch.tensor(np.asarray(x_train), dtype=torch.float32, device=torch.device(device)).reshape(-1, 1)
    y_t = torch.tensor(np.asarray(y_train), dtype=torch.float32, device=torch.device(device)).reshape(-1)
    opt = torch.optim.LBFGS(shared, lr=0.5, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        pred = probe(x_t)[:, dim]
        loss = torch.mean((pred - y_t) ** 2)
        loss.backward()
        return loss

    try:
        opt.step(closure)
    except RuntimeError:
        return {"useful": False, "reason": "probe_optimizer_failed"}

    probe_pred = _predict_at_x(probe, x_eval, device)[:, dim]
    _, tau = _effective_sample_size(np.asarray(y_eval) - np.asarray(joint_pred_dim))
    improvement = _block_improvement_test(
        y_eval, joint_pred_dim, probe_pred, block_len=max(2.0, tau), n_boot=n_boot,
        confidence=confidence, random_seed=100 + dim)
    return {"useful": bool(improvement["supported"]), "reason": "significant_probe_improvement" if improvement["supported"] else "probe_not_significantly_better",
            "improvement_test": improvement}


def _shared_gradient_conflict(model, x, y_true, device):
    """Check whether the two dimensions push shared loc/width parameters in opposite directions."""
    if getattr(model, "ndim", 1) != 2:
        return {"conflict": False, "reason": "not_2d"}

    shared = []
    for i in range(model.nrup):
        rup = model.prof[model.key_rup(i)]
        shared.extend([rup.loc, rup.width])
    if not shared:
        return {"conflict": False, "reason": "no_shared_rupture_parameters"}

    original_requires_grad = [p.requires_grad for p in shared]
    for p in shared:
        p.requires_grad_(True)

    x_tensor = torch.tensor(np.asarray(x), dtype=torch.float32, device=torch.device(device)).reshape(-1, 1)
    y_tensor = torch.tensor(np.asarray(y_true), dtype=torch.float32, device=torch.device(device))
    pred = model(x_tensor)
    dim_losses = [torch.mean((pred[:, d] - y_tensor[:, d]) ** 2) for d in range(2)]
    grad_vectors = []

    try:
        for d in range(2):
            grads = torch.autograd.grad(dim_losses[d], shared, retain_graph=d == 0, allow_unused=True)
            pieces = [torch.zeros_like(p).reshape(-1) if g is None else g.reshape(-1) for g, p in zip(grads, shared)]
            grad_vectors.append(torch.cat(pieces).detach())
    finally:
        for p, flag in zip(shared, original_requires_grad):
            p.requires_grad_(flag)

    g0, g1 = grad_vectors
    norm0, norm1 = torch.linalg.vector_norm(g0).item(), torch.linalg.vector_norm(g1).item()
    if norm0 <= 1e-12 or norm1 <= 1e-12:
        return {"conflict": False, "reason": "one_gradient_near_zero", "norms": [norm0, norm1], "cosine": np.nan}

    dot = torch.dot(g0, g1).item()
    cosine = dot / (norm0 * norm1)
    return {"conflict": bool(dot < 0.0), "reason": "opposing_shared_gradients" if dot < 0.0 else "shared_gradients_not_opposed",
            "dot": float(dot), "cosine": float(cosine), "norms": [float(norm0), float(norm1)]}


def _num_model_params(model):
    return int(sum(p.numel() for p in model.parameters()))


def _actual_rup_locs(model):
    """Return fitted bounded rupture locations in normalized x coordinates."""
    locs = []
    for i in range(model.nrup):
        rup = model.prof[model.key_rup(i)]
        raw = rup.loc.detach().cpu().reshape(-1)[0].item()
        low, high = rup.loc_bounds
        locs.append(float(low + (high - low) / (1.0 + np.exp(-raw))))
    return locs


def _fit_selected_dimensions(smooth_data, dimensions, prof_num, selected_n_rup, selected_rup_locs, device, rand, history, x_eval):
    """Fit only requested dimensions; no-signal dimensions are never sent through NN_optimize again."""
    fits = {}
    for dim in dimensions:
        one_dim_data = np.column_stack((smooth_data[:, 0], smooth_data[:, dim + 1]))
        dim_data_obj = Data(one_dim_data, prof_num, init_p=None, device=device, rand=rand,
                            forced_n_rup=selected_n_rup, forced_rup_locs=selected_rup_locs)
        dim_model, dim_losses = _fit_model(dim_data_obj, history=history, rand=rand)
        fits[dim] = {"data_obj": dim_data_obj, "model": dim_model, "losses": dim_losses,
                     "pred_eval": _predict_at_x(dim_model, x_eval, device)}
    return fits


def _combine_loss_histories(loss_dicts):
    histories = [np.asarray(loss["total_loss"], dtype=float) for loss in loss_dicts]
    max_len = max(len(h) for h in histories)
    padded = [np.pad(h, (0, max_len - len(h)), mode="edge") if len(h) < max_len else h for h in histories]
    return {"total_loss": np.mean(np.vstack(padded), axis=0).tolist(), "states": [], "dimension_losses": loss_dicts}


def _to_numpy(values):
    return values.detach().cpu().numpy() if torch.is_tensor(values) else np.asarray(values)


def _table_dimension_labels(table, n_dim):
    """Map internal zero-based dimension indices to the labels emitted by post_process_params_v4 (normally 1, 2, ...)."""
    if "Dimension" not in table.columns:
        return {dim: dim for dim in range(n_dim)}
    labels = [v for v in pd.unique(table["Dimension"]) if pd.notna(v)]
    labels = sorted(labels)
    return {dim: labels[dim] if dim < len(labels) else dim + 1 for dim in range(n_dim)}


def _replace_dimension_table(table, dim_table, dim, dimension_labels):
    if "Dimension" not in table.columns or "Dimension" not in dim_table.columns:
        return table
    target_label = dimension_labels[dim]
    dim_table = dim_table.copy()
    dim_table["Dimension"] = target_label
    out = pd.concat((table.loc[table["Dimension"] != target_label].copy(), dim_table), ignore_index=True)
    sort_cols = [c for c in ("Dimension", "Rupture") if c in out.columns]
    return out.sort_values(sort_cols, kind="stable").reset_index(drop=True) if sort_cols else out


def _remove_no_signal_dimensions(table, no_signal_dims, dimension_labels):
    """Remove no-signal dimension rows; if all dimensions are no-signal, retain one metadata row with fit fields NaN."""
    if not no_signal_dims or "Dimension" not in table.columns:
        return table
    labels_to_remove = [dimension_labels[d] for d in no_signal_dims]
    out = table.loc[~table["Dimension"].isin(labels_to_remove)].copy()
    if not out.empty:
        return out.reset_index(drop=True)

    placeholder = table.iloc[[0]].copy()
    placeholder["Dimension"] = np.nan
    fit_names = {"rupture", "disp", "displacement", "width", "actual width", "loc", "rupture loc",
                 "rupture location", "slope", "ramp", "origin", "loss", "final loss"}
    for col in placeholder.columns:
        if str(col).strip().lower() in fit_names:
            placeholder[col] = np.nan
    return placeholder.reset_index(drop=True)


def run_optimization(smooth_data, orig_data, prof_num, sigma, rand=False, uncert=False, coords=None, win_bounds=None,
                     init_p=None, history=False, device="cpu", dimension_screen_bootstraps=200,
                     dimension_screen_confidence=0.95):
    """
    Staged model selection:
      1) normal joint fit;
      2) if detected ruptures > 3, compare detected count with a 2-rupture refit by BIC;
      3) test rupture signal in each dimension by rupture-vs-linear BIC;
      4) no-signal dimensions receive no fit (NaN curve / no parameter rows);
      5) expensive independent refits require residual evidence, opposing shared gradients, and a cheap loc/width probe that significantly improves the fit.
    """
    data_obj = Data(smooth_data, prof_num, init_p, device, rand)
    model, losses = _fit_model(data_obj, history=history, rand=rand)
    x_eval, y_eval, data_norm_orig = _prepare_original_eval_data(orig_data, data_obj)
    pred_eval = _predict_at_x(model, x_eval, device)
    bic, rss_by_dim = _model_bic(y_eval, pred_eval, _num_model_params(model))

    selection_info = {
        "detected_n_rup": int(data_obj.detected_n_rup), "initial_n_rup": int(data_obj.n_rup),
        "selected_n_rup": int(data_obj.n_rup), "rupture_candidates": {int(data_obj.n_rup): {"bic": bic, "rss_by_dim": rss_by_dim}},
        "dimension_model": "joint", "independent_refits_run": False, "refit_dimensions": [],
    }

    # Rupture-count selection happens before dimension screening.
    if data_obj.detected_n_rup > 3:
        two_rup_data = Data(smooth_data, prof_num, init_p=None, device=device, rand=rand, forced_n_rup=2)
        two_rup_model, two_rup_losses = _fit_model(two_rup_data, history=history, rand=rand)
        two_rup_pred_eval = _predict_at_x(two_rup_model, x_eval, device)
        two_rup_bic, two_rup_rss = _model_bic(y_eval, two_rup_pred_eval, _num_model_params(two_rup_model))
        selection_info["rupture_candidates"][2] = {"bic": two_rup_bic, "rss_by_dim": two_rup_rss}
        if two_rup_bic < bic:
            data_obj, model, losses, pred_eval = two_rup_data, two_rup_model, two_rup_losses, two_rup_pred_eval
            bic, rss_by_dim = two_rup_bic, two_rup_rss
        selection_info["selected_n_rup"] = int(data_obj.n_rup)

    independent_fits, selected_independent_dims = {}, []
    signal_dims = list(range(data_obj.n_dim))
    no_signal_dims = []

    if data_obj.n_dim == 2:
        signal_tests = _dimension_signal_test(x_eval, y_eval, pred_eval, data_obj.n_rup, n_boot=dimension_screen_bootstraps, confidence=dimension_screen_confidence)
        signal_dims = [item["dimension"] for item in signal_tests if item["has_signal"]]
        no_signal_dims = [d for d in range(2) if d not in signal_dims]
        selection_info["signal_tests"] = signal_tests
        selection_info["signal_dimensions"] = signal_dims
        selection_info["no_signal_dimensions"] = no_signal_dims

        gradient_check = _shared_gradient_conflict(model, x_eval, y_eval, device)
        selection_info["shared_gradient_check"] = gradient_check

        selected_rup_locs = (np.asarray(_actual_rup_locs(model)) * max(len(data_obj.x) - 1, 1)).tolist()

        # Exactly one signal-bearing dimension: refit only that dimension, and only when it is underfit AND
        # the two dimensions are fighting over shared loc/width. The no-signal dimension is never refit.
        if len(signal_dims) == 1:
            signal_dim = signal_dims[0]
            fit_screen = _single_dimension_underfit_screen(
                y_eval[:, signal_dim], pred_eval[:, signal_dim], n_boot=dimension_screen_bootstraps,
                confidence=dimension_screen_confidence, random_seed=0)
            selection_info["signal_dimension_fit_screen"] = fit_screen

            if fit_screen["underfit"] and gradient_check["conflict"]:
                probe = _probe_shared_geometry(
                    model, data_obj.x, data_obj.y[:, signal_dim], x_eval, y_eval[:, signal_dim],
                    pred_eval[:, signal_dim], signal_dim, device, n_boot=dimension_screen_bootstraps,
                    confidence=dimension_screen_confidence)
                selection_info["signal_dimension_probe"] = probe

                if probe["useful"]:
                    selection_info["independent_refits_run"] = True
                    selection_info["refit_dimensions"] = [signal_dim]
                    independent_fits = _fit_selected_dimensions(
                        smooth_data, [signal_dim], prof_num, data_obj.n_rup, selected_rup_locs,
                        device, rand, history, x_eval)

                    candidate = independent_fits[signal_dim]
                    k_dim = _dimension_rupture_param_count(data_obj.n_rup)
                    joint_dim_bic, joint_dim_rss = _model_bic(y_eval[:, signal_dim], pred_eval[:, signal_dim], k_dim)
                    independent_dim_bic, independent_dim_rss = _model_bic(
                        y_eval[:, signal_dim], candidate["pred_eval"][:, 0], _num_model_params(candidate["model"]))
                    selection_info["signal_dimension_candidates"] = {
                        "joint": {"bic": joint_dim_bic, "rss": joint_dim_rss[0]},
                        "independent": {"bic": independent_dim_bic, "rss": independent_dim_rss[0]},
                    }
                    if independent_dim_bic < joint_dim_bic:
                        selected_independent_dims = [signal_dim]
                        selection_info["dimension_model"] = "single_independent_with_no_signal"

            if not selected_independent_dims:
                selection_info["dimension_model"] = "joint_signal_only"

        # Both dimensions contain rupture signal: the expensive two-fit fallback requires BOTH a residual
        # discrepancy and opposing gradients on the shared loc/width parameters.
        elif len(signal_dims) == 2:
            residual_screen = _dimension_residual_screen(
                y_eval, pred_eval, n_boot=dimension_screen_bootstraps,
                confidence=dimension_screen_confidence, random_seed=0)
            selection_info["dimension_screen"] = residual_screen

            if residual_screen["residual_problem"] and gradient_check["conflict"]:
                worse_dim = residual_screen["worse_dimension"]
                probe = _probe_shared_geometry(
                    model, data_obj.x, data_obj.y[:, worse_dim], x_eval, y_eval[:, worse_dim],
                    pred_eval[:, worse_dim], worse_dim, device, n_boot=dimension_screen_bootstraps,
                    confidence=dimension_screen_confidence)
                selection_info["dimension_probe"] = probe

                if probe["useful"]:
                    selection_info["independent_refits_run"] = True
                    selection_info["refit_dimensions"] = [0, 1]
                    independent_fits = _fit_selected_dimensions(
                        smooth_data, [0, 1], prof_num, data_obj.n_rup, selected_rup_locs,
                        device, rand, history, x_eval)
                    independent_pred_eval = np.column_stack((independent_fits[0]["pred_eval"][:, 0], independent_fits[1]["pred_eval"][:, 0]))
                    independent_param_count = sum(_num_model_params(independent_fits[d]["model"]) for d in (0, 1))
                    independent_bic, independent_rss = _model_bic(y_eval, independent_pred_eval, independent_param_count)
                    selection_info["dimension_candidates"] = {
                        "joint": {"bic": bic, "rss_by_dim": rss_by_dim, "residual_scores": _residual_score_summary(y_eval, pred_eval)},
                        "independent": {"bic": independent_bic, "rss_by_dim": independent_rss,
                                        "residual_scores": _residual_score_summary(y_eval, independent_pred_eval)},
                    }
                    if independent_bic < bic:
                        selected_independent_dims = [0, 1]
                        selection_info["dimension_model"] = "independent"

        else:
            selection_info["dimension_model"] = "no_signal"

    # Post-process the selected rupture-count joint model once, then replace only dimensions whose independent
    # refit actually won. No-signal dimensions are removed from the parameter table and plotted as NaN.
    table, norm_vals, scaled_vals, lin_seg, _, org_p = post_process_params_v4(data_obj, model, coords, losses["total_loss"][-1])
    dimension_labels = _table_dimension_labels(table, data_obj.n_dim)
    norm_vals = np.asarray(norm_vals).copy()
    if norm_vals.ndim == 1:
        norm_vals = norm_vals[:, None]
    scaled_vals = _to_numpy(scaled_vals).copy()
    if scaled_vals.ndim == 1:
        scaled_vals = scaled_vals[:, None]

    final_model = model
    uncertainty_specs = []
    if selected_independent_dims:
        dim_losses = []
        final_models = [None] * data_obj.n_dim
        for dim in range(data_obj.n_dim):
            if dim in selected_independent_dims:
                fit = independent_fits[dim]
                dim_table, dim_norm, dim_scaled, dim_lin_seg, _, dim_org_p = post_process_params_v4(
                    fit["data_obj"], fit["model"], coords, fit["losses"]["total_loss"][-1])
                table = _replace_dimension_table(table, dim_table, dim, dimension_labels)
                norm_vals[:, dim] = np.asarray(dim_norm).reshape(-1)
                scaled_vals[:, dim] = _to_numpy(dim_scaled).reshape(-1)
                final_models[dim] = fit["model"]
                dim_losses.append(fit["losses"])
                uncertainty_specs.append((fit["data_obj"], dim_org_p, dim_lin_seg, dim_scaled))
            elif dim in signal_dims:
                final_models[dim] = model
        final_model = final_models
        if len(selected_independent_dims) == data_obj.n_dim:
            losses = _combine_loss_histories(dim_losses)
        elif len(selected_independent_dims) == 1:
            losses = independent_fits[selected_independent_dims[0]]["losses"]
    elif signal_dims:
        uncertainty_specs = [(data_obj, org_p, lin_seg, scaled_vals)]

    for dim in no_signal_dims:
        norm_vals[:, dim] = np.nan
        scaled_vals[:, dim] = np.nan
    table = _remove_no_signal_dimensions(table, no_signal_dims, dimension_labels)
    losses["model_selection"] = selection_info

    # No uncertainty is produced when no dimension contains supported rupture signal.
    if uncert and signal_dims:
        uncertainties = []
        for uncert_data_obj, uncert_org_p, uncert_lin_seg, uncert_scaled_vals in uncertainty_specs:
            uncertainty_data = np.hstack(((uncert_data_obj.x / uncert_data_obj.scale_shift[0]).reshape(-1, 1), uncert_data_obj.y))
            scaled_for_uncert = _to_numpy(uncert_scaled_vals)
            seg_i = 0
            for loc in uncert_org_p["loc"]:
                u, p1, p99 = calc_uncertainties(
                    uncertainty_data[uncert_lin_seg[seg_i][0]:uncert_lin_seg[seg_i][1]],
                    uncertainty_data[uncert_lin_seg[seg_i + 1][0]:uncert_lin_seg[seg_i + 1][1]],
                    loc.item(), win_bounds, 1000)
                uncertainties.append(plot_uncert_with_lines(
                    u, uncert_data_obj.x / uncert_data_obj.scale_shift[0], uncert_data_obj.y,
                    scaled_for_uncert, p1, p99))
                seg_i += 2
    else:
        uncertainties = None

    n_output_dims = smooth_data.shape[1] - 1
    if n_output_dims > 1:
        fig, ax = plt.subplots(figsize=(25, 15), nrows=1, ncols=2)
        flips = [data_obj.par_flip, data_obj.perp_flip]
        titles = ["First Dimension, Normalized", "Second Dimension, Normalized"]
        for dim in range(2):
            ax[dim].plot(data_norm_orig[:, 0], flips[dim] * data_norm_orig[:, dim + 1], "o")
            ax[dim].plot(data_norm_orig[:, 0], norm_vals[:, dim], "-")
            ax[dim].set_title(titles[dim] + (" (No rupture signal)" if dim in no_signal_dims else ""))
            ax[dim].set_xlabel("Horizontal Distance")
            ax[dim].set_ylabel("Displacement")
    else:
        fig, ax = plt.subplots(figsize=(20, 15), nrows=1, ncols=2)
        scaled_plot = scaled_vals.reshape(-1)
        x_plot = np.linspace(smooth_data[0, 0], smooth_data[-1, 0], len(scaled_plot))
        ax[0].plot(smooth_data[:, 0], data_obj.par_flip * smooth_data[:, 1], "o")
        ax[0].plot(x_plot, scaled_plot, "-", linewidth=3)
        ax[0].set_title("Smoothed Rescaled")
        ax[0].grid(True, linewidth=1.8)
        ax[0].tick_params(axis="both", labelsize=20)
        ax[0].set_xlabel("Distance Along Profile", fontdict={"size": 33})
        ax[0].set_ylabel("Displacement", fontdict={"size": 33})

        ax[1].plot(orig_data[:, 0], data_obj.par_flip * orig_data[:, 1], "o")
        ax[1].plot(x_plot, scaled_plot, "-", linewidth=3)
        ax[1].set_title("Rescaled")
        ax[1].grid(True, linewidth=1.8)
        ax[1].tick_params(axis="both", labelsize=20)
        ax[1].set_xlabel("Distance Along Profile", fontdict={"size": 33})
        ax[1].set_ylabel("Displacement", fontdict={"size": 33})

    return table, fig, final_model, uncertainties, losses

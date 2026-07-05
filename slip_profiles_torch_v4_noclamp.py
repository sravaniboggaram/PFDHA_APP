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

def regional_weighted_loss(y_pred, y_act, loc_params, width_params, scale):
    temp_loc = [float(p.detach()) for p in loc_params]
    temp_width = [float(p.detach()) for p in width_params]

    lin_seg, sig_seg, _ = split_profile(temp_loc, temp_width, scale=scale)
    # if j % 300 == 0:
    #     plt.plot(np.linspace(0, len(y_pred), len(y_pred.detach().numpy())), y_pred.detach().numpy())
    #     plt.plot(np.linspace(0, len(y_act), len(y_act.detach().numpy())), y_act.detach().numpy())
    #     plt.axvline(x=sig_seg[0][0], color='r')
    #     plt.axvline(x=sig_seg[0][1], color='r')
    #     plt.show()

    loss = torch.tensor(0.0)
    w_lin = 10.0
    w_sig = 1.0

    for i in range(len(sig_seg)):
        s1, e1, s2, e2 = lin_seg[i][0], lin_seg[i][1], sig_seg[i][0], sig_seg[i][1]
        lin_diff = (y_pred[s1:e1] - y_act[s1:e1])**2
        sig_diff = torch.abs(y_pred[s2:e2] - y_act[s2:e2])
        loss += w_lin*torch.sum(lin_diff) + w_sig*torch.sum(sig_diff)

    loss += w_lin*torch.sum(((y_pred[lin_seg[-1][0]: lin_seg[-1][1]] - y_act[lin_seg[-1][0]: lin_seg[-1][1]])**2))

    return loss


def penalty_loss_old(y_pred, y_act, locs=[1], lam=10):
    return torch.nn.MSELoss(reduction='sum')(y_pred, y_act)
    # if len(locs) == 1:
    #     return torch.nn.MSELoss(reduction='sum')(y_pred, y_act)
    # p = torch.tensor(0)
    # for l in range(1, len(locs)):
    #     p = torch.add(p, torch.tensor(locs[l].item() - locs[l-1].item()))
    # p = torch.subtract(p, torch.tensor(0.5*len(locs)-1))
    # mse = torch.nn.MSELoss(reduction='sum') 
    # return torch.subtract(mse(y_pred, y_act), lam*p)


def loc_loss(y_pred, y_act, loc_params):
    if len(loc_params) == 1:
        return torch.nn.MSELoss()(y_pred, y_act)

    loc_penalty = torch.tensor(1, dtype=torch.float32)
    for i in range(1, len(loc_params)):
        loc_penalty = torch.add(loc_penalty,
                                (loc_params[i] - loc_params[i-1]) - torch.tensor(0.1, dtype=torch.float32),
                                alpha=3)
    
    return torch.subtract(torch.nn.MSELoss()(y_pred, y_act), loc_penalty)

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
    loss_fn = torch.nn.MSELoss()

    y_b = torch.tensor(y, dtype=torch.float32, device=device)
    y_pred = model(x_tensor)
    if data.n_dim > 1:
        y_b = torch.concat((y_b[:,0], y_b[:,1]))
        y_pred = torch.concat((y_pred[:,0], y_pred[:,1]))

    prev_loss = loss_fn(y_pred, y_b).item()
    losses['total_loss'].append(prev_loss)
        
    def training_loop(ratio_limit, iter_limit, vars, prev_loss, lr=1e-3, lf=None):
        opt.param_groups[0]['lr'] = lr
        set_trainable(model.named_parameters(), vars)

        opt.zero_grad()
        ratio = 0.0
        iter_n = 0

        while True:
            iter_n += 1
            if collect_param_vals:
                losses['states'].append(model.state_dict())
            
            y_pred = model(x_tensor)

            if data.n_dim > 1:
                y_pred = torch.concat((y_pred[:,0], y_pred[:,1]))
                    
            loss = loss_fn(y_pred, y_b) if lf == "L1" else loss_fn(y_pred, y_b)
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
        
        return {'mse': rmse(y_pred, y_b).item(),
                #"regional": regional_weighted_loss(y_pred, y_b, loc_p, width_p, scale).item(),
                'L1': L1(y_pred, y_b).item()}
        #total_n_epochs += iter_n

    l = training_loop(0.99, 20000, ["all_true"], prev_loss)
    prev_loss = l['mse']

    loss_fn = torch.nn.L1Loss()
    l = training_loop(0.999, 4000, ["loc", "width"], prev_loss, lr=1e-4, lf="L1")
    prev_loss = l['mse']

    loss_fn = torch.nn.MSELoss()
    training_loop(0.9999, 3000, ["ramp", "slope", "origin", "disp"], prev_loss, lr=1e-4,lf="L1")

    # fig, ax = plt.subplots(nrows=2, ncols=1)
    # ax[0].plot(x, y)
    # ax[0].plot(x, model(x_tensor).detach().numpy())

    # lin_seg, _, _ = split_profile([param_transf_to_act(l.item(), data.param_bounds['loc']) for l in loc_p],
    #                               [param_transf_to_act(w.item(), data.param_bounds['width']) for w in width_p],
    #                               scale)
    # ax[0].axvline(x=lin_seg[0][1])
    # ax[0].axvline(x=lin_seg[1][0])
    # plt.show()

    # for name, p in model.named_parameters():
    #     #print(name, p)
    #     if "d0.ramp" in name:
    #         ramp = stats.linregress(x[lin_seg[0][0]:lin_seg[0][1]].flatten(), y[:,0][lin_seg[0][0]:lin_seg[0][1]].flatten())
    #         ramp_slope, _ = ramp.slope, ramp.intercept
    #         p.data *= 0
    #         p.data += torch.tensor(param_act_to_transf(ramp_slope, data.param_bounds['ramp']), dtype=torch.float32)
    #     elif "d1.ramp" in name:
    #         ramp = stats.linregress(x[lin_seg[0][0]:lin_seg[0][1]].flatten(), y[:,1][lin_seg[0][0]:lin_seg[0][1]].flatten())
    #         ramp_slope, _ = ramp.slope, ramp.intercept
    #         p.data *= 0
    #         p.data += torch.tensor(ramp_slope, dtype=torch.float32)

    # l = training_loop(0.9999, 5000, ["disp"], prev_loss, lr=1e-4)
    
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

    print("OPTIMIZER LOSS: ", losses['total_loss'][-1])

    print("Done! Profile "+str(data.prof_id))

    return model, losses

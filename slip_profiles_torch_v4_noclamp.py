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

    total = 0
        
    def training_loop(ratio_limit, iter_limit, vars, lr=1e-3):
        opt.param_groups[0]['lr'] = lr
        set_trainable(model.named_parameters(), vars)

        y_pred = model(x_tensor)
        
        if data.n_dim > 1:
            y_pred = torch.concat((y_pred[:,0], y_pred[:,1]))
                
        prev_loss = loss_fn(y_pred, y_b)

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
                    
            loss = loss_fn(y_pred, y_b)
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
        
        return iter_n

    n = training_loop(0.999, 20000, ["all_true"])

    total += n

    loss_fn = torch.nn.L1Loss()
    n = training_loop(0.999, 4000, ["loc", "width"], lr=1e-4)

    total += n

    loss_fn = torch.nn.MSELoss()
    n = training_loop(0.9999, 4000, ["ramp", "slope", "origin", "disp"], lr=1e-4)

    total += n
    
    for name, p in model.named_parameters():
        print(name, p)
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

    return model, losses, total

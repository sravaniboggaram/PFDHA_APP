import torch
from torch import nn
from torch import Tensor
from torch.nn import functional
from helper_funs import *


# Slip Profile Layers
# ---   ---   ---   ---   ---
class LayerBase1Dim(nn.Module):
    def __init__(self, seed_origin:float, seed_ramp:float) -> None:
        super().__init__()

        self.origin = seed_origin
        self.ramp = seed_ramp

        #optimizable parameters
        self.ramp   = nn.Parameter(torch.tensor([[seed_ramp]], dtype=float), requires_grad=True) #slope of base profile
        self.origin = nn.Parameter(torch.tensor([seed_origin], dtype=float), requires_grad=True) #displacment at origin


    def forward(self, s:Tensor) -> Tensor:
        '''Compute displacement of base profile (origin and slope)'''

        #transform profile axis
        d = functional.linear(s, self.ramp, self.origin)
        
        return d

class LayerSingleRup1Dim(nn.Module):
    '''Single Rupture Layer (1 Dimensional)'''
    def __init__(self, seed_ymax:float, seed_slope:float, seed_width) -> None:
        super().__init__()

        self.width = seed_width

        #optimizable parameters
        self.y_max  = nn.Parameter(torch.tensor([seed_ymax], dtype=float), requires_grad=True)  #displacement
        self.slope = nn.Parameter(torch.tensor([seed_slope], dtype=float), requires_grad=True) #slope
        
    def forward(self, s_sig:Tensor, s_hinge: Tensor, base_s_max: Tensor, sig_max: Tensor, hinge_max: Tensor) -> Tensor:
        '''Compute displacement from single rupture 
           (zero displacement at origin, rupture location at s=0)'''
        
        sigmoid = functional.sigmoid(s_sig)
        hinge = self.slope*functional.softplus(s_hinge, beta=self.width.item()) # scale with width as par

        disp = (self.y_max - base_s_max - self.slope*hinge_max) / sig_max

        return disp*sigmoid + hinge

class LayerSingleRupMDim(nn.Module):
    def __init__(self, ndim:int, 
                 seed_loc:float, seed_width,
                 seed_ymax, seed_slope, s_max: float) -> None:
        super().__init__()
        
        #initialize seed if unspecified
        if seed_ymax is None:  seed_ymax  = [1.0] * ndim
        if seed_slope is None: seed_slope = [0.0] * ndim

        #fixed parameters
        self.ndim = ndim
        self.s_max = s_max
        #optimizable parameters
        self.loc   = nn.Parameter(torch.tensor([seed_loc], dtype=float), requires_grad=True)     #rupture location
        self.width = nn.Parameter(torch.tensor([[seed_width]], dtype=float), requires_grad=True) #rupture width

        #building block layers
        self.prof = nn.ModuleDict([[self.key_dim(j), LayerSingleRup1Dim(seed_ymax[j], seed_slope[j], self.width.data)] 
                                   for j in range(self.ndim)])
    
    def key_dim(self, j:int) -> str:
        
        return 'd%i'%j
    
    def forward(self, s:Tensor, base_smax: Tensor) -> Tensor:
        '''Compute displacement from single rupture, multiple dimensions
           (zero displacement at origin, rupture location from linear layer)'''
        
        #transform profile axis
        s_sig = functional.linear(s, self.width, -self.loc*self.width)
        s_hinge = functional.linear(s, torch.tensor([[1.0]], dtype=float), -self.loc)

        
        sig_max = functional.sigmoid(self.width*(self.s_max - self.loc))
        hinge_max = functional.softplus(self.s_max - self.loc, beta=self.width.item())
        
        #compute displacement multiple dimenstions
        d = torch.cat([self.prof[self.key_dim(j)](s_sig, s_hinge, base_smax[:,j:j+1], sig_max, hinge_max) for j in range(self.ndim)], dim=1)


        return d

# Slip Profile Neural Network
# ---   ---   ---   ---   ---
class SlipProfileNN(nn.Module):
    def __init__(self, ndim:int=1, nrup:int=1,
                 seed_origin = None, seed_ramp = None, 
                 seed_loc = None, seed_width = None,
                 seed_ymax = None, seed_slope = None, s_max = None) -> None:
        super().__init__()


        #initialize seed for base profile parameters if unspecified
        if seed_origin is None: seed_origin = [0.0] * ndim
        if seed_ramp is None: seed_ramp  = [0.0] * ndim
        #initialize seed for slip profile parameters if unspecified
        if seed_loc   is None: seed_loc   = [(l+0.5)/nrup for l in range(nrup)]
        if seed_width is None: seed_width = [100.] * nrup
        if seed_ymax  is None: seed_ymax  = [None] * nrup
        if seed_slope is None: seed_slope = [None] * nrup

        #fixed parameters
        self.nrup = nrup #number of ruptures
        self.ndim = ndim #number of dimensions
        self.s_max = torch.tensor(s_max, dtype=float)

        #building block layers
        self.prof = nn.ModuleDict([[self.key_rup(l), LayerSingleRupMDim(ndim, seed_loc[l], seed_width[l], 
                                                                              seed_ymax[l], seed_slope[l],
                                                                              s_max)] 
                                   for l in range(self.nrup)])
        self.base = nn.ModuleDict([[self.key_dim(j), LayerBase1Dim(seed_origin[j], seed_ramp[j])] 
                                   for j in range(self.ndim)])

    def key_rup(self, l:int) -> str:
        
        return 'r%i'%l

    def key_dim(self, j:int) -> str:    

        #inherit key_dim method from LayerSingleRupMDim
        return self.prof[self.key_rup(0)].key_dim(j)
    
    def forward(self, s:Tensor) -> Tensor:
        '''Compute displacement from multiple ruptres'''
        #base profile (origin and linear slope)
        d = torch.cat([self.base[self.key_dim(j)](s) for j in range(self.ndim)], dim=1)

        s_max = torch.ones_like(s)*self.s_max

        base_smax = torch.cat([self.base[self.key_dim(j)](s_max) for j in range(self.ndim)], dim=1)

        #add displacement of each rupture
        for l in range(self.nrup):
            d += self.prof[self.key_rup(l)](s, base_smax)
        
        return d
    

def set_trainable(model_params, bool_val, *args):
    if args[0] == "all":
        for name, param in model_params:
            param.requires_grad = bool_val
    else:
        for a in args:
            for name, param in model_params:
                if a in name:
                    param.requires_grad = bool_val
                else:
                    param.requires_grad = not bool_val
        

def rmse(y_pred, y_act):
    return torch.sqrt(torch.mean((y_pred - y_act)**2)).detach().numpy()


def NN_optimize(data, collect_param_vals=False):
    model = SlipProfileNN(ndim=data.n_dim, nrup=data.n_rup,
                          seed_origin=data.param_0['origin'],
                          seed_ramp=data.param_0['ramp'],
                          seed_loc=data.param_0['loc'],
                          seed_width=data.param_0['width'],
                          seed_ymax=[[data.y[-1]]],
                          seed_slope=data.param_0['slope'],
                          s_max=data.x[-1])
    
    named_params = model.named_parameters()

    x, y = data.x, data.y
    learn_rate, n_epoch = data.lr, data.n_epochs
    x_tensor = torch.tensor(x, dtype=float).unsqueeze(0).T
    
    width_p = []
    non_width_p = []
    losses = {'total_loss':[], 'states': []}

    for name, param in named_params:
        if collect_param_vals:
            losses[name] = []
        if name[-5:] == "width":
            width_p.append(param)
        else:
            non_width_p.append(param)
    opt_params = [
        {'params': width_p, 'lr':1},
        {'params': non_width_p, 'lr': learn_rate}
    ]

    opt = torch.optim.ASGD(opt_params)
    loss_fn = torch.nn.L1Loss(reduction='sum')

    scheduler1 = torch.optim.lr_scheduler.LinearLR(opt)
    scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min')

    y_b = torch.tensor(y)
    y_pred = model(x_tensor)
    if data.n_dim > 1:
        y_b = torch.concat((y_b[:,0], y_b[:,1]))
        y_pred = torch.concat((y_pred[:,0], y_pred[:,1]))

    prev_loss = loss_fn(y_pred, y_b).item()
    losses['total_loss'].append(prev_loss)
    ratio = 0.0
    iter_n = 0

    while True:
        iter_n += 1
        if collect_param_vals:
            losses['states'].append(model.state_dict())
        y_pred = model(x_tensor)

        if data.n_dim > 1:
            y_pred = torch.concat((y_pred[:,0], y_pred[:,1]))
            
        with torch.no_grad():
            for name, param in model.named_parameters():
                if collect_param_vals:
                    losses[name].append(param)
                ind = name.rfind(".") + 1
                temp = name[ind:]
                if temp == 'y_max':
                    param.requires_grad = True
                    continue
                param.data = torch.clamp(param, min=data.param_bounds[temp][0], max=data.param_bounds[temp][1]).data
                param.requires_grad = True
                
        #loss = loss_fn(y_pred, y_b)
        loss = loss_fn(y_pred, y_b)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if iter_n%100 == 0:
            ratio = loss.item()/prev_loss
            prev_loss = loss.item()
            #print(iter_n,ratio)

        scheduler1.step()
        scheduler2.step(loss)
        losses['total_loss'].append(loss.item())

        if n_epoch is not None and iter_n >= n_epoch:
            break
        elif ratio >= 0.999 or iter_n >= 20000:
            break


    # df["RMSE Init"] = [rmse_init]
    # df["RMSE Final"] = [rmse_final]

    print("OPTIMIZER")
    for name, p in model.named_parameters():
        print(name, p.item())

    print("OPTIMIZER LOSS: ", loss.item())

    print("Done! Profile "+str(data.prof_id))

    fig, ax = plt.subplots(figsize=(10,10))

    ax.plot(data.x, data.y, 'o')
    ax.plot(data.x, model(x_tensor).detach().numpy(), '-')
    fig.savefig('output.png')

    return model, losses

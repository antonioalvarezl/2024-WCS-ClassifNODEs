import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint

# Set the maximum number of steps for the ODE solver
MAX_NUM_STEPS = 1000

# Define custom activation functions and their derivatives
def tworelu(input):
    return torch.relu(input)**2 

def trunrelu(input):
    return torch.clamp(input, min=0, max=1)

def tanh_prime(input):
    # Derivative of the tanh function
    return 1 - torch.tanh(input) ** 2

def relu_prime(input):
    # Derivative of the ReLU function
    return (input >= 0).float()

# Dictionary of activation functions
activations = {
    'tanh': nn.Tanh(),
    'relu': nn.ReLU(),
    'sigmoid': nn.Sigmoid(),
    'leakyrelu': nn.LeakyReLU(negative_slope=0.25, inplace=True),
    '2relu': tworelu,
    'trunrelu': trunrelu,
    'tanh_prime': tanh_prime,
    'relu_prime': relu_prime,               
}

# Dictionary mapping architecture types
architectures = {'inside': -1, 'outside': 0, 'bottleneck': 1}

class Dynamics(nn.Module):
    """
    Defines the nonlinear dynamics for the neural ODE. 
    Supports multiple architectures and activation functions.
    """
    def __init__(self, device, input_dim, hidden_dim, non_linearity='tanh', architecture='inside', 
                 T=10, num_vals=10, seed_params=1):
        super(Dynamics, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        torch.manual_seed(seed_params)
        torch.cuda.manual_seed_all(seed_params)
        self.sigma = non_linearity
        # Ensure valid non_linearity and architecture
        if non_linearity not in activations or architecture not in architectures:
            raise ValueError("Invalid activation function or architecture type.")

        self.non_linearity = activations[non_linearity]
        self.architecture = architectures[architecture]
        self.T = T
        self.num_vals = num_vals
        if self.architecture == 1:
            ##-- R^{d_input} -> R^{d_hid} layer -- 
            self.fc1_time = nn.Sequential(*[nn.Linear(self.input_dim, self.hidden_dim) for _ in range(self.num_vals)])
            ##-- R^{d_hid} -> R^{d_input} layer --
            self.fc3_time = nn.Sequential(*[nn.Linear(self.hidden_dim, self.input_dim, bias=False) for _ in range(self.num_vals)])
        else:
            ##-- R^{d_input} -> R^{d_input} layer --
            self.fc2_time = nn.Sequential(*[nn.Linear(self.input_dim, self.input_dim) for _ in range(self.num_vals)])

    def forward(self, t, x):
        """
        Compute the dynamics f(x(t), u(t)) based on the current time t and input x.
        """
        delta_t = self.T / self.num_vals
        k = int(t / delta_t)

        if self.architecture == 1:  # Bottleneck architecture
            x = self.non_linearity(self.fc1_time[k](x))
            return self.fc3_time[k](x)
        else:  # Inside or outside architecture
            fc2 = self.fc2_time[k]
            return self.non_linearity(fc2(x)) if self.architecture == -1 else fc2(self.non_linearity(x))
        
class Semiflow(nn.Module):
    """
    Computes the semiflow x'(t) = f(theta(t), x(t)) for a given neural ODE.
    Supports both standard and adjoint ODE integration.
    """
    def __init__(self, device, dynamics, adjoint=False, T=10, step_size=0.1, method='euler', seed_params=1):
        super(Semiflow, self).__init__()
        self.adjoint = adjoint
        self.device = device
        self.dynamics = dynamics
        self.T = T
        self.dt = step_size
        self.method = method
        torch.manual_seed(seed_params)
        torch.cuda.manual_seed_all(seed_params)
        
    def forward(self, x, eval_times=None):
        if eval_times is None:
            integration_time = torch.tensor([0, self.T]).float().type_as(x)
        else:
            integration_time = eval_times.type_as(x)

        if self.adjoint:  
            out = odeint_adjoint(self.dynamics, x, integration_time, method=self.method, options={'step_size': self.dt})
            # out = odeint_adjoint(self.dynamics, x, integration_time, rtol = 0.1, atol = 0.1, method='dopri5', options={'max_num_steps': MAX_NUM_STEPS})
        else:   
            out = odeint(self.dynamics, x, integration_time, method=self.method, options={'step_size': self.dt})                    
            # out = odeint(self.dynamics, x, integration_time, rtol = 0.1, atol = 0.1, method='dopri5', options={'max_num_steps': MAX_NUM_STEPS})

            #odeint Returns:
            #         y: Tensor, where the first dimension corresponds to different
            #             time points. Contains the solved value of y for each desired time point in
            #             `t`, with the initial value `y0` being the first element along the first
            #             dimension.
            
            #i need to put the out into the odeint for the adj_out
            # adj_out = odeint(self.adj_dynamics, torch.eye(x.shape[0]), torch.flip(integration_time,[0]), method='euler', options={'step_size': self.step_size}) #this is new for the adjoint
        if eval_times is None:
            if out is not None:
                return out[1]
        else:
            return out

    def trajectory(self, x, timesteps):
        """
        Returns the full trajectory of the state over time.
        """
        integration_time = torch.linspace(0., self.T, timesteps)
        return self.forward(x, eval_times=integration_time)

class NeuralODE(nn.Module):
    """
    Neural ODE implementation that encapsulates the dynamics, semiflow, and optional final layer.
    """
    def __init__(self, device, fixed_projector, input_dim, hidden_dim, output_dim=2, non_linearity='tanh',
                 adjoint=False, architecture='inside', T=10, num_vals=10, step_size=0.1, method='euler', 
                 final_layer=False, seed_params=1):
        super(NeuralODE, self).__init__()
        self.device = device
        self.input_dim, self.hidden_dim, self.output_dim = input_dim, hidden_dim, output_dim
        self.T, self.num_vals, self.dt, self.sigma = T, num_vals, step_size, non_linearity
        self.architecture, self.fixed_projector, self.non_linearity = architecture, fixed_projector, activations[non_linearity]
        self.method, self.adjoint, self.seed_params = method, adjoint, seed_params
        self.final_layer = final_layer
        torch.manual_seed(self.seed_params)
        torch.cuda.manual_seed_all(self.seed_params)

        self.best_param = []
        self.trained = False
        self.fwd_dynamics = Dynamics(device, input_dim, hidden_dim, non_linearity, architecture, self.T, self.num_vals, seed_params) 
        self.flow = Semiflow(device, self.fwd_dynamics, adjoint, T, step_size, method, seed_params)
        #self.adj_flow = Semiflow(device, adj_dynamics, adjoint, T, num_vals, step_size, seed_params)  
        if self.final_layer:
            if not fixed_projector:
                self.linear_layer = nn.Linear(self.flow.dynamics.input_dim, self.output_dim)
            else: 
                self.linear_layer = fixed_projector
            
    def forward(self, x, return_features=False, i=None):
        if self.trained and i is not None:
            self.load_state_dict(self.best_param[i])
            self.fwd_dynamics = Dynamics(self.device, self.input_dim, self.hidden_dim, self.non_linearity, self.architecture, self.T, self.num_vals, self.seed_params) 
            self.flow = Semiflow(self.device, self.fwd_dynamics, self.adjoint, self.T, self.dt, self.method, self.seed_params)

        features = self.flow(x)
        traj = self.flow.trajectory(x, int(self.T / self.dt) + 1)
        if self.final_layer:
            pred = self.linear_layer(features)
            if self.output_dim == 1:
                pred = pred.squeeze()
                if return_features:
                    return features, pred, traj
                return pred, traj
            else:
                proj_traj = self.linear_layer(traj).squeeze()
                if return_features:
                    return features, pred, proj_traj
                return pred, proj_traj
        else:
            pred = features
            proj_traj = traj
            #    pred = self.non_linearity(pred)
            #    proj_traj = self.non_linearity(proj_traj)
            if return_features:
                return features, pred, traj
            return pred, traj
            
            
class adj_Dynamics(nn.Module):
    """
    Defines the adjoint dynamics for neural ODEs dot(x(t)) = f(u(t), x(t))
    They are given by dot(p(t)) = -D_xf(u(t), x(t)) * p using the forward pass dynamics and trajectory.
    """
    def __init__(self, dynamics, x_traj, non_linearity=nn.Tanh(), seed_params=1):
        super(adj_Dynamics, self).__init__()
        self.fwd_dynamics = dynamics
        self.x_traj = x_traj
        self.non_linearity = activations[f'{non_linearity}_prime']

        if self.non_linearity is None:
            raise ValueError(f"Derivative for activation function {non_linearity} not found.")
        
        torch.manual_seed(seed_params)
        torch.cuda.manual_seed_all(seed_params)

  
    def forward(self, t, p):
        num_vals = self.fwd_dynamics.num_vals
        delta_t = self.fwd_dynamics.T / num_vals
        k = int(t / delta_t)
        x = self.x_traj[num_vals - k - 1]
        
        # Compute the gradient application
        if self.fwd_dynamics.architecture < 1:
            # Accessing backward-in-time parameters
            w_t = self.fwd_dynamics.fc2_time[num_vals - k - 1].weight
            b_t = self.fwd_dynamics.fc2_time[num_vals - k - 1].bias
            if self.fwd_dynamics.architecture == -1:
                x = torch.matmul(x, w_t.t()) + b_t
                x_w = self.non_linearity(x)
                grad = torch.matmul(w_t.t(), torch.diag_embed(x_w))
            elif self.fwd_dynamics.architecture == 0:
                x_w = self.non_linearity(x)
                grad = torch.matmul(w_t.t(), torch.diag_embed(x_w))
            # Apply the gradient to the co-state vector p
            return -torch.matmul(grad, p.unsqueeze(-1)).squeeze()
        else:
            # Accessing backward-in-time parameters
            w_t = self.fwd_dynamics.fc3_time[num_vals - k - 1].weight
            a_t = self.fwd_dynamics.fc1_time[num_vals - k - 1].weight
            b_t = self.fwd_dynamics.fc1_time[num_vals - k - 1].bias
            
            x_w = torch.matmul(x,a_t.t()) + b_t
            x_w = self.non_linearity(x_w)
            x_w = torch.matmul(a_t.t(),torch.diag_embed(x_w))
            grad = torch.matmul(x_w ,w_t.t())
            return -torch.matmul(grad, p.unsqueeze(-1)).squeeze()

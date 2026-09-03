import torch
import torch.distributed as dist
from torch.optim.optimizer import Optimizer, required
from comm_helpers import communicate


class Scaffold(Optimizer):
    r"""Implements the Scaffold algorithm. Both local and server optimizers are SGD. """

    # The constructor accepts other parameters just to keep the interface consistent
    def __init__(self, params, ratio, gmf, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, variance=0, mu=0, slr=1.0, 
                 clients_per_round=-1, total_clients=-1):
        
        self.ratio = ratio
        self.itr = 0
        self.clients_per_round = clients_per_round
        self.total_clients = total_clients

        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if slr <= 0.0:
            raise ValueError("Invalid server learning rate: {}".format(slr))
        if self.clients_per_round <= 0:
            raise ValueError("Invalid clients per round: {}".format(self.clients_per_round))
        if self.total_clients <= 0:
            raise ValueError("Invalid total clients: {}".format(self.total_clients))
        if self.clients_per_round != self.total_clients:
            raise ValueError("Current implementation only supports full client participation")

        # Only required parameters from the constructor are used, rest are zeroed
        defaults = dict(lr=lr, momentum=0, dampening=0, slr=slr,
                        weight_decay=weight_decay, nesterov=False, variance=0)
        
        super(Scaffold, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Scaffold, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                
                param_state = self.state[p]
                if 'old_init' not in param_state:
                    param_state['old_init'] = torch.clone(p.data).detach()
                
                # initialize control variate to zero for scaffold client
                if 'c_i' not in param_state:
                    param_state['c_i'] = torch.zeros_like(d_p)
                
                # server's control variate (also initialized to zero)
                if 'c' not in param_state:
                    param_state['c'] = torch.zeros_like(d_p)

                # x^(t+1, k) = x^(t, k) - eta * (g^(t, k) + c_i^(t, k) - c^(t))
                p.data.add_(-group['lr'], d_p - param_state['c_i'] + param_state['c']) 

        self.itr += 1

        return loss

    def average(self):
        param_list = []
        control_variate_list = []
        
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if 'old_init' not in param_state:
                    continue
                with torch.no_grad():
                    # Synchronize server's control variate
                    c_i_plus = param_state['c_i'] - param_state['c'] + (param_state['old_init'] - p.data) / (self.itr * group['lr'])
                    param_state['c'].mul_(1/self.clients_per_round)                    
                    param_state['c'].add_(1/self.total_clients, c_i_plus - param_state['c_i'])
                    
                    # Update client's control variate
                    param_state['c_i'] = torch.clone(c_i_plus).detach()

                    # Syncrhonize model parameters                    
                    p.data.sub_(param_state['old_init']) # delta^(t) = x^(t, tau) - x^(t)
                    p.data.mul_(self.ratio) # delta^(t) * n_i/n
                
                param_list.append(p)
                control_variate_list.append(param_state['c'])
                

        # Only deltas get averaged
        communicate(param_list, dist.all_reduce)
        communicate(control_variate_list, dist.all_reduce)

        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if 'old_init' not in param_state:
                    continue
                
                # update model parameters
                p.data.mul_(group['slr']) # eta * delta^(t)_agg
                p.data.add_(param_state['old_init']) # x^(t+1) = eta * delta^(t)_agg + x^(t)
                param_state['old_init'] = torch.clone(p.data).detach() # Store x^(t+1)

        self.itr = 0

    def set_ratio(self, ratio):
        self.ratio = ratio
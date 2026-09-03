import torch
import torch.distributed as dist
from torch.optim.optimizer import Optimizer, required
from comm_helpers import communicate


class FedAdam(Optimizer):
    r"""Implements the FedAdam algorithm where local optimizer is SGD and server 
        optimizer is Adam.
     """

    # The constructor accepts other parameters just to keep the interface consistent
    def __init__(self, params, ratio, gmf, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, variance=0, mu=0, slr=1.0,
                 clients_per_round=-1, total_clients=-1):
        
        self.ratio = ratio
        self.beta1 = gmf # global momentum factor as beta1
        self.beta2 = 0.99
        self.adaptivity = 1e-3

        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if slr <= 0.0:
            raise ValueError("Invalid server learning rate: {}".format(slr))
        if gmf <= 0.0:
            raise ValueError("Invalid global momentum factor: {}".format(gmf))
        
        # Only required parameters from the constructor are used, rest are zeroed
        # Note: FedAdam is defined without momentum
        defaults = dict(lr=lr, momentum=0, dampening=0, slr=slr,
                        weight_decay=weight_decay, nesterov=False, variance=0)
        
        super(FedAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(FedAdam, self).__setstate__(state)
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

                p.data.add_(-group['lr'], d_p)

        return loss

    def average(self):
        param_list = []
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if 'old_init' not in param_state:
                    continue
                with torch.no_grad():
                    p.data.sub_(param_state['old_init']) # delta^(t) = x^(t, tau) - x^(t)
                    p.data.mul_(self.ratio) # delta^(t) * n_i/n
                param_list.append(p)

        # Only deltas get averaged
        communicate(param_list, dist.all_reduce)

        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if 'old_init' not in param_state:
                    continue
                if 'global_momentum_buffer' not in param_state:
                    param_state['global_momentum_buffer'] = torch.clone(p.data).detach() # delta^(t)_agg  
                    param_state['global_momentum_buffer'].mul_(1-self.beta1) # m^(t) = (1 - beta1) * delta^(t)_agg
                else:
                    param_state['global_momentum_buffer'].mul_(self.beta1) # m^(t) = beta1 * m^(t-1)
                    param_state['global_momentum_buffer'].add_(1-self.beta1, p.data) # m^(t) += (1-beta1) * delta^(t)_agg
                
                if 'global_second_moment_buffer' not in param_state:
                    param_state['global_second_moment_buffer'] = torch.clone(p.data**2).detach() # delta^(t)_agg^2  
                    param_state['global_second_moment_buffer'].mul_(1-self.beta2) # v^(t) = (1 - beta2) * delta^(t)_agg^2
                else:
                    param_state['global_second_moment_buffer'].mul_(self.beta2) # v^(t) = beta2 * v^(t-1)
                    param_state['global_second_moment_buffer'].add_(1-self.beta2, p.data**2) # v^(t) += (1-beta2) * delta^(t)_agg^2

                p.data.copy_(param_state['old_init']) # x^(t+1) = x^(t)
                p.data.add_(group['slr'], param_state['global_momentum_buffer']/(torch.sqrt(param_state['global_second_moment_buffer']) + self.adaptivity)) # x^(t+1) += eta * m^(t)/(sqrt[v^(t)] + epsilon)
                
                param_state['old_init'] = torch.clone(p.data).detach() # Store x^(t+1)

    def set_ratio(self, ratio):
        self.ratio = ratio
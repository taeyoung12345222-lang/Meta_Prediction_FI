import torch
import torch.distributed as dist
from torch.optim.optimizer import Optimizer, required
from comm_helpers import communicate


class FedAvg(Optimizer):
    r"""Implements the FedAvg algorithm. Both local and server optimizers are SGD. """

    # The constructor accepts other parameters just to keep the interface consistent
    def __init__(self, params, ratio, gmf, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, variance=0, mu=0, slr=1.0,
                 clients_per_round=-1, total_clients=-1):
        
        self.ratio = ratio

        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if slr <= 0.0:
            raise ValueError("Invalid server learning rate: {}".format(slr))

        # Only required parameters from the constructor are used, rest are zeroed
        defaults = dict(lr=lr, momentum=0, dampening=0, slr=slr,
                        weight_decay=weight_decay, nesterov=False, variance=0)
        
        super(FedAvg, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(FedAvg, self).__setstate__(state)
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
                p.data.mul_(group['slr']) # eta * delta^(t)_agg
                p.data.add_(param_state['old_init']) # x^(t+1) = eta * delta^(t)_agg + x^(t)
                param_state['old_init'] = torch.clone(p.data).detach() # Store x^(t+1)

    def set_ratio(self, ratio):
        self.ratio = ratio
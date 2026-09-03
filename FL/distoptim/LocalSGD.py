import torch
from torch.optim.optimizer import Optimizer, required

class LocalSGD(Optimizer):
    r"""SGD algorithm for local training only. This is not Federated. """

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
        
        super(LocalSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(LocalSGD, self).__setstate__(state)
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
        """
            Since this is not federated, we just return.
        """        
        return

    def set_ratio(self, ratio):
        self.ratio = ratio
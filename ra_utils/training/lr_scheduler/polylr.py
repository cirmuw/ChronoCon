from torch.optim.lr_scheduler import _LRScheduler


class PolyLRScheduler(_LRScheduler):
    def __init__(self, optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9, current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr


class PolyLRSchedulerMultiLR(_LRScheduler):
    """
    Same as PolyLRScheduler, but gets initial learning rate from optimizer and allows for different 
    learning rates for each parameter group.
    
    *Note:* Likely there is also something like this in torch.optim, but I did not look. 
    """
    def __init__(self, optimizer, max_steps: int, exponent: float = 0.9, current_step: int = None):
        self.optimizer = optimizer
        self.initial_lrs = [param_group['lr'] for param_group in optimizer.param_groups]
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        for i, param_group in enumerate(self.optimizer.param_groups):
            new_lr = self.initial_lrs[i] * (1 - current_step / self.max_steps) ** self.exponent            
            param_group['lr'] = new_lr
import torch

class LinearWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, target_lr, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.target_lr = target_lr
        self.initial_lrs = [group['lr'] for group in optimizer.param_groups]
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch > self.warmup_epochs:
            return [group['lr'] for group in self.optimizer.param_groups]
        warmup_factor = self.last_epoch / self.warmup_epochs
        return [lr + warmup_factor * (self.target_lr - lr) for lr in self.initial_lrs]

class CompositeScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, schedulers, total_epochs, last_epoch=-1):
        self.schedulers = schedulers
        self.total_epochs = total_epochs
        self.epochs_per_scheduler = total_epochs // len(schedulers)
        super().__init__(schedulers[0].optimizer, last_epoch)

    def get_lr(self):
        current_scheduler = self.schedulers[self.last_epoch // self.epochs_per_scheduler]
        return current_scheduler.get_lr()

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        current_scheduler = self.schedulers[self.last_epoch // self.epochs_per_scheduler]
        current_scheduler.step(epoch - self.epochs_per_scheduler * (self.last_epoch // self.epochs_per_scheduler))
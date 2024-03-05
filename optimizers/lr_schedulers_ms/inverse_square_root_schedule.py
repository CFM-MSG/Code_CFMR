from mindspore.experimental import optim
# import mindspore.nn as nn

class InverseSquareRootSchedule(optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, args, last_epoch=-1):
        super(InverseSquareRootSchedule, self).__init__(optimizer, last_epoch)
        # print('base_lr', self.base_lrs)
        warmup_end_lr = self.base_lrs[0]
        if args['warmup_init_lr'] < 0:
            warmup_init_lr = warmup_end_lr
        self.warmup_init_lr = args['warmup_init_lr']
        self.warmup_updates = args['warmup_updates']

        # linearly warmup for the first args.warmup_updates
        self.lr_step = (warmup_end_lr - self.warmup_init_lr) / self.warmup_updates

        # then, decay prop. to the inverse square root of the update number
        self.decay_factor = warmup_end_lr * self.warmup_updates**0.5
        # initial learning rate
        self.lr = self.warmup_init_lr
        # self.optimizer.set_lr(self.lr)

    def get_lr(self):
        # print(self.last_epoch)
        lrs = [lr.value() for lr in self._last_lr]
        # print(lrs)
        if self.last_epoch == 0 or self.last_epoch % self.step_size != 0:
            return lrs
        print("+++", [lr * self.gamma for lr in lrs])
        return [lr * self.gamma for lr in lrs]

    def _get_closed_form_lr(self):
        # print('base_lr', self.base_lrs)
        if self.last_epoch < self.warmup_updates:
            self.lr = [self.warmup_init_lr + self.last_epoch*self.lr_step]
        else:
            self.lr = [self.decay_factor * self.last_epoch**-0.5]
        # self.optimizer.set_lr(self.lr)
        return self.lr
        # return [base_lr for base_lr in self.base_lrs]
        # return [base_lr * self.gamma ** (self.last_epoch // self.step_size)
        #         for base_lr in self.base_lrs]

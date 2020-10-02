# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from . import FairseqLRScheduler, register_lr_scheduler


@register_lr_scheduler('my_inverse_sqrt')
class InverseSquareRootSchedule(FairseqLRScheduler):
    """Decay the LR based on the inverse square root of the update number.

    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (``--warmup-init-lr``) until the configured
    learning rate (``--lr``). Thereafter we decay proportional to the number of
    updates, with a decay factor set to align with the configured learning rate.

    During warmup::

      lrs = torch.linspace(args.warmup_init_lr, args.lr, args.warmup_updates)
      lr = lrs[update_num]

    After warmup::

      decay_factor = args.lr * sqrt(args.warmup_updates)
      lr = decay_factor / sqrt(update_num)
    """

    def __init__(self, args, optimizer):
        super().__init__(args, optimizer)
        if len(args.lr) > 1:
            raise ValueError(
                'Cannot use a fixed learning rate schedule with inverse_sqrt.'
                ' Consider --lr-scheduler=fixed instead.'
            )
        self.warmup_end_lr = args.lr[0]

        # initial learning rate, follow the equation (attnetion is all your need)
        self.warmup_steps = float(args.warmup_updates)

        self.multiplier = 10 * (args.encoder_embed_dim ** -0.5)
        num_updates = 0
        self.decay_factor = self.multiplier * min((num_updates + 1) * (self.warmup_steps ** -1.5),
                                                      (num_updates + 1) ** -0.5)
        self.lr = self.decay_factor * self.warmup_end_lr

        self.optimizer.set_lr(self.lr)
        print("initial lr: {}".format(self.optimizer.get_lr()))
    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        # fmt: off
        parser.add_argument('--warmup-updates', default=4000, type=int, metavar='N',
                            help='warmup the learning rate linearly for the first N updates')
        # fmt: on

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        # we don't change the learning rate at epoch boundaries
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""

        self.decay_factor = self.multiplier * min((num_updates + 1) * (self.warmup_steps ** -1.5),
                                                  (num_updates + 1) ** -0.5)

        self.lr = self.decay_factor * self.warmup_end_lr

        self.optimizer.set_lr(self.lr)
        return self.lr

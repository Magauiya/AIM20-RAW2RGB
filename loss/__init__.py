from importlib import import_module

import torch.nn as nn

'''
Losses:
- L1 
- VGG16  perceptual loss (source: https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49)
- MSE

not implemented:
- VGG19

'''


class Loss(nn.modules.loss._Loss):
    def __init__(self, args, device):
        super(Loss, self).__init__()
        print('[*] Loss function:')

        self.loss = []
        self.loss_module = nn.ModuleList()
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type == 'VGG16':  # .find('VGG') >= 0:
                module = import_module('loss.VGG16')
                loss_function = getattr(module, 'VGG16')(resize=True)
            elif loss_type == 'ColorLoss':
                module = import_module('loss.ColorLoss')
                loss_function = getattr(module, 'ColorLoss')()

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        self.loss_module.to(device)

    def forward(self, source, target):
        losses = []
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                loss = l['function'](source, target)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                # self.log[-1, i] += effective_loss.item()
            # elif l['type'] == 'DIS':
            # self.log[-1, i] += self.loss[i - 1]['function'].loss

        loss_sum = sum(losses)

        return loss_sum

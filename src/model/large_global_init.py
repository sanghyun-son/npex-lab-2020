import torch
from torch import nn
from torch.nn import init


class Net(nn.Module):
    '''
    A larger image2image CNN.
    '''

    def __init__(self):
        # This line is very important!
        super().__init__()
        m = []
        m.append(nn.Conv2d(3, 128, 3, padding=1))
        m.append(nn.ReLU(inplace=True))

        for _ in range(20):
            m.append(nn.Conv2d(128, 128, 3, padding=1))
            m.append(nn.ReLU(inplace=True))

        m.append(nn.Conv2d(128, 3, 3, padding=1))
        self.seq = nn.Sequential(*m)

        # For initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                m.bias.data.fill_(0)

        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        LargerNet forward function.

        Args:
            x (B x 3 x H x W Tensor): An input image

        Return:
            (B x 3 x H x W Tensor): An output image
        '''
        # Autograd will keep the history.
        x = x + self.seq(x)
        return x

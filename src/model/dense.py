import torch
from torch import nn


class Net(nn.Module):
    '''
    A simple image2image CNN.
    '''

    def __init__(self):
        # This line is very important!
        super().__init__()
        k = 51

        self.conv1 = nn.Conv2d(3, k, 3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(3 + k, k, 3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(3 + 2 * k, k, 3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(3 + 3 * k, 3, 3, padding=1)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        SimpleNet forward function.

        Args:
            x (B x 3 x H x W Tensor): An input image

        Return:
            (B x 3 x H x W Tensor): An output image
        '''
        # Autograd will keep the history.
        d1 = self.conv1(x)
        d1 = self.relu1(d1)

        d2_input = torch.cat((x, d1), dim=1)
        d2 = self.conv2(d2_input)
        d2 = self.relu2(d2)

        d3_input = torch.cat((d2_input, d2), dim=1)
        d3 = self.conv3(d3_input)
        d3 = self.relu3(d3)

        d4_input = torch.cat((d3_input, d3), dim=1)
        out = self.conv4(d4_input)
        return out

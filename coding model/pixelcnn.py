import torch
import torch.nn as nn
import torch.nn.functional as f

class pixelcnn(nn.Module):

    def __init__(self, In_Channel, hidden_channel,window_size=3):
        super(pixelcnn, self).__init__()
        self.window_size = window_size
        self.channels = In_Channel
        self.conv = nn.Sequential(
                MaskedConv2d('A',In_Channel, hidden_channel, k_size=2*window_size+1, stride=1, pad=window_size),
                nn.PReLU(),
                MaskedConv2d('B',hidden_channel, hidden_channel, k_size=2*window_size+1, stride=1, pad=window_size),
                nn.PReLU(),
                MaskedConv2d('B',hidden_channel, hidden_channel, k_size=2*window_size+1, stride=1, pad=window_size),
                nn.PReLU(),
                MaskedConv2d('B',hidden_channel, hidden_channel, k_size=2*window_size+1, stride=1, pad=window_size),
                nn.PReLU(),
                MaskedConv2d('B',hidden_channel, 1, k_size=2*window_size+1, stride=1, pad=window_size)
                )

    def forward(self,x,softmax=False):
        if softmax:
            pred = self.conv(x)
            return f.softmax(pred)
        else:
            pred = self.conv(x)
            return f.sigmoid(pred)

class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, c_in, c_out, k_size, stride, pad, bias=True):
        """2D Convolution with masked weight for Autoregressive connection"""

        super(MaskedConv2d, self).__init__(
            c_in, c_out, k_size, stride, pad, bias)
        assert mask_type in ['A', 'B']
        self.mask_type = mask_type
        ch_out, ch_in, height, width = self.weight.size()

        # Mask
        #         -------------------------------------
        #        |  1       1       1       1       1 |
        #        |  1       1       1       1       1 |
        #        |  1       1    1 if B     0       0 |   H // 2
        #        |  0       0       0       0       0 |   H // 2 + 1
        #        |  0       0       0       0       0 |
        #         -------------------------------------
        #  index    0       1     W//2    W//2+1

        mask = torch.ones(ch_out, ch_in, height, width)
        if mask_type == 'A':
            # First Convolution Only
            # => Restricting connections to
            #    already predicted neighborhing channels in current pixel
            mask[:, ch_in-1, height // 2, width // 2:] = 0
            mask[:, ch_in-1, height // 2 + 1:] = 0
        else:
            mask[:, :, height // 2, width // 2 + 1:] = 0
            mask[:, :, height // 2 + 1:] = 0
        self.register_buffer('mask', mask)

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)

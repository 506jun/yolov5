# TODO: ghostnetv2
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
class Bottleneck(nn.Module):
    """A bottleneck layer with optional shortcut and group convolution for efficient feature extraction."""

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        """Initializes a standard bottleneck layer with optional shortcut and group convolution, supporting channel
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Processes input through two convolutions, optionally adds shortcut if channel dimensions match; input is a
        tensor.
        """
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
class C3(nn.Module):
    """Implements a CSP Bottleneck module with three convolutions for enhanced feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes C3 module with options for channel count, bottleneck repetition, shortcut usage, group
        convolutions, and expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Performs forward propagation using concatenated outputs from two convolutions and a Bottleneck sequence."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
def autopad(k, p=None, d=1):
    """
    Pads kernel to 'same' output shape, adjusting for optional dilation; returns padding size.

    `k`: kernel, `p`: padding, `d`: dilation.
    """
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
class Conv(nn.Module):
    """Applies a convolution, batch normalization, and activation function to an input tensor in a neural network."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initializes a standard convolution layer with optional batch normalization and activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies a convolution followed by batch normalization and an activation function to the input tensor `x`."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Applies a fused convolution and activation function to the input tensor `x`."""
        return self.act(self.conv(x))
class DWConv(Conv):
    """Implements a depth-wise convolution layer with optional activation for efficient spatial filtering."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """Initializes a depth-wise convolution layer with optional activation; args: input channels (c1), output
        channels (c2), kernel size (k), stride (s), dilation (d), and activation flag (act).
        """
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)
class GhostConvV2(nn.Module):
    # Ghostv2 Convolution https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/ghostnetv2_pytorch
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True, mode=None):  # ch_in, ch_out, kernel, stride, groups
        super(GhostConvV2, self).__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)
        self.mode = mode
        self.gate_fn = nn.Sigmoid()
        if mode in ['attn']:
             self.short_conv = nn.Sequential(
                nn.Conv2d(c1, c2, k, s, k // 2, bias=False),
                nn.BatchNorm2d(c2),
                nn.Conv2d(c2, c2, kernel_size=(1, 5), stride=1, padding=(0, 2), groups=c2, bias=False),
                nn.BatchNorm2d(c2),
                nn.Conv2d(c2, c2, kernel_size=(5, 1), stride=1, padding=(2, 0), groups=c2, bias=False),
                nn.BatchNorm2d(c2)
                 )

    def forward(self, x):
        y = self.cv1(x)
        if self.mode in ['attn']:
            res = self.short_conv(F.avg_pool2d(x, kernel_size=2, stride=2))
            # res=self.short_conv(x)
            out = torch.cat((y, self.cv2(y)), 1)
            return out * F.interpolate(self.gate_fn(res), size=(out.shape[-2], out.shape[-1]),mode='nearest')
        return torch.cat((y, self.cv2(y)), 1)
class GhostBottleneckV2(nn.Module):
    # Ghostv2 Convolution https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/ghostnetv2_pytorch
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConvV2(c1, c_, 1, 1, mode='attn'),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConvV2(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)
class C3GhostV2(C3):
    # C3 module with Ghostv2Bottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneckV2(c_, c_) for _ in range(n)))
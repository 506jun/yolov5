import torch
import torch.nn as nn

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)
    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        # 替换自适应池化为固定池化 + 平均操作
        self.pool_h = nn.AvgPool2d(kernel_size=1, stride=1)  # 占位符，实际使用平均操作
        self.pool_w = nn.AvgPool2d(kernel_size=1, stride=1)  # 占位符，实际使用平均操作
        
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        
        # 使用平均操作代替自适应池化
        x_h = x.mean(dim=3, keepdim=True)  # 替代 self.pool_h(x)
        x_w = x.mean(dim=2, keepdim=True)  # 替代 self.pool_w(x)
        
        # 调整维度以匹配原始实现
        x_w = x_w.permute(0, 1, 3, 2)
        
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        # 拆分并调整维度
        split_size = [h, w]
        x_h, x_w = torch.split(y, split_size, dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        
        out = identity * a_w * a_h
        return out
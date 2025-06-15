    
    # current plan
# Input Image
# │
# ├── Sliding Window Extraction (e.g., 64×64 or 128×128 windows)
# │     └── For each window:
# │
# │     ├── CRL Block (3 pointwise convs + channel attention)       ← emphasize color structure
# │     ├── Conv Block 1                                            ← low-level feature extraction
# │     │    ├── Conv2d → BN → GELU
# │     │    ├── Conv2d → BN → GELU
# │     │    ├── Conv2d → BN → GELU
# │     ├── ResNet Block 1 BasicBlock               ← residual refinement
# │     ├── Conv Block 2
# │     │    ├── Conv2d → BN → GELU
# │     │    ├── Conv2d → BN → GELU
# │     │    ├── Conv2d → BN → GELU
# │     ├──  Channel Attention
# │     ├── ResNet Block 2 bottleneck
# │     ├── Conv Block 3
# │     │    ├── Conv2d → BN → GELU
# │     ├── ResNet Block 3 bottleneck
# │     ├── Global Average Pool                                     ← reduce to vector
# │     └── Linear Layer → Suspiciousness Score (1D logit or scalar)
# │
# ├── Collect all window scores (e.g., list of N scores)
# │
# ├── Aggregation Layer (e.g., mean, max, learned attention)        ← fuse window evidence
# │
# └── Final Decision (threshold or classifier)                      ← real vs fake image

# block1 = ResidualBlock(inplanes=64, planes=64, block_type="basic")  # after Conv Block 1
# block2 = ResidualBlock(inplanes=128, planes=64, stride=2, downsample=..., block_type="bottleneck")  # after Conv Block 2
# block3 = ResidualBlock(inplanes=256, planes=64, stride=1, block_type="bottleneck")  # after Conv Block 3

import sys, torch
import torch.nn as nn
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
from pytorch_wavelets import DWTForward
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, block_type="basic", expansion=4):
        super().__init__()
        self.block_type = block_type
        self.expansion = expansion if block_type == "bottleneck" else 1
        self.downsample = downsample
        self.act = nn.GELU()

        if block_type == "basic":
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)

        elif block_type == "bottleneck":
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = nn.Conv2d(planes, planes * expansion, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(planes * expansion)

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))

        if self.block_type == "bottleneck":
            out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.act(out + identity)
        return out

def conv_block(in_c, out_c, k=3, stride=1, pad=1):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=k, stride=stride, padding=pad),
        nn.BatchNorm2d(out_c),
        nn.GELU()
    )

def channel_attention(x, shared_mlp):
    avg = F.adaptive_avg_pool2d(x, 1)
    max_ = F.adaptive_max_pool2d(x, 1)
    scale = torch.sigmoid(shared_mlp(avg) + shared_mlp(max_))
    return x * scale

def crl_block(x, weights, biases, norms, shared_mlp):
    x = F.gelu(F.conv2d(x, weights[0], biases[0]))
    x = norms[0](x)
    x = F.gelu(F.conv2d(x, weights[1], biases[1]))
    x = norms[1](x)
    x = F.gelu(F.conv2d(x, weights[2], biases[2]))
    return channel_attention(x, shared_mlp)

def make_resblock(inplanes, planes, block_type="basic"):
    expansion = 4 if block_type == "bottleneck" else 1
    downsample = None
    if inplanes != planes * expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * expansion)
        )
    return ResidualBlock(inplanes, planes, block_type=block_type, downsample=downsample)

class LLnet(nn.Module):
    def __init__(self, wave: str = "haar", window_size: int = 64):
        super().__init__()
        self.dwt = DWTForward(J=1, wave=wave, mode="symmetric")
        self.window_size = window_size

        self.crl_weight1 = nn.Parameter(torch.randn((16, 3, 1, 1)))
        self.crl_bias1 = nn.Parameter(torch.randn((16,)))
        self.crl_weight2 = nn.Parameter(torch.randn((32, 16, 1, 1)))
        self.crl_bias2 = nn.Parameter(torch.randn((32,)))
        self.crl_weight3 = nn.Parameter(torch.randn((3, 32, 1, 1)))
        self.crl_bias3 = nn.Parameter(torch.randn((3,)))

        self.norm1 = nn.BatchNorm2d(16)
        self.norm2 = nn.BatchNorm2d(32)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(3, 3, 1), nn.ReLU(), nn.Conv2d(3, 3, 1)
        )

        self.conv1 = conv_block(3, 32)
        self.conv2 = conv_block(32, 64)
        self.conv3 = conv_block(256, 128, k=1, pad=0)

        self.res1 = ResidualBlock(32, 32)
        self.res2 = make_resblock(64, 64, block_type="bottleneck")
        self.res3 = make_resblock(128, 128, block_type="bottleneck")

        self.shared_attn_mlp = nn.Sequential(
            nn.Conv2d(64, 64, 1), nn.GELU(), nn.Conv2d(64, 64, 1)
        )

        self.final_fc = nn.Linear(512, 1)

    # def wavelet_split(self, x):
    #     yl, yh_list = self.dwt(x)
        
    #     bands = [yl]
        
    #     for band in yh_list:
    #         for i in range(3):
    #             bands.append(band[:,:, i]) # i to get eahc of the 3 bands..?
    #     return bands
    def wavelet_split(self, x):
        yl, yh_list = self.dwt(x)
        bands = [yl]  # start with LL
        
        for level_idx, band in enumerate(yh_list):
            for i, name in zip(range(3), ['LH', 'HL', 'HH']):
                single_band = band[:, :, i, :, :]  # (B, C, H, W)
                bands.append(single_band)
        
        return bands

    def extract_windows(self, x):
        B, C, H, W = x.shape
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        windows = x.unfold(2, self.window_size, self.window_size).unfold(3, self.window_size, self.window_size)
        B, C, nh, nw, h, w = windows.shape
        return windows.permute(0, 2, 3, 1, 4, 5).reshape(-1, C, h, w), nh, nw
    
    def forward(self, x, return_all=False):
        bands = self.wavelet_split(x)
        # x = bands[0]
        per_band_scores = []
        bands = [bands[0]] # Currently only use LL band, low freq band.. todo try using all bands :))
        
        for band in bands:
        
            windows, nh, nw = self.extract_windows(band)

            out = crl_block(windows, [self.crl_weight1, self.crl_weight2, self.crl_weight3],
                            [self.crl_bias1, self.crl_bias2, self.crl_bias3],
                            [self.norm1, self.norm2], self.shared_mlp)

            out = self.conv1(out)
            out = self.res1(out)
            out = self.conv2(out)

            out = channel_attention(out, self.shared_attn_mlp)

            out = self.res2(out)
            out = self.conv3(out)
            out = self.res3(out)
            out = F.adaptive_avg_pool2d(out, 1).squeeze(-1).squeeze(-1)
            out = self.final_fc(out)

            out = out.view(band.size(0), -1)
            if return_all:
                # return out.view(x.size(0), nh, nw)
                per_band_scores.append(out.view(band.size(0), nh, nw))  # [B, nh, nw]
            # return out.mean(dim=1)
            else:
                per_band_scores.append(out.mean(dim=1, keepdim=True))   # [B, 1]
            # return out.mean(dim=1, keepdim=True)  # shape: [B, 1]
        #out of loop?
        if return_all:
            return torch.stack(per_band_scores, dim=1)  # shape: [B, num_bands, nh, nw]
        else:
            return torch.stack(per_band_scores, dim=1).mean(dim=1, keepdim=True)  # [B, 1]
    
  
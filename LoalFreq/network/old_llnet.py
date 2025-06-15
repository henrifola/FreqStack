#my oerigina lcode
# import sys, torch
# import torch.nn as nn
# import torchvision.transforms as T
# import matplotlib.pyplot as plt
# from PIL import Image
# from pytorch_wavelets import DWTForward
# import torch.nn.functional as F


# import torch.nn as nn

# class ResidualBlock(nn.Module):
#     def __init__(self, inplanes, planes, stride=1, downsample=None, block_type="basic", expansion=4):
#         super().__init__()
#         self.block_type = block_type
#         self.expansion = expansion if block_type == "bottleneck" else 1
#         self.downsample = downsample
#         self.relu = nn.ReLU(inplace=True)

#         if block_type == "basic":
#             self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#             self.bn1 = nn.BatchNorm2d(planes)
#             self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#             self.bn2 = nn.BatchNorm2d(planes)

#         elif block_type == "bottleneck":
#             self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#             self.bn1 = nn.BatchNorm2d(planes)
#             self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#             self.bn2 = nn.BatchNorm2d(planes)
#             self.conv3 = nn.Conv2d(planes, planes * expansion, kernel_size=1, bias=False)
#             self.bn3 = nn.BatchNorm2d(planes * expansion)

#     def forward(self, x):
#         identity = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         if self.block_type == "bottleneck":
#             out = self.conv3(out)
#             out = self.bn3(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out += identity
#         out = self.relu(out)
#         return out


# class LLnet(nn.Module):
    
#     def __init__(self, wave: str = "haar"):
#         super(LLnet, self).__init__()
#         self.dwt = DWTForward(J=1, wave=wave, mode="symmetric")  # 1-level DWT
        
#         self.weight1= nn.Parameter(torch.randn((64,3,1,1)).cuda())
#         self.bias1 = nn.Parameter(torch.randn((64,)).cuda())
        
#         self.resent1 = self._make_layer(ResidualBlock, planes=64, blocks=1, block_type="basic")
#         self.resnet2 = self._make_layer(ResidualBlock, planes=64, blocks=2, stride=2, block_type="bottleneck")
#         self.resnet3 = self._make_layer(ResidualBlock, planes=64, blocks=1, block_type="bottleneck")
        
#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = (
#             nn.Sequential(
#                 conv1x1(self.inplanes, planes * block.expansion, stride),
#                 nn.BatchNorm2d(planes * block.expansion)
#             )
#             if stride != 1 or self.inplanes != planes * block.expansion
#             else None
#         )

#         layers = [block(self.inplanes, planes, stride, downsample)]
#         self.inplanes = planes * block.expansion

#         layers += [block(self.inplanes, planes) for _ in range(1, blocks)]
#         return nn.Sequential(*layers)

        
#     def wavelet_split(self, x):
#         # x = : (B, 3 , H, W)
#         yl, yh_list = self.dwt(x)
#         yh = yh_list[0] 
        
#         rgbLL = yl # shape (B, 3, H/2, W/2)
#         rgbLH = yh[:, :, 0, ...] 
#         rgbHL = yh[:, :, 1, ...]
#         rgbHH = yh[:, :, 2, ...]
        
#         #Plan, window each band
#         #Do regular convs for each window, followed by cannel wise attention, conv
        
#         #Vs, ViT and channel attention for each window band?
        
        
#         return rgbLL, rgbLH, rgbHL, rgbHH
    
#     def extract_windows(self, x, window_size, stride):
#         B, C, H, W = x.shape

#         #padding sizes
#         pad_bottom = (window_size - H % window_size) % window_size
#         pad_right = (window_size - W % window_size) % window_size

#        # padding
#         img_padded = F.pad(x, (0, pad_right, 0, pad_bottom), mode='reflect')
#         padded_H, padded_W = img_padded.shape[-2:]

       
#         windows = img_padded.unfold(2, window_size, window_size).unfold(3, window_size, window_size)
#         B_, C_, nH, nW, wH, wW = windows.shape

#         windows = windows.permute(0, 2, 3, 1, 4, 5).reshape(-1, C, wH, wW)

#         return windows, (padded_H, padded_W)
    
#     def channel_attention(self, x):
#         scale_avg = F.adaptive_avg_pool2d(x, 1)
#         scale_max = F.adaptive_max_pool2d(x, 1)

#         scale = torch.sigmoid(self.shared_mlp(scale_avg) + self.shared_mlp(scale_max))
#         x = x * scale # using the channel attention, Applies attention per channel
    
#     def crl(self, x):
#         x = F.gelu(F.conv2d(x, self.crl_weight1, self.crl_bias1, stride=1, padding=0))  # 3 → 16
#         x = self.norm1(x)
#         x = F.gelu(F.conv2d(x, self.crl_weight2, self.crl_bias2, stride=1, padding=0))  # 16 → 32
#         x = self.norm2(x)
#         x = F.gelu(F.conv2d(x, self.crl_weight3, self.crl_bias3, stride=1, padding=0))  # 32 → 3

#         x = self.channel_attention(x)
    
#         return x  # [B, 3, H, W]
    
#     def forward_pipeline(self, x):
#         # Or how about resnetlayer here
#         x = F.gelu(self.conv1(x))
#         x = F.gelu(self.conv2(x))
        
    
#     def forward(self, x):
#         rgbLL, rgbLH, rgbHL, rgbHH = self.wavelet_split(x)
        
#         windows, padded_size = self.extract_windows(x, 2, 2)
#         # for each window, do a cnn, 
        
        
        
#         return 2   


# if __name__ == "__main__":
#     model = LLnet()
#     dummy_input = torch.randn(1, 3, 64, 64)   
#     model.forward(dummy_input)  
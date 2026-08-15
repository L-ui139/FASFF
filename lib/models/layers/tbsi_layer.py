from functools import partial
from turtle import forward

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.layers.attn_blocks import CASTBlock
from math import sqrt
from einops import rearrange
from lib.models.layers.DCT import DCT8x8, DCT7x7, DCT3x3
from timm.models.layers import trunc_normal_
import math

import numpy as np

ch = 64
n_blocks = 8


class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FourierUnit, self).__init__()
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2 + 2, out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        batch = x.shape[0]
        fft_dim = (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm='ortho')
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        height, width = ffted.shape[-2:]
        coords_vert = torch.linspace(0, 1, height)[None, None, :, None].expand(batch, 1, height, width).to(ffted)
        coords_hor = torch.linspace(0, 1, width)[None, None, None, :].expand(batch, 1, height, width).to(ffted)
        ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)
        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])
        ifft_shape_slice = x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm='ortho')
        return output


class SpectralTransform(nn.Module):
    def __init__(self, channels):
        super(SpectralTransform, self).__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.fu = FourierUnit(channels, channels)  # 传递通道数
        self.conv2 = nn.Conv2d(channels * 2, channels, 3, 1, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.fu(x1)
        x = self.conv2(torch.cat([x, x2], dim=1))
        return x


class FFC(nn.Module):
    def __init__(self, in_channels):
        super(FFC, self).__init__()
        self.in_channels = in_channels
        mid_channels = in_channels // 2

        self.convl2l = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.convl2g = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.convg2l = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.convg2g = SpectralTransform(mid_channels)

    def forward(self, x):
        if isinstance(x, tuple):
            # 如果输入是元组 (x_l, x_g)
            x_l, x_g = x
        else:
            # 如果输入是单个张量，将其拆分为两部分
            # 沿着通道维度拆分为两部分
            x_l = x[:, :self.in_channels // 2, :, :]
            x_g = x[:, self.in_channels // 2:, :, :]

        out_xl = self.convl2l(x_l) + self.convg2l(x_g)
        out_xg = self.convl2g(x_l) + self.convg2g(x_g)

        return out_xl, out_xg


# class SFIB(nn.Module):
#     def __init__(self):
#         super(SFIB, self).__init__()
#         self.ffc = FFC()
#         self.bn_l = nn.BatchNorm2d(ch // 2)
#         self.bn_g = nn.BatchNorm2d(ch // 2)
#         self.act_l = nn.ReLU(inplace=True)
#         self.act_g = nn.ReLU(inplace=True)

#     def forward(self, x):
#         x_l, x_g = self.ffc(x)
#         x_l = self.act_l(self.bn_l(x_l))
#         x_g = self.act_g(self.bn_g(x_g))
#         return x_l, x_g
class SFIB(nn.Module):
    def __init__(self, in_channels):
        super(SFIB, self).__init__()
        self.ffc = FFC(in_channels)
        self.bn_l = nn.BatchNorm2d(in_channels // 2)
        self.bn_g = nn.BatchNorm2d(in_channels // 2)
        self.act_l = nn.ReLU(inplace=True)
        self.act_g = nn.ReLU(inplace=True)

    def forward(self, x):
        # 确保输入是张量或元组
        if not (isinstance(x, tuple) or torch.is_tensor(x)):
            raise TypeError(f"Unsupported input type for FFC: {type(x)}")

        # 获取FFC输出
        x_l, x_g = self.ffc(x)

        # 确保输出是张量
        if not torch.is_tensor(x_l):
            x_l = x_l[0] if isinstance(x_l, tuple) else x_l
        if not torch.is_tensor(x_g):
            x_g = x_g[0] if isinstance(x_g, tuple) else x_g

        # 处理张量
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))

        # 返回拼接后的张量
        return torch.cat([x_l, x_g], dim=1)


class ResnetBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SFIB()
        self.conv2 = SFIB()

    def forward(self, x):
        x_l, x_g = torch.split(x, (ch // 2, ch // 2), dim=1)
        id_l, id_g = x_l, x_g
        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))
        x_l, x_g = id_l + x_l, id_g + x_g
        out = torch.cat((x_l, x_g), dim=1)
        return out


class SFIN(nn.Module):
    def __init__(self):
        super(SFIN, self).__init__()
        self.blocks = []
        for i in range(n_blocks):
            self.blocks.append(ResnetBlock())
        self.body = nn.Sequential(*self.blocks)
        self.head_conv = nn.Conv2d(1, ch, 3, 1, 1)
        self.tail_conv = nn.Conv2d(ch, 1, 3, 1, 1)

    def forward(self, x):
        x = self.head_conv(x)
        shortcut = x
        x = self.body(x)
        x += shortcut
        x = self.tail_conv(x)
        return x


if __name__ == '__main__':
    flops, params = info(SFIN(), (1, 256, 256), as_strings=False,
                         print_per_layer_stat=False, verbose=False)
    print(flops, params)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def depthwise_conv(inp, oup, kernel_size=3, stride=1, relu=False):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, kernel_size // 2, groups=inp, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True) if relu else nn.Sequential(),
    )


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class GhostBottleneck(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride):
        super(GhostBottleneck, self).__init__()
        assert stride in [1, 2]

        self.conv = nn.Sequential(
            # pw
            GhostModule(inp, hidden_dim, kernel_size=1, relu=True),
            # dw
            depthwise_conv(hidden_dim, hidden_dim, kernel_size, stride, relu=False) if stride == 2 else nn.Sequential(),
            # Squeeze-and-Excite
            SELayer(hidden_dim),
            # pw-linear
            GhostModule(hidden_dim, oup, kernel_size=1, relu=False),
        )

        if stride == 1 and inp == oup:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                depthwise_conv(inp, inp, kernel_size, stride, relu=False),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, sr_ratio=1, qkv_bias=False, qk_scale=None):
        super(CrossAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.q2 = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr1 = nn.Conv2d(dim, dim, kernel_size=sr_ratio + 1, stride=sr_ratio,
                                 padding=sr_ratio // 2, groups=dim)
            self.norm1 = nn.LayerNorm(dim)
            self.sr2 = nn.Conv2d(dim, dim, kernel_size=sr_ratio + 1, stride=sr_ratio,
                                 padding=sr_ratio // 2, groups=dim)
            self.norm2 = nn.LayerNorm(dim)

    def forward(self, x1, x2, H, W):
        B, N, C = x1.shape
        q1 = self.q1(x1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q2 = self.q2(x2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_1 = x1.permute(0, 2, 1).reshape(B, C, H, W)
            x_1 = self.sr1(x_1).reshape(B, C, -1).permute(0, 2, 1)
            x_1 = self.norm1(x_1)
            kv1 = self.kv1(x_1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

            x_2 = x2.permute(0, 2, 1).reshape(B, C, H, W)
            x_2 = self.sr2(x_2).reshape(B, C, -1).permute(0, 2, 1)
            x_2 = self.norm2(x_2)
            kv2 = self.kv2(x_2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv1 = self.kv1(x1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            kv2 = self.kv2(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k1, v1 = kv1[0], kv1[1]
        k2, v2 = kv2[0], kv2[1]

        attn1 = (q1 @ k2.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1)
        attn2 = (q2 @ k1.transpose(-2, -1)) * self.scale
        attn2 = attn2.softmax(dim=-1)

        main_out = (attn1 @ v2).transpose(1, 2).reshape(B, N, C)
        aux_out = (attn2 @ v1).transpose(1, 2).reshape(B, N, C)

        return main_out, aux_out


class FeatureInteraction(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, sr_ratio=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.channel_proj1 = nn.Linear(dim, dim // reduction * 2)
        self.channel_proj2 = nn.Linear(dim, dim // reduction * 2)
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        self.cross_attn = CrossAttention(dim // reduction, num_heads=num_heads, sr_ratio=sr_ratio)
        self.end_proj1 = nn.Linear(dim // reduction * 2, dim)
        self.end_proj2 = nn.Linear(dim // reduction * 2, dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x1, x2, H, W):
        y1, z1 = self.act1(self.channel_proj1(x1)).chunk(2, dim=-1)
        y2, z2 = self.act2(self.channel_proj2(x2)).chunk(2, dim=-1)
        c1, c2 = self.cross_attn(z1, z2, H, W)
        y1 = torch.cat((y1, c1), dim=-1)
        y2 = torch.cat((y2, c2), dim=-1)
        main_out = self.norm1(x1 + self.end_proj1(y1))
        aux_out = self.norm2(x2 + self.end_proj2(y2))
        return main_out, aux_out


class ChannelEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):  # 强制使用 BatchNorm2d
        super(ChannelEmbed, self).__init__()
        self.out_channels = out_channels
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        self.channel_embed = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // reduction, kernel_size=1, bias=True),
            nn.Conv2d(out_channels // reduction, out_channels // reduction, kernel_size=3,
                      stride=1, padding=1, bias=True, groups=out_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1, bias=True),
            norm_layer(out_channels)
        )
        self.norm = norm_layer(out_channels)

    def forward(self, x, H, W):
        B, N, _C = x.shape
        x = x.permute(0, 2, 1).reshape(B, _C, H, W).contiguous()
        residual = self.residual(x)
        x = self.channel_embed(x)
        out = self.norm(residual + x)
        return out


class FeatureFusion(nn.Module):
    def __init__(self, dim, reduction=1, sr_ratio=1, num_heads=None, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.cross = FeatureInteraction(dim=dim, reduction=reduction,
                                        num_heads=num_heads, sr_ratio=sr_ratio)
        self.channel_emb = ChannelEmbed(in_channels=dim * 2, out_channels=dim,
                                        reduction=reduction, norm_layer=norm_layer)

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)
        x1, x2 = self.cross(x1, x2, H, W)
        fuse = torch.cat((x1, x2), dim=-1)
        fuse = self.channel_emb(fuse, H, W)
        return fuse


class FreConv(nn.Module):
    def __init__(self, c, reduction, k=1, p=0):
        super(FreConv, self).__init__()
        if reduction == 1:
            self.freq_attention = nn.Sequential(
                nn.Conv2d(c, 1, kernel_size=k, padding=p, bias=False),
            )
        else:
            self.freq_attention = nn.Sequential(
                nn.Conv2d(c, c // reduction, kernel_size=k, bias=False, padding=p),
                nn.ReLU(),
                nn.Conv2d(c // reduction, 1, kernel_size=k, padding=p, bias=False)
            )

    def forward(self, x):
        return self.freq_attention(x)


class DCTSA(nn.Module):
    def __init__(self, freq_num, channel, step, reduction=1, groups=1, select_method='all'):
        super(DCTSA, self).__init__()
        self.freq_num = freq_num
        self.channel = channel
        self.reduction = reduction
        self.select_method = select_method
        self.groups = groups
        self.step = step

        self.avg_pool_c = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.max_pool_c = nn.AdaptiveMaxPool3d((None, 1, 1))

        self.alpha = nn.Parameter(torch.FloatTensor([0.5]))
        self.beta = nn.Parameter(torch.FloatTensor([0.5]))

        if freq_num == 64:
            self.dct_filter = DCT8x8()
            self.p = int((self.dct_filter.freq_range - 1) / 2)
        elif freq_num == 49:
            self.dct_filter = DCT7x7()
            self.p = int((self.dct_filter.freq_range - 1) / 2)
        elif freq_num == 9:
            self.dct_filter = DCT3x3()
            self.p = int((self.dct_filter.freq_range - 1) / 2)
        else:
            self.dct_filter = DCT8x8()
            self.p = int((self.dct_filter.freq_range - 1) / 2)

        if self.select_method == 'all':
            self.dct_c = self.dct_filter.freq_num
        elif 's' in self.select_method:
            self.dct_c = 1
        elif 'top' in self.select_method:
            self.dct_c = int(self.select_method.replace('top', ''))
        else:
            self.dct_c = self.dct_filter.freq_num  # 默认使用所有频率

        self.freq_attention = FreConv(self.dct_c, reduction=reduction, k=7, p=3)
        self.sigmoid = nn.Sigmoid()

        self.fc_t = nn.Linear(step, step, bias=False)

        self.t = nn.Parameter(torch.FloatTensor([0.6]))  # 时间权重
        self.s = nn.Parameter(torch.FloatTensor([0.5]))  # 空间权重

        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))  # 确保输出为8x8

    def forward(self, x):
        orig_dtype = x.dtype

        x = x.float()

        # x: [T, B, C, H, W]
        t, b, c, h, w = x.shape
        x = rearrange(x, 't b c h w -> b t c h w')  # [B, T, C, H, W]

        avg_map = self.avg_pool_c(x)  # (b, t, c, 1, 1)
        max_map = self.max_pool_c(x)
        map_add = self.alpha * avg_map + self.beta * max_map
        map_add = rearrange(map_add, 'b t c 1 1 -> b c t')  # [B, C, T]
        map_fusion_t = self.fc_t(map_add).transpose(1, 2)  # [B, T, C]

        t_mean_sig = self.sigmoid(torch.mean(map_fusion_t, dim=2))  # [B, T]
        t_mean_sig = rearrange(t_mean_sig, 'b t -> b t 1 1 1')  # [B, T, 1, 1, 1]
        t_mean_sig = t_mean_sig.repeat(1, 1, c, h, w)  # [B, T, C, H, W]

        x_t = x * t_mean_sig + x  # 时间增强特征 [B, T, C, H, W]

        x_t_mean = torch.mean(x_t, dim=1, keepdim=False)  # [B, C, H, W]
        x_t_pooled = self.adaptive_pool(x_t_mean)  # [B, C, 8, 8]

        if self.select_method == 'all':
            dct_weight = self.dct_filter.filter
            dct_weight = dct_weight.unsqueeze(1)  # [dct_c, 1, 8, 8]
            dct_weight = dct_weight.repeat(1, self.channel, 1, 1)  # [dct_c, C, 8, 8]
        elif 's' in self.select_method:
            filter_id = int(self.select_method.replace('s', ''))
            dct_weight = self.dct_filter.get_filter(filter_id)  # [8, 8]
            dct_weight = dct_weight.unsqueeze(0).unsqueeze(0)  # [1, 1, 8, 8]
            dct_weight = dct_weight.repeat(1, self.channel, 1, 1)  # [1, C, 8, 8]
        elif 'top' in self.select_method:
            filter_id = self.dct_filter.get_topk(self.dct_c)
            dct_weight = self.dct_filter.get_filter(filter_id)  # [dct_c, 8, 8]
            dct_weight = dct_weight.unsqueeze(1)  # [dct_c, 1, 8, 8]
            dct_weight = dct_weight.repeat(1, self.channel, 1, 1)  # [dct_c, C, 8, 8]
        else:
            dct_weight = self.dct_filter.filter
            dct_weight = dct_weight.unsqueeze(1)
            dct_weight = dct_weight.repeat(1, self.channel, 1, 1)

        dct_bias = torch.zeros(self.dct_c).to(dct_weight.device)
        dct_feature = F.conv2d(x_t_pooled, dct_weight, dct_bias, stride=1, padding=self.p)  # [b, dct_c, H', W']

        if dct_feature.size(2) != 8 or dct_feature.size(3) != 8:
            dct_feature = F.interpolate(dct_feature, size=(8, 8), mode='bilinear', align_corners=False)

        dct_feature = self.freq_attention(dct_feature)  # [b, 1, 8, 8]

        dct_feature = dct_feature.unsqueeze(1)  # [B, 1, 1, 8, 8]
        dct_feature = dct_feature.repeat(1, t, c, 1, 1)  # [B, T, C, 8, 8]

        x_t_flat = rearrange(x_t, 'b t c h w -> (b t) c h w')  # [B*T, C, H, W]
        x_t_resized = F.interpolate(x_t_flat, size=(8, 8), mode='bilinear', align_corners=False)  # [B*T, C, 8, 8]
        x_t_resized = rearrange(x_t_resized, '(b t) c h w -> b t c h w', b=b, t=t)  # [B, T, C, 8, 8]

        x_s = x_t_resized * self.sigmoid(dct_feature) + x_t_resized  # [B, T, C, 8, 8]

        x_s_flat = rearrange(x_s, 'b t c h w -> (b t) c h w')  # [B*T, C, 8, 8]
        x_s_resized = F.interpolate(x_s_flat, size=(h, w), mode='bilinear', align_corners=False)  # [B*T, C, H, W]
        x_s = rearrange(x_s_resized, '(b t) c h w -> b t c h w', b=b, t=t)  # [B, T, C, H, W]

        x_out = (x_t * self.t + x_s * self.s) / 2  # [B, T, C, H, W]
        x_out = rearrange(x_out, 'b t c h w -> t b c h w')  # [T, B, C, H, W]

        x_out = x_out.to(orig_dtype)
        return x_out


class TBSILayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, fusion_reduction=16, fusion_sr_ratio=4,
                 ghost_ratio=2):
        super().__init__()

        self.t_fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )

        self.ca_s2t_v2f = CASTBlock(
            dim=dim, num_heads=num_heads, mode='s2t', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
            attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer
        )
        self.ca_t2s_f2v = CASTBlock(
            dim=dim, num_heads=num_heads, mode='t2s', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
            attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer
        )
        self.ca_s2t_i2f = CASTBlock(
            dim=dim, num_heads=num_heads, mode='s2t', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
            attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer
        )
        self.ca_t2s_f2i = CASTBlock(
            dim=dim, num_heads=num_heads, mode='t2s', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
            attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer
        )
        self.ca_t2t_f2v = CASTBlock(
            dim=dim, num_heads=num_heads, mode='t2t', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
            attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer
        )
        self.ca_t2t_f2i = CASTBlock(
            dim=dim, num_heads=num_heads, mode='t2t', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
            attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer
        )

        self.dctsa_v = DCTSA(freq_num=64, channel=dim, step=1, reduction=16)
        self.dctsa_i = DCTSA(freq_num=64, channel=dim, step=1, reduction=16)
        self.dctsa_f = DCTSA(freq_num=64, channel=dim, step=1, reduction=16)
        self.feature_fusion = FeatureFusion(
            dim=dim,
            reduction=fusion_reduction,
            sr_ratio=fusion_sr_ratio,
            num_heads=num_heads,
            norm_layer=nn.BatchNorm2d
        )

        self.sf_blocks = nn.Sequential(
            SFIB(in_channels=dim),
            SFIB(in_channels=dim)
        )
        self.ghost_bottleneck = GhostBottleneck(
            inp=dim,
            hidden_dim=dim * ghost_ratio,
            oup=dim,
            kernel_size=3,
            stride=1
        )

        self.norm_pre_fusion = norm_layer(dim)
        self.norm_post_fusion = norm_layer(dim)
        self.fourier_unit = FourierUnit(in_channels=dim, out_channels=dim)

    def apply_dctsa(self, module, x, spatial_shape):
        B, N, C = x.shape

        x_patch = token2patch(x)  # [B, C, H, W]

        x_patch = x_patch.unsqueeze(0)  # [1, B, C, H, W]

        x_enhanced = module(x_patch)  # [1, B, C, H, W]

        x_enhanced = x_enhanced.squeeze(0)  # [B, C, H, W]

        return patch2token(x_enhanced)  # [B, N, C]

    def forward(self, x_v, x_i, lens_z):
        x_v_template = x_v[:, :lens_z, :]
        x_v_search = x_v[:, lens_z:, :]
        x_i_template = x_i[:, :lens_z, :]
        x_i_search = x_i[:, lens_z:, :]

        x_v_template = self.apply_dctsa(self.dctsa_v, x_v_template, (8, 8))
        x_i_template = self.apply_dctsa(self.dctsa_i, x_i_template, (8, 8))

        fused_t = torch.cat([x_v_template, x_i_template], dim=2)
        fused_t = self.t_fusion(fused_t)
        fused_t = self.apply_dctsa(self.dctsa_f, fused_t, (8, 8))

        fused_t = self.ca_s2t_i2f(torch.cat([fused_t, x_i_search], dim=1))[:, :lens_z, :]
        temp_x_v_search = self.ca_t2s_f2v(torch.cat([fused_t, x_v_search], dim=1))[:, lens_z:, :]

        fused_t = self.ca_s2t_v2f(torch.cat([fused_t, x_v_search], dim=1))[:, :lens_z, :]
        temp_x_i_search = self.ca_t2s_f2i(torch.cat([fused_t, x_i_search], dim=1))[:, lens_z:, :]

        x_v_search = temp_x_v_search
        x_i_search = temp_x_i_search

        x_v_template = self.ca_t2t_f2v(torch.cat([x_v_template, fused_t], dim=1))[:, :lens_z, :]
        x_i_template = self.ca_t2t_f2i(torch.cat([x_i_template, fused_t], dim=1))[:, :lens_z, :]

        B, L_t, C = x_v_template.shape

        x_v_template_norm = self.norm_pre_fusion(x_v_template)
        x_i_template_norm = self.norm_pre_fusion(x_i_template)

        H = W = int(math.sqrt(L_t))
        x_v_map = x_v_template_norm.permute(0, 2, 1).reshape(B, C, H, W)
        x_i_map = x_i_template_norm.permute(0, 2, 1).reshape(B, C, H, W)

        fused_map = self.feature_fusion(x_v_map, x_i_map)
        sf_enhanced = self.sf_blocks(fused_map)
        freq_enhanced = self.fourier_unit(sf_enhanced)

        freq_enhanced = fused_map + freq_enhanced

        ghost_enhanced = self.ghost_bottleneck(freq_enhanced)

        fused_seq = ghost_enhanced.flatten(2).transpose(1, 2)
        fused_seq = self.norm_post_fusion(fused_seq)

        x_v_template = x_v_template + fused_seq
        x_i_template = x_i_template + fused_seq

        x_v = torch.cat([x_v_template, x_v_search], dim=1)
        x_i = torch.cat([x_i_template, x_i_search], dim=1)

        return x_v, x_i


def token2patch(token):
    if token.dim() == 4:
        return token
    elif token.dim() == 3:
        B, N, C = token.shape
        if N == 256:
            return token.permute(0, 2, 1).reshape(B, C, 16, 16)
        elif N == 64:
            return token.permute(0, 2, 1).reshape(B, C, 8, 8)
        h = int(sqrt(N))
        w = N // h
        return token.permute(0, 2, 1).reshape(B, C, h, w)
    else:
        raise ValueError(f"Invalid input dimension: {token.dim()}")


def patch2token(patch):
    if patch.dim() == 3:
        return patch
    elif patch.dim() == 4:
        B, C, H, W = patch.shape
        N = H * W
        token = patch.reshape(B, C, N).permute(0, 2, 1)
        if N == 256:
            return token
        elif N == 64:
            return token
        return token
    else:
        raise ValueError(f"Invalid input dimension: {patch.dim()}")
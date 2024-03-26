import sys
sys.path.append("..")

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np


import torch
import torch.nn as nn

class PixelShuffle3D(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelShuffle3D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size, channels, in_depth, in_height, in_width = x.size()
        channels //= self.upscale_factor ** 3

        out_depth = in_depth * self.upscale_factor
        out_height = in_height * self.upscale_factor
        out_width = in_width * self.upscale_factor

        x = x.view(batch_size, channels, self.upscale_factor, self.upscale_factor, self.upscale_factor,
                   in_depth, in_height, in_width)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(batch_size, channels, out_depth, out_height, out_width)

        return x


class SingleDeconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.block = nn.ConvTranspose3d(
            in_planes,
            out_planes,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
        )

    def forward(self, x):
        return self.block(x)


class SingleConv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super().__init__()
        padding = torch.div((kernel_size - 1), 2, rounding_mode='trunc').item()
        self.block = nn.Conv3d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,#(kernel_size - 1) // 2,
        )

    def forward(self, x):
        return self.block(x)


class Conv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleConv3DBlock(in_planes, out_planes, kernel_size),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(inplace=True), # TODO: True default
        )

    def forward(self, x):
        return self.block(x)


class Deconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            Conv3DBlock(out_planes, out_planes, kernel_size),
            Conv3DBlock(out_planes, out_planes, kernel_size),
            SingleDeconv3DBlock(in_planes, out_planes),
        )

    def forward(self, x):
        return self.block(x)



    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights



class UNet(nn.Module):
    def __init__(
        self,
        feature_size=16,
        patch_size=(60, 60, 60),
        in_channels=1,
        out_channels=1,
        embed_dim=768,
        num_heads=12,
        dropout=0.1,
    ):
        super().__init__()

        self.upsampler = SingleDeconv3DBlock(1, feature_size) # removed

        # U-Net Decoder
        self.encoder0 = nn.Sequential(
            Conv3DBlock(in_channels, feature_size, 3),
            Conv3DBlock(feature_size, feature_size * 2, 3),
        )

        self.encoder1 = nn.Sequential(
            Conv3DBlock(feature_size * 2, feature_size * 2, 3),
            Conv3DBlock(feature_size * 2, feature_size * 4, 3),
        )

        self.downsampler = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder2 = nn.Sequential(
            Conv3DBlock(feature_size * 4, feature_size * 4, 3),
            Conv3DBlock(feature_size * 4, feature_size * 8, 3),
            SingleDeconv3DBlock(feature_size * 8, feature_size * 8),
        )

        self.decoder2 = nn.Sequential(
            Conv3DBlock(feature_size * 12, feature_size * 8, 3),
            Conv3DBlock(feature_size * 8, feature_size * 4, 3),
            SingleDeconv3DBlock(feature_size * 4, feature_size * 4),
        )

        self.decoder1 = nn.Sequential(
            Conv3DBlock(feature_size * 6, feature_size * 2, 3),
            Conv3DBlock(feature_size * 2, feature_size * 2, 3),
            SingleDeconv3DBlock(feature_size * 2, feature_size * 2),
        )

        self.decoder0header = nn.Sequential(
            Conv3DBlock(feature_size * 3, feature_size * 2, 3),
            Conv3DBlock(feature_size * 2, feature_size, 3),
            Conv3DBlock(feature_size, out_channels, 3),
        )


    def _print_available_memory(self, cuda_device_index=0):
        # Set the device to the desired CUDA device
        torch.cuda.set_device(cuda_device_index)

        # Get total and allocated memory in bytes
        total_memory = torch.cuda.get_device_properties(cuda_device_index).total_memory
        allocated_memory = torch.cuda.memory_allocated(cuda_device_index)

        # Calculate free memory
        free_memory = total_memory - allocated_memory

        # Convert bytes to gigabytes for easier interpretation
        total_memory_gb = total_memory / (1024**3)
        allocated_memory_gb = allocated_memory / (1024**3)
        free_memory_gb = free_memory / (1024**3)
        print("--------------------------------------")
        print(f"Total VRAM: {total_memory_gb:.2f} GB")
        print(f"Allocated VRAM: {allocated_memory_gb:.2f} GB")
        print(f"Free VRAM: {free_memory_gb:.2f} GB")
        print("--------------------------------------")

    def forward(self, x):
        zu = self.upsampler(x)
        z0 = self.encoder0(x)
        z1 = self.encoder1(self.downsampler(z0))
        z2 = self.encoder2(self.downsampler(z1))
        z3 = self.decoder2(torch.cat([z1, z2], dim=1))
        z4 = self.decoder1(torch.cat([z0, z3], dim=1))
        out = self.decoder0header(torch.cat([zu, z4], dim=1))

        return out

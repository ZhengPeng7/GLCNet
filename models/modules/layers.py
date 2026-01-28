import torch.nn as nn


def build_act_layer(act_layer):
    """Build activation layer from config string."""
    if act_layer == 'ReLU':
        return nn.ReLU()
    elif act_layer == 'SiLU':
        return nn.SiLU()
    elif act_layer == 'GELU':
        return nn.GELU()
    raise NotImplementedError(f'build_act_layer does not support {act_layer}')


class ToChannelsFirst(nn.Module):
    """Convert (B, H, W, C) to (B, C, H, W)."""
    def forward(self, x):
        return x.permute(0, 3, 1, 2)


class ToChannelsLast(nn.Module):
    """Convert (B, C, H, W) to (B, H, W, C)."""
    def forward(self, x):
        return x.permute(0, 2, 3, 1)


def build_norm_layer(dim, norm_layer, in_format='channels_first', out_format='channels_first', eps=1e-6, num_groups=32):
    """Build 2D normalization layer (BN/LN/GN) with optional format conversion."""
    layers = []

    if norm_layer == 'BN':
        if in_format == 'channels_last':
            layers.append(ToChannelsFirst())
        layers.append(nn.BatchNorm2d(dim, eps=eps))
        if out_format == 'channels_last':
            layers.append(ToChannelsLast())

    elif norm_layer == 'LN':
        if in_format == 'channels_first':
            layers.append(ToChannelsLast())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == 'channels_first':
            layers.append(ToChannelsFirst())

    elif norm_layer == 'GN':
        if in_format == 'channels_last':
            layers.append(ToChannelsFirst())
        num_groups = min(num_groups, dim)
        while dim % num_groups != 0:
            num_groups -= 1
        layers.append(nn.GroupNorm(num_groups, dim, eps=eps))
        if out_format == 'channels_last':
            layers.append(ToChannelsLast())

    else:
        raise NotImplementedError(f'build_norm_layer does not support {norm_layer}')

    return nn.Sequential(*layers) if len(layers) > 1 else layers[0]


def build_norm_layer_1d(dim, norm_layer, eps=1e-3, num_groups=32, affine=True):
    """Build 1D normalization layer (BN/LN/GN) for Linear outputs."""
    if norm_layer == 'BN':
        return nn.BatchNorm1d(dim, eps=eps, affine=affine)
    elif norm_layer == 'LN':
        return nn.LayerNorm(dim, eps=eps, elementwise_affine=affine)
    elif norm_layer == 'GN':
        num_groups = min(num_groups, dim)
        while dim % num_groups != 0:
            num_groups -= 1
        return nn.GroupNorm(num_groups, dim, eps=eps, affine=affine)
    else:
        raise NotImplementedError(f'build_norm_layer_1d does not support {norm_layer}')

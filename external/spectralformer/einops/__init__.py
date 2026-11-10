"""Minimal local compatibility shim for the two einops patterns used upstream."""
import torch


def rearrange(x, pattern, **axes):
    if pattern == 'b n (h d) -> b h n d':
        b, n, hd = x.shape
        h = int(axes['h'])
        return x.reshape(b, n, h, hd // h).permute(0, 2, 1, 3)
    if pattern == 'b h n d -> b n (h d)':
        b, h, n, d = x.shape
        return x.permute(0, 2, 1, 3).reshape(b, n, h * d)
    raise NotImplementedError(pattern)


def repeat(x, pattern, **axes):
    if pattern == '() n d -> b n d':
        return x.expand(int(axes['b']), -1, -1)
    raise NotImplementedError(pattern)

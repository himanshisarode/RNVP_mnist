import torch
import torch.nn as nn
import torch.nn.functional as F

from models.real_nvp.coupling_layer import CouplingLayer, SpatialMaskType, ChannelMaskType
from util import squeeze_2x2


class RealNVP(nn.Module):
    def __init__(self, num_scales=2, in_channels=3, mid_channels=64, num_blocks=8,
                 mask_pairs=[(SpatialMaskType.CHECKERBOARD, ChannelMaskType.HALF)]):

        super(RealNVP, self).__init__()

        self.register_buffer('data_constraint', torch.tensor([0.9], dtype=torch.float32))

        self.mask_pairs = mask_pairs

        self.flows = _RealNVP(
            scale_idx=0,
            num_scales=num_scales,
            in_channels=in_channels,
            mid_channels=mid_channels,
            num_blocks=num_blocks,
            mask_pairs=self.mask_pairs,
            layer_counter=[0]   # mutable counter shared across recursion
        )

    def forward(self, x, reverse=False):
        sldj = None

        if not reverse:
            if x.min() < 0 or x.max() > 1:
                raise ValueError(f'Expected x in [0,1], got {x.min()}/{x.max()}')

            x, sldj = self._pre_process(x)

        x, sldj = self.flows(x, sldj, reverse)
        return x, sldj

    def _pre_process(self, x):
        y = (x * 255. + torch.rand_like(x)) / 256.
        y = (2 * y - 1) * self.data_constraint
        y = (y + 1) / 2
        y = y.log() - (1. - y).log()

        ldj = F.softplus(y) + F.softplus(-y) \
            - F.softplus((1. - self.data_constraint).log() - self.data_constraint.log())

        sldj = ldj.reshape(ldj.size(0), -1).sum(-1)
        return y, sldj


# ======================================================
# INTERNAL BUILDER
# ======================================================

class _RealNVP(nn.Module):
    def __init__(self, scale_idx, num_scales, in_channels, mid_channels, num_blocks,
                 mask_pairs, layer_counter):

        super(_RealNVP, self).__init__()

        self.is_last_block = scale_idx == num_scales - 1

        self.mask_pairs = mask_pairs
        self.layer_counter = layer_counter  # shared counter

        # =========================
        # SPATIAL COUPLINGS
        # =========================
        self.in_couplings = nn.ModuleList([
            self._make_spatial_layer(in_channels, mid_channels, num_blocks, False),
            self._make_spatial_layer(in_channels, mid_channels, num_blocks, True),
            self._make_spatial_layer(in_channels, mid_channels, num_blocks, False),
        ])

        if self.is_last_block:
            self.in_couplings.append(
                self._make_spatial_layer(in_channels, mid_channels, num_blocks, True)
            )

        else:
            # =========================
            # CHANNEL COUPLINGS
            # =========================
            self.out_couplings = nn.ModuleList([
                self._make_channel_layer(4 * in_channels, 2 * mid_channels, num_blocks, False),
                self._make_channel_layer(4 * in_channels, 2 * mid_channels, num_blocks, True),
                self._make_channel_layer(4 * in_channels, 2 * mid_channels, num_blocks, False),
            ])

            self.next_block = _RealNVP(
                scale_idx + 1,
                num_scales,
                2 * in_channels,
                2 * mid_channels,
                num_blocks,
                mask_pairs,
                layer_counter
            )

    # =========================
    # HELPERS
    # =========================

    def _get_next_pair(self):
        idx = self.layer_counter[0]
        pair = self.mask_pairs[idx % len(self.mask_pairs)]
        self.layer_counter[0] += 1
        return pair

    def _make_spatial_layer(self, in_c, mid_c, num_blocks, reverse):
        spatial_mask, _ = self._get_next_pair()

        return CouplingLayer(
            in_c, mid_c, num_blocks,
            spatial_mask_type=spatial_mask,
            channel_mask_type=None,
            reverse_mask=reverse
        )

    def _make_channel_layer(self, in_c, mid_c, num_blocks, reverse):
        _, channel_mask = self._get_next_pair()

        return CouplingLayer(
            in_c, mid_c, num_blocks,
            spatial_mask_type=None,
            channel_mask_type=channel_mask,
            reverse_mask=reverse
        )

    # =========================
    # FORWARD
    # =========================

    def forward(self, x, sldj, reverse=False):

        if reverse:
            if not self.is_last_block:

                x = squeeze_2x2(x, reverse=False, alt_order=True)
                x, x_split = x.chunk(2, dim=1)

                x, sldj = self.next_block(x, sldj, reverse)

                x = torch.cat((x, x_split), dim=1)
                x = squeeze_2x2(x, reverse=True, alt_order=True)

                x = squeeze_2x2(x, reverse=False)
                for coupling in reversed(self.out_couplings):
                    x, sldj = coupling(x, sldj, reverse)
                x = squeeze_2x2(x, reverse=True)

            for coupling in reversed(self.in_couplings):
                x, sldj = coupling(x, sldj, reverse)

        else:
            for coupling in self.in_couplings:
                x, sldj = coupling(x, sldj, reverse)

            if not self.is_last_block:

                x = squeeze_2x2(x, reverse=False)
                for coupling in self.out_couplings:
                    x, sldj = coupling(x, sldj, reverse)
                x = squeeze_2x2(x, reverse=True)

                x = squeeze_2x2(x, reverse=False, alt_order=True)
                x, x_split = x.chunk(2, dim=1)

                x, sldj = self.next_block(x, sldj, reverse)

                x = torch.cat((x, x_split), dim=1)
                x = squeeze_2x2(x, reverse=True, alt_order=True)

        return x, sldj
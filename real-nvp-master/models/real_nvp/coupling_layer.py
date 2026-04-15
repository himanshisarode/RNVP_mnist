import torch
import torch.nn as nn

from enum import Enum
from models.resnet import ResNet
from util import (
    checkerboard_mask,
    diagonal_mask,
    vertical_mask,
    horizontal_mask,
    quadrant_mask,
    border_mask,
    alternate_border_mask
)


# =========================
# ENUMS
# =========================

class SpatialMaskType(Enum):
    CHECKERBOARD = "checkerboard"
    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"
    DIAGONAL = "diagonal"
    QUADRANT = "quadrant"
    BORDER = "border"
    ALT_BORDER = "alt_border"


class ChannelMaskType(Enum):
    HALF = "half"              # original split
    ALTERNATE = "alternate"    # even-odd channels
    BORDER = "border"          # first+last channels


# =========================
# CHANNEL MASK HELPERS
# =========================

def channel_half(x, reverse):
    if reverse:
        return x.chunk(2, dim=1)
    else:
        x_change, x_id = x.chunk(2, dim=1)
        return x_id, x_change


def channel_alternate(x, reverse):
    c = x.size(1)
    idx = torch.arange(c, device=x.device)

    if reverse:
        id_idx = idx % 2 == 1
    else:
        id_idx = idx % 2 == 0

    change_idx = ~id_idx

    x_id = x[:, id_idx, :, :]
    x_change = x[:, change_idx, :, :]
    return x_id, x_change


def channel_border(x, reverse):
    c = x.size(1)
    idx = torch.arange(c, device=x.device)

    id_idx = (idx == 0) | (idx == c - 1)

    if reverse:
        id_idx = ~id_idx

    change_idx = ~id_idx

    x_id = x[:, id_idx, :, :]
    x_change = x[:, change_idx, :, :]
    return x_id, x_change


# =========================
# COUPLING LAYER
# =========================

class CouplingLayer(nn.Module):
    def __init__(self, in_channels, mid_channels, num_blocks,
                 spatial_mask_type=None,
                 channel_mask_type=None,
                 reverse_mask=False):

        super(CouplingLayer, self).__init__()

        self.spatial_mask_type = spatial_mask_type
        self.channel_mask_type = channel_mask_type
        self.reverse_mask = reverse_mask

        # Determine input channels for NN
        if channel_mask_type is not None:
            if channel_mask_type == ChannelMaskType.HALF:
                nn_in = in_channels // 2
            elif channel_mask_type == ChannelMaskType.ALTERNATE:
                nn_in = (in_channels + 1) // 2
            elif channel_mask_type == ChannelMaskType.BORDER:
                nn_in = 2 if in_channels > 2 else 1
        else:
            nn_in = in_channels

        self.st_net = ResNet(
            nn_in, mid_channels, 2 * nn_in,
            num_blocks=num_blocks,
            kernel_size=3,
            padding=1,
            double_after_norm=(spatial_mask_type is not None)
        )

        self.rescale = nn.utils.weight_norm(Rescale(nn_in))

    # =========================
    # FORWARD
    # =========================

    def forward(self, x, sldj=None, reverse=True):

        # =====================
        # SPATIAL MASK CASE
        # =====================
        if self.spatial_mask_type is not None:

            h, w = x.size(2), x.size(3)

            if self.spatial_mask_type == SpatialMaskType.CHECKERBOARD:
                b = checkerboard_mask(h, w, self.reverse_mask, device=x.device)

            elif self.spatial_mask_type == SpatialMaskType.VERTICAL:
                b = vertical_mask(h, w, self.reverse_mask, device=x.device)

            elif self.spatial_mask_type == SpatialMaskType.HORIZONTAL:
                b = horizontal_mask(h, w, self.reverse_mask, device=x.device)

            elif self.spatial_mask_type == SpatialMaskType.DIAGONAL:
                b = diagonal_mask(h, w, self.reverse_mask, device=x.device)

            elif self.spatial_mask_type == SpatialMaskType.QUADRANT:
                b = quadrant_mask(h, w, quadrant=0, reverse=self.reverse_mask, device=x.device)

            elif self.spatial_mask_type == SpatialMaskType.BORDER:
                b = border_mask(h, w, self.reverse_mask, device=x.device)

            elif self.spatial_mask_type == SpatialMaskType.ALT_BORDER:
                b = alternate_border_mask(h, w, self.reverse_mask, device=x.device)

            else:
                raise ValueError("Unknown spatial mask")

            x_b = x * b

            st = self.st_net(x_b)
            s, t = st.chunk(2, dim=1)

            s = self.rescale(torch.tanh(s))
            s = s * (1 - b)
            t = t * (1 - b)

            if reverse:
                x = x * torch.exp(-s) - t
            else:
                x = (x + t) * torch.exp(s)
                sldj += s.view(s.size(0), -1).sum(-1)

        # =====================
        # CHANNEL MASK CASE
        # =====================
        else:

            if self.channel_mask_type == ChannelMaskType.HALF:
                x_id, x_change = channel_half(x, self.reverse_mask)

            elif self.channel_mask_type == ChannelMaskType.ALTERNATE:
                x_id, x_change = channel_alternate(x, self.reverse_mask)

            elif self.channel_mask_type == ChannelMaskType.BORDER:
                x_id, x_change = channel_border(x, self.reverse_mask)

            else:
                raise ValueError("Unknown channel mask")

            st = self.st_net(x_id)
            s, t = st.chunk(2, dim=1)

            s = self.rescale(torch.tanh(s))

            if reverse:
                x_change = x_change * torch.exp(-s) - t
            else:
                x_change = (x_change + t) * torch.exp(s)
                sldj += s.view(s.size(0), -1).sum(-1)

            # reconstruct
            if self.channel_mask_type == ChannelMaskType.HALF:
                if self.reverse_mask:
                    x = torch.cat((x_id, x_change), dim=1)
                else:
                    x = torch.cat((x_change, x_id), dim=1)

            else:
                # for alternate and border → need scatter back
                x_new = x.clone()
                c = x.size(1)
                idx = torch.arange(c, device=x.device)

                if self.channel_mask_type == ChannelMaskType.ALTERNATE:
                    if self.reverse_mask:
                        id_idx = idx % 2 == 1
                    else:
                        id_idx = idx % 2 == 0
                else:
                    id_idx = (idx == 0) | (idx == c - 1)
                    if self.reverse_mask:
                        id_idx = ~id_idx

                change_idx = ~id_idx

                x_new[:, id_idx, :, :] = x_id
                x_new[:, change_idx, :, :] = x_change
                x = x_new

        return x, sldj


# =========================
# RESCALE
# =========================

class Rescale(nn.Module):
    def __init__(self, num_channels):
        super(Rescale, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_channels, 1, 1))

    def forward(self, x):
        return self.weight * x
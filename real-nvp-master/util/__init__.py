# =========================
# ARRAY UTILITIES (MASKS + SQUEEZE)
# =========================

from util.array_util import (
    squeeze_2x2,

    # Spatial masks
    checkerboard_mask,
    diagonal_mask,
    vertical_mask,
    horizontal_mask,
    quadrant_mask,
    border_mask,
    alternate_border_mask
)


# =========================
# NORMALIZATION UTILITIES
# =========================

from util.norm_util import (
    get_norm_layer,
    get_param_groups,
    WNConv2d
)


# =========================
# OPTIMIZATION UTILITIES
# =========================

from util.optim_util import (
    bits_per_dim,
    clip_grad_norm
)


# =========================
# SHELL / LOGGING UTILITIES
# =========================

from util.shell_util import (
    AverageMeter
)
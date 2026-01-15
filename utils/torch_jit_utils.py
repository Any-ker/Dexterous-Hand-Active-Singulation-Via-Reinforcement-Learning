# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch


@torch.jit.script
def batch_quat_apply(quat, vec):
    """Rotate each vector in `vec` by the matching quaternion in `quat`."""
    shape = vec.shape
    quat = quat.unsqueeze(1)
    xyz = quat[:, :, :3]
    cross_once = torch.cross(xyz, vec, dim=-1) * 2.0
    rotated = vec + quat[:, :, 3:] * cross_once + torch.cross(xyz, cross_once, dim=-1)
    return rotated.view(shape)


@torch.jit.script
def batch_sided_distance(src, dst):
    """Compute the minimum distance from each src point to the dst set."""
    pairwise = torch.cdist(src, dst)
    distances, _ = torch.min(pairwise, dim=-1)
    return distances


__all__ = ["batch_quat_apply", "batch_sided_distance"]
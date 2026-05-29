import torch
import torch.nn as nn
import torch.nn.functional as F


def charbonnier(x, alpha=0.45, eps=1e-3):
    return (x ** 2 + eps ** 2) ** alpha


def warp(image, flow):
    B, C, H, W = image.shape

    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, dtype=image.dtype, device=image.device),
        torch.arange(W, dtype=image.dtype, device=image.device),
        indexing='ij',
    )
    grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)
    grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)

    sample_x = grid_x + flow[:, 0]   
    sample_y = grid_y + flow[:, 1]   

    sample_x = 2.0 * sample_x / (W - 1) - 1.0
    sample_y = 2.0 * sample_y / (H - 1) - 1.0

    grid = torch.stack([sample_x, sample_y], dim=-1)

    warped = F.grid_sample(
        image, grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True,
    )
    return warped

def photometric_loss(ref, deformed, flow, alpha=0.45, eps=1e-3):
    warped = warp(deformed, flow)
    diff = ref - warped                          
    return charbonnier(diff, alpha=alpha, eps=eps).mean()


def smoothness_loss(flow, alpha=0.45, eps=1e-3):
 
    u = flow[:, 0:1]   
    v = flow[:, 1:2]   

    du_dx = u[:, :, :, 1:] - u[:, :, :, :-1]   
    du_dy = u[:, :, 1:, :] - u[:, :, :-1, :]   
    dv_dx = v[:, :, :, 1:] - v[:, :, :, :-1]   
    dv_dy = v[:, :, 1:, :] - v[:, :, :-1, :]  

    loss = (
        charbonnier(du_dx, alpha=alpha, eps=eps).mean() +
        charbonnier(du_dy, alpha=alpha, eps=eps).mean() +
        charbonnier(dv_dx, alpha=alpha, eps=eps).mean() +
        charbonnier(dv_dy, alpha=alpha, eps=eps).mean()
    )
    return loss

def unsupervised_loss(
    network_output,
    ref,
    deformed,
    weights=None,
    lambda_smooth=0.5,
    alpha=0.45,
    eps=1e-3,
):

    if not isinstance(network_output, (list, tuple)):
        network_output = [network_output]

    if weights is None:
        weights = [0.005, 0.01, 0.02, 0.08, 0.32]

    assert len(weights) == len(network_output), (
        f"Expected {len(weights)} scale outputs, got {len(network_output)}"
    )

    B, C, H, W = ref.shape
    total_loss = 0.0

    for flow_s, weight in zip(network_output, weights):
        flow_up = F.interpolate(
            flow_s, size=(H, W),
            mode='bilinear', align_corners=False,
        )
        l_photo = photometric_loss(ref, deformed, flow_up, alpha=alpha, eps=eps)
        l_smooth = smoothness_loss(flow_up, alpha=alpha, eps=eps)

        scale_loss = l_photo + lambda_smooth * l_smooth
        total_loss = total_loss + weight * scale_loss

    return total_loss


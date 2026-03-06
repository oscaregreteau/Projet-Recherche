import tensorflow as tf
from warp import warp_image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def local_response_norm(img, kernel_size=9):
    avg = tf.nn.avg_pool2d(img, ksize=kernel_size, strides=1, padding='SAME')
    return img - avg


def charbonnier(x, alpha, beta=1.0, eps=1e-3):
    """Generalised Charbonnier loss: (|beta*x|^2 + eps^2)^alpha"""
    return tf.pow((beta * x) ** 2 + eps ** 2, alpha)


def smooth_loss_mask_correction(valid_mask, order=1):
    """
    Builds a corrected mask: ignores smoothness at pixels whose neighbourhood
    contains at least one invalid pixel.
    order=1 uses a 3x3 kernel; order=2 uses a 5x5 kernel.
    """
    if order == 1:
        kernel = tf.constant([
            [[0, 0, 0],
             [0, 1, 1],
             [0, 1, 0]]
        ], dtype=tf.float32)                          # [1, 3, 3]
        kernel = tf.reshape(kernel, [3, 3, 1, 1])    # [H, W, in, out]
        threshold = 2.95
    else:
        kernel = tf.constant([
            [[0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 1, 1, 1],
             [0, 0, 1, 0, 0],
             [0, 0, 1, 0, 0]]
        ], dtype=tf.float32)
        kernel = tf.reshape(kernel, [5, 5, 1, 1])
        threshold = 4.95

    mask_cor = tf.nn.conv2d(valid_mask, kernel, strides=[1, 1, 1, 1], padding='SAME')
    mask_cor = tf.cast(tf.greater_equal(mask_cor, threshold), tf.float32)
    return mask_cor


# ---------------------------------------------------------------------------
# Photo loss
# ---------------------------------------------------------------------------

def photo_loss(flow, frame0, frame1, alpha=0.25, beta=1.0):
    """
    Photometric loss with local response normalisation and Charbonnier penalty.
    Sums absolute diff across channels before applying robust loss.
    """
    frame0 = local_response_norm(frame0)
    frame1 = local_response_norm(frame1)

    warped = warp_image(frame1, flow)
    diff = frame0 - warped
    dist = tf.reduce_sum(tf.abs(diff), axis=3, keepdims=True)
    return charbonnier(dist, alpha=alpha, beta=beta)

def grad_loss(flow, grad0, grad1, alpha, beta):
    """Like photo_loss but operates on pre-computed image gradients."""
    return photo_loss(flow, grad0, grad1, alpha=alpha, beta=beta)

def _flow_neighbor_diffs(flow):
    """
    Applies a finite-difference conv to u and v channels separately.
    Returns concatenated [du_x, du_y, dv_x, dv_y] or [du_x, dv_x, du_y, dv_y]
    depending on kernel order — here we return [dx, dy] stacked per channel.
    """
    kernel = tf.constant([
        [[[0, 0, 0],
          [0, 1, -1],
          [0, 0, 0]]],
        [[[0, 0, 0],
          [0, 1, 0],
          [0, -1, 0]]]
    ], dtype=tf.float32)                              # [2, 1, 3, 3]
    kernel = tf.transpose(kernel, perm=[3, 2, 1, 0]) # [3, 3, 1, 2]

    u = flow[:, :, :, 0:1]
    v = flow[:, :, :, 1:]

    diff_u = tf.nn.conv2d(u, kernel, strides=[1, 1, 1, 1], padding='SAME')
    diff_v = tf.nn.conv2d(v, kernel, strides=[1, 1, 1, 1], padding='SAME')
    return diff_u, diff_v


def smooth_loss(flow, alpha=0.37, beta=1.0,
                valid_pixel_mask=None, img_grad=None, boundary_alpha=0.0):
    """
    First-order smoothness loss with optional:
      - edge-stopping via image gradient (boundary_alpha > 0)
      - valid pixel masking
    """
    diff_u, diff_v = _flow_neighbor_diffs(flow)
    diffs = tf.concat([diff_u, diff_v], axis=3)
    dists = tf.reduce_sum(tf.abs(diffs), axis=3, keepdims=True)
    robust = charbonnier(dists, alpha=alpha, beta=beta)

    if img_grad is not None:
        d_mag = tf.sqrt(tf.reduce_sum(img_grad ** 2, axis=3, keepdims=True) + 1e-8)
        edge_mask = tf.exp(-boundary_alpha * d_mag)
        robust = robust * edge_mask

    if valid_pixel_mask is not None:
        robust = robust * smooth_loss_mask_correction(valid_pixel_mask, order=1)

    return robust


def smooth_loss_2nd(flow, alpha=0.37, beta=1.0,
                    valid_pixel_mask=None, img_grad=None, boundary_alpha=0.0):
    """
    Second-order smoothness loss (penalises curvature rather than slope).
    Applies the finite-difference kernel twice.
    """
    kernel = tf.constant([
        [[[0, 0, 0],
          [0, 1, -1],
          [0, 0, 0]]],
        [[[0, 0, 0],
          [0, 1, 0],
          [0, -1, 0]]]
    ], dtype=tf.float32)
    kernel = tf.transpose(kernel, perm=[3, 2, 1, 0])  # [3, 3, 1, 2]

    u = flow[:, :, :, 0:1]
    v = flow[:, :, :, 1:]

    diff_u = tf.nn.conv2d(u, kernel, strides=[1, 1, 1, 1], padding='SAME')
    diff_v = tf.nn.conv2d(v, kernel, strides=[1, 1, 1, 1], padding='SAME')

    diff_uu = tf.nn.conv2d(diff_u[:, :, :, 0:1], kernel, strides=[1, 1, 1, 1], padding='SAME')
    diff_uv = tf.nn.conv2d(diff_u[:, :, :, 1:2], kernel, strides=[1, 1, 1, 1], padding='SAME')
    diff_vu = tf.nn.conv2d(diff_v[:, :, :, 0:1], kernel, strides=[1, 1, 1, 1], padding='SAME')
    diff_vv = tf.nn.conv2d(diff_v[:, :, :, 1:2], kernel, strides=[1, 1, 1, 1], padding='SAME')

    diffs = tf.concat([diff_uu, diff_uv, diff_vu, diff_vv], axis=3)
    dists = tf.reduce_sum(tf.abs(diffs), axis=3, keepdims=True)
    robust = charbonnier(dists, alpha=alpha, beta=beta)

    if img_grad is not None:
        d_mag = tf.sqrt(tf.reduce_sum(img_grad ** 2, axis=3, keepdims=True) + 1e-8)
        edge_mask = tf.exp(-boundary_alpha * d_mag)
        robust = robust * edge_mask

    if valid_pixel_mask is not None:
        robust = robust * smooth_loss_mask_correction(valid_pixel_mask, order=2)

    return robust

def asymmetric_smooth_loss(flow, occ_mask, valid_pixel_mask,
                            alpha, beta, occ_alpha, occ_beta,
                            img_grad=None, boundary_alpha=0.0):
    """
    Gradient routing trick: smoothness gradients can only flow from
    non-occluded regions into occluded ones, not the reverse.
    """
    flow_valid   = tf.stop_gradient(flow * occ_mask)
    flow_invalid = flow * (1.0 - occ_mask)
    routed_flow  = flow_valid + flow_invalid

    occ_smooth     = smooth_loss(routed_flow, alpha=occ_alpha, beta=occ_beta,
                                  img_grad=img_grad, boundary_alpha=boundary_alpha)
    non_occ_smooth = smooth_loss(flow, alpha=alpha, beta=beta,
                                  valid_pixel_mask=occ_mask,
                                  img_grad=img_grad, boundary_alpha=boundary_alpha)

    valid = smooth_loss_mask_correction(valid_pixel_mask, order=1)
    return (non_occ_smooth + occ_smooth) * valid

def border_occlusion_mask(flow):
    """
    Returns a mask of pixels whose flow vectors stay within the image frame.
    Pixels that map outside are considered occluded (mask = 0).
    """
    b, h, w = tf.shape(flow)[0], tf.shape(flow)[1], tf.shape(flow)[2]

    # grid of base coordinates
    grid_y, grid_x = tf.meshgrid(
        tf.range(h, dtype=tf.float32),
        tf.range(w, dtype=tf.float32),
        indexing='ij'
    )
    grid = tf.stack([grid_x, grid_y], axis=-1)          # [H, W, 2]
    grid = tf.expand_dims(grid, 0)                       # [1, H, W, 2]

    warped_coords = grid + flow                          # [B, H, W, 2]

    in_x = tf.logical_and(warped_coords[..., 0] >= 0,
                           warped_coords[..., 0] <= tf.cast(w - 1, tf.float32))
    in_y = tf.logical_and(warped_coords[..., 1] >= 0,
                           warped_coords[..., 1] <= tf.cast(h - 1, tf.float32))

    mask = tf.cast(tf.logical_and(in_x, in_y), tf.float32)
    return tf.expand_dims(mask, -1)                      # [B, H, W, 1]

def unsup_flow_loss(flow, frame0, frame1, valid_pixel_mask, params):
    """
    Full unsupervised optical flow loss.

    params dict example
    -------------------
    {
        "photo":    {"alpha": 0.25, "beta": 1.0},
        "smooth":   {"alpha": 0.37, "beta": 1.0, "weight": 1.0},
        "smooth_occ": {"alpha": 0.37, "beta": 1.0},
        "smooth2nd":  {"alpha": 0.37, "beta": 1.0, "weight": 0.1},
        "grad":     {"alpha": 0.37, "beta": 1.0, "weight": 0.5},
        "boundary_alpha": 5.0,
        "use_asymmetric_smooth": True,
        "use_smooth2nd": True,
        "use_grad_loss": True,
        "use_boundaries": True,
    }
    """
    p = params

    rgb0  = local_response_norm(frame0)
    rgb1  = local_response_norm(frame1)

    img_grad = None
    if p.get("use_boundaries", False):
        img_grad = tf.image.sobel_edges(rgb0)             # [B, H, W, C, 2]
        img_grad = tf.reshape(img_grad,
                    tf.concat([tf.shape(img_grad)[:3], [-1]], axis=0))

    boundary_alpha = p.get("boundary_alpha", 0.0)

    occ_mask         = border_occlusion_mask(flow)
    occ_invalid_mask = valid_pixel_mask * occ_mask

    photo = photo_loss(flow, rgb0, rgb1,
                       alpha=p["photo"]["alpha"],
                       beta=p["photo"]["beta"])
    photo_masked = photo * occ_invalid_mask

    grad_masked = tf.zeros_like(photo_masked)
    if p.get("use_grad_loss", False):
        gx0 = tf.image.sobel_edges(rgb0)
        gx1 = tf.image.sobel_edges(rgb1)
        gx0 = tf.reshape(gx0, tf.concat([tf.shape(gx0)[:3], [-1]], axis=0))
        gx1 = tf.reshape(gx1, tf.concat([tf.shape(gx1)[:3], [-1]], axis=0))
        g_loss = grad_loss(flow, gx0, gx1,
                           alpha=p["grad"]["alpha"],
                           beta=p["grad"]["beta"])
        grad_masked = g_loss * occ_invalid_mask

    if p.get("use_asymmetric_smooth", False):
        smooth = asymmetric_smooth_loss(
            flow, occ_mask, valid_pixel_mask,
            alpha=p["smooth"]["alpha"],     beta=p["smooth"]["beta"],
            occ_alpha=p["smooth_occ"]["alpha"], occ_beta=p["smooth_occ"]["beta"],
            img_grad=img_grad, boundary_alpha=boundary_alpha
        )
    else:
        smooth = smooth_loss(
            flow,
            alpha=p["smooth"]["alpha"], beta=p["smooth"]["beta"],
            valid_pixel_mask=valid_pixel_mask,
            img_grad=img_grad, boundary_alpha=boundary_alpha
        )

    smooth2nd = tf.zeros_like(smooth)
    if p.get("use_smooth2nd", False):
        smooth2nd = smooth_loss_2nd(
            flow,
            alpha=p["smooth2nd"]["alpha"], beta=p["smooth2nd"]["beta"],
            valid_pixel_mask=valid_pixel_mask,
            img_grad=img_grad, boundary_alpha=boundary_alpha
        )

    photo_avg   = tf.reduce_mean(photo_masked,  axis=[1, 2])
    grad_avg    = tf.reduce_mean(grad_masked,   axis=[1, 2]) * p.get("grad",     {}).get("weight", 0.0)
    smooth_avg  = tf.reduce_mean(smooth,        axis=[1, 2]) * p["smooth"]["weight"]
    smooth2_avg = tf.reduce_mean(smooth2nd,     axis=[1, 2]) * p.get("smooth2nd", {}).get("weight", 0.0)

    total = photo_avg + smooth_avg + grad_avg + smooth2_avg
    return tf.reduce_mean(total), {
        "photo":     tf.reduce_mean(photo_avg),
        "smooth":    tf.reduce_mean(smooth_avg),
        "smooth2nd": tf.reduce_mean(smooth2_avg),
        "grad":      tf.reduce_mean(grad_avg),
    }

def multiscale_loss(flows, frame0, frame1, valid_pixel_mask=None,
                    params=None, scale_weights=None):
    """
    Applies unsup_flow_loss at each predicted scale and returns a weighted sum.
    flows : list of tensors, finest-to-coarsest or coarsest-to-finest — just
            make sure scale_weights lines up.
    """
    if scale_weights is None:
        scale_weights = [1.0, 0.5, 0.25, 0.125, 0.0625]

    if params is None:
        params = {
            "photo":    {"alpha": 0.25, "beta": 1.0},
            "smooth":   {"alpha": 0.37, "beta": 1.0, "weight": 1.0},
            "smooth_occ": {"alpha": 0.37, "beta": 1.0},
            "smooth2nd":  {"alpha": 0.37, "beta": 1.0, "weight": 0.0},
            "grad":       {"alpha": 0.37, "beta": 1.0, "weight": 0.0},
            "boundary_alpha": 0.0,
            "use_asymmetric_smooth": False,
            "use_smooth2nd": False,
            "use_grad_loss": False,
            "use_boundaries": False,
        }

    if valid_pixel_mask is None:
        b, h, w = tf.shape(frame0)[0], tf.shape(frame0)[1], tf.shape(frame0)[2]
        valid_pixel_mask = tf.ones([b, h, w, 1], dtype=tf.float32)

    total = 0.0
    scale_losses = []

    for flow, sw in zip(flows, scale_weights):
        h_s = tf.shape(flow)[1]
        w_s = tf.shape(flow)[2]

        f0_s    = tf.image.resize(frame0,            [h_s, w_s])
        f1_s    = tf.image.resize(frame1,            [h_s, w_s])
        mask_s  = tf.image.resize(valid_pixel_mask,  [h_s, w_s])

        loss_s, components = unsup_flow_loss(flow, f0_s, f1_s, mask_s, params)
        weighted = sw * loss_s
        scale_losses.append(weighted)
        total += weighted

    return total, scale_losses
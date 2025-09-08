import numpy as np

def sift_to_lightglue_format(sift_features, image_size=None, device="cuda:0", rootsift=True):
    """
    sift_features is an array with shape Nx132
    (col, row, scale, orientation) in columns 0-3 and (sift_descriptor) in the following 128 columns
    image_size, if specified, is expected to be a tuple --> image_size = (W, H)
    """
    from lightglue.sift import sift_to_rootsift
    import torch
    assert sift_features.shape[1] == 132
    lightglue_sift = {
        "keypoints": sift_features[:, :2],
        "scales": sift_features[:, 2],
        "oris": np.deg2rad(sift_features[:, 3]),
        "descriptors": sift_features[:, 4:]
    }
    # TODO opencv keypoint responses (equivalent to lightglue scores) are not used here for compatiblity
    # maybe in the future keypoint responses should be incorporated to further improve lightglue performance
    if image_size is not None:
        lightglue_sift["image_size"] = np.array(image_size)
    for k in lightglue_sift:
        lightglue_sift[k] = torch.Tensor(lightglue_sift[k][np.newaxis, ...]).to(device)
    if rootsift:
        # lightglue normalization - by default is true
        lightglue_sift["descriptors"] = sift_to_rootsift(lightglue_sift["descriptors"])
    return lightglue_sift

def lightglue_matching(features_i, features_j, ransac_thr=0.3, max_matches=None, R=None):
    """
    matches_ij: Mx2 array representing M matches. Each match is represented by two values (i, j)
                which means that the i-th kp/row in s2p_features_i matches the j-th kp/row in s2p_features_j

    IMPORTANT!!! 10 GPU GB are necessary to avoid out-of-memory errors (with FT_kp_max=10000)
                 GPU device is set by default
    """
    import torch
    from lightglue import LightGlue
    from lightglue.utils import rbd
    from .ft_opencv import geometric_filtering

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if R is not None:
        # a rotation multiple of 90 degrees exists between both images
        # lightglue matching needs both images with the same orientation
        # happy idea: can we circumvent this by simply rotating the SIFT kp coordinates?
        H, W, phi_deg = R
        features_j_ = rotate_opencv_sift_keypoint_coordinates(features_j, H, W, phi_deg)
    else:
        features_j_ = features_j

    feats0 = sift_to_lightglue_format(features_i, device=DEVICE)
    feats1 = sift_to_lightglue_format(features_j_, device=DEVICE)

    matcher = LightGlue(features='sift').eval().to(DEVICE)
    matches01 = matcher({'image0': feats0, 'image1': feats1})

    matches01 = rbd(matches01) # remove batch dimension - ligthglue utils
    matches_ij = matches01["matches"].detach().cpu().numpy() # (M, 2) torch tensor to numpy
    scores_ij = matches01['scores'].detach().cpu().numpy()   # (M,) confidence for each match
    n_matches = matches_ij.shape[0] if len(matches_ij) > 0 else 0

    """"
    # uncomment check max gpu memory use
    allocated = torch.cuda.memory_allocated(DEVICE) / (1024 ** 3)  # bytes to GB
    reserved = torch.cuda.memory_reserved(DEVICE) / (1024 ** 3)    # bytes to GB
    print(f"START - GPU Memory Allocated: {allocated:.2f} GB")
    print(f"START - GPU Memory Reserved:  {reserved:.2f} GB")
    """

    # free cuda memory
    del matches01, feats0, feats1, matcher
    torch.cuda.empty_cache()

    """"
    # uncomment to verify gpu memory is ~0 after the release
    allocated = torch.cuda.memory_allocated(DEVICE) / (1024 ** 3)  # bytes to GB
    reserved = torch.cuda.memory_reserved(DEVICE) / (1024 ** 3)    # bytes to GB
    print(f"END - GPU Memory Allocated: {allocated:.2f} GB")
    print(f"END - GPU Memory Reserved:  {reserved:.2f} GB")
    """

    if n_matches > 0:
        pix_i = features_i[:, :2].copy()
        pix_j = features_j[:, :2].copy()
        matches_ij, ransac_mask = geometric_filtering(pix_i, pix_j, matches_ij, ransac_thr, return_mask=True)
        if ransac_mask is not None:
            scores_ij = scores_ij[ransac_mask.ravel().astype(bool)] 
        #assert matches_ij.shape[0] == scores_ij.shape[0]
    else:
        matches_ij = None
    n_matches_final = 0 if matches_ij is None else matches_ij.shape[0]

    max_matches = 300 # max_matches = None may generate a lot of redundant matches
    if (max_matches is not None) and (n_matches_final > max_matches):
        sorted_indices = np.argsort(-scores_ij.ravel()) # sort from major confidence prediction to minor
        scores_ij = scores_ij[sorted_indices]
        matches_ij = matches_ij[sorted_indices]
        scores_ij = scores_ij[:max_matches]
        matches_ij = matches_ij[:max_matches]
        n_matches_final = max_matches

    return matches_ij, n_matches, n_matches_final




def affine_image_rotation(w, h, phi_deg, center=None):
    phi = np.deg2rad(phi_deg)
    c, s = np.cos(phi), np.sin(phi)
    # counterclockwise rotation matrix
    R = np.array([[c, s],
                  [-s,  c]], dtype=float)

    if center is None:
        center = np.array([(w - 1) / 2.0, (h - 1) / 2.0], dtype=float)  # (cx, cy)

    # Original image corners in (x,y)
    corners = np.array([[0,     0],
                        [w - 1, 0],
                        [0,     h - 1],
                        [w - 1, h - 1]], dtype=float)

    # Rotate corners about the chosen center
    rc = (corners - center) @ R.T + center

    # Compute translation so that all rotated corners are inside the new canvas
    min_xy = rc.min(axis=0)                    # (min_x, min_y)
    t = -min_xy                                # shift so mins are at 0

    # New canvas size (W', H') that tightly fits the rotated image
    max_xy = rc.max(axis=0)
    w_new = int(np.ceil(max_xy[0] - min_xy[0] + 1))
    h_new = int(np.ceil(max_xy[1] - min_xy[1] + 1))

    # Build 2x3 affine: [R | b], with b = -R*c + c + t
    b = (-R @ center) + center + t
    A = np.hstack([R, b.reshape(2, 1)])        # shape (2,3)

    return A, (w_new, h_new)

def rotate_points(points, A):
    pts = np.asarray(points, dtype=float)      # (N,2) as (x,y)
    R = A[:, :2]
    b = A[:, 2]
    return pts @ R.T + b

def rotate_opencv_sift_keypoint_coordinates(sift_features, H, W, phi_deg):
    #sift_features --> Nx132 array with N sift keypoint descriptors
    #                  each row/keypoint is represented by 132 values:
    #                  (col, row, scale, orientation) in columns 0-3 and (sift_descriptor) in the following 128 columns
    #H, W --> shape of the original image (not the rotated one)
    #phi_deg --> rotation angle in degrees, multiple of 90
    kp_coordinates = np.vstack([sift_features[:, 0], sift_features[:, 1]]).T
    A, _ = affine_image_rotation(W, H, phi_deg=phi_deg)
    rot_sift_features = sift_features.copy()
    rot_sift_features[:, :2] = rotate_points(kp_coordinates, A) # update the point coordinates
    rot_sift_features[:, 3] = sift_features[:, 3] - phi_deg # update the orientation
    return rot_sift_features


########################
# All functions below this originally provided by Elías
########################

def rotate_image(rpc1, rpc2, rgb2):
    #Lightglue matching requires that both images have the same orientation
    from bundle_adjust.cam_utils import suggest_quarter_rotation_from_rpc_scales
    k, rot_angle, debug_info = suggest_quarter_rotation_from_rpc_scales(rpc1, rpc2)
    #Rotate the second image by k*90 degrees (counterclockwise)
    rgb2_rot = np.rot90(rgb2, k=k)
    return rgb2_rot

def _get_center_and_scales(rpc):
    # Works with typical rpcm.RPCModel attribute names; adjust if yours differ.
    lon0 = getattr(rpc, "lon_offset", getattr(rpc, "lon0", 0.0))
    lat0 = getattr(rpc, "lat_offset", getattr(rpc, "lat0", 0.0))
    h0 = getattr(rpc, "height_offset", getattr(rpc, "h0", 0.0))
    slon = getattr(rpc, "lon_scale", getattr(rpc, "lon_scale", 1.0))
    slat = getattr(rpc, "lat_scale", getattr(rpc, "lat_scale", 1.0))
    sh = getattr(rpc, "height_scale", getattr(rpc, "h_scale", 1.0))
    return float(lon0), float(lat0), float(h0), float(slon), float(slat), float(sh)

def _angle(vec2):
    return np.arctan2(vec2[1], vec2[0])

def _snap_quarter_turns(angle_rad):
    ang_deg = np.degrees(angle_rad) % 360.0
    k = int(np.round(ang_deg / 90.0)) % 4
    return k

def _local_vectors_from_scales(rpc, lon, lat, h, dlon, dlat):
    u0, v0 = rpc.projection(lon, lat, h)
    uE, vE = rpc.projection(lon + dlon, lat, h)
    uN, vN = rpc.projection(lon, lat + dlat, h)
    vE_vec = np.array([uE - u0, vE - v0], dtype=float)
    vN_vec = np.array([uN - u0, vN - v0], dtype=float)
    return vE_vec, vN_vec

def suggest_quarter_rotation_from_rpc_scales(rpc_ref, rpc_sec, frac=0.02):
    """
    Suggest how many 90° CCW rotations to apply to the SECOND image to roughly
    align it with the FIRST, using RPC offsets/scales.

    Args:
        rpc_ref, rpc_sec: RPC objects with *.project(lon, lat, h) and
                          attributes lon_offset/lat_offset/height_offset and
                          lon_scale/lat_scale/height_scale (rpcm-compatible).
        frac (float): fraction of lon_scale/lat_scale to use as small step.

    Returns:
        k (int): quarter turns CCW to apply to SECOND image (0,1,2,3)
        rot_deg (int): k*90 degrees
        debug (dict): angles for inspection
    """
    lon1, lat1, h1, slon1, slat1, _ = _get_center_and_scales(rpc_ref)
    lon2, lat2, h2, slon2, slat2, _ = _get_center_and_scales(rpc_sec)

    # Use the average center; use a conservative small step from the two scales
    lon = 0.5 * (lon1 + lon2)
    lat = 0.5 * (lat1 + lat2)
    h = 0.5 * (h1 + h2)

    dlon = frac * 0.5 * (slon1 + slon2)
    dlat = frac * 0.5 * (slat1 + slat2)

    # Guard against pathological scales; fall back to tiny perturbations if needed
    if not np.isfinite(dlon) or dlon == 0.0:
        dlon = 1e-6
    if not np.isfinite(dlat) or dlat == 0.0:
        dlat = 1e-6

    _, vN1 = _local_vectors_from_scales(rpc_ref, lon, lat, h, dlon, dlat)
    _, vN2 = _local_vectors_from_scales(rpc_sec, lon, lat, h, dlon, dlat)

    theta1 = _angle(vN1)
    theta2 = _angle(vN2)
    dtheta = (theta2 - theta1) % (2 * np.pi)

    k = _snap_quarter_turns(dtheta)
    return (
        k,
        int(k * 90),
        {
            "theta_ref_deg": float(np.degrees(theta1) % 360.0),
            "theta_sec_deg": float(np.degrees(theta2) % 360.0),
            "delta_deg": float(np.degrees(dtheta) % 360.0),
            "dlon_used_deg": float(dlon),
            "dlat_used_deg": float(dlat),
        },
    )
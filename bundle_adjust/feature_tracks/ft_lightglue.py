import os
import numpy as np

from bundle_adjust import loader
from bundle_adjust.loader import flush_print, get_id

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

def superpoint_to_lightglue_format(superpoint_features, image_size=None, device="cuda:0"):
    """
    superpoint_features is an array with shape Nx260
    (col, row, score, dummy orientation) in columns 0-3 and (superpoint descriptor) in the following 256 columns
    image_size, if specified, is expected to be a tuple --> image_size = (W, H)
    """
    import torch
    assert superpoint_features.shape[1] == 260
    lightglue_superpoint = {
        "keypoints": superpoint_features[:, :2],
        "keypoint_scores": superpoint_features[:, 2],
        "descriptors": superpoint_features[:, 4:]
    }
    if image_size is not None:
        lightglue_superpoint["image_size"] = np.array(image_size)
    for k in lightglue_superpoint:
        lightglue_superpoint[k] = torch.Tensor(lightglue_superpoint[k][np.newaxis, ...]).to(device)
    return lightglue_superpoint

def lightglue_feature_format(features, image_size=None, device="cuda:0", features_type="sift"):
    if features_type == "sift":
        return sift_to_lightglue_format(features, image_size=image_size, device=device)
    if features_type == "superpoint":
        return superpoint_to_lightglue_format(features, image_size=image_size, device=device)
    raise ValueError("Unknown LightGlue features_type {}".format(features_type))

def lightglue_matching(features_i, features_j, ransac_thr=0.3, matcher=None, max_matches=None, R=None, features_type="sift"):
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

    # avoid loading the matcher for every single pair
    if matcher is None:
        # fallback for direct/debug calls
        matcher = LightGlue(features=features_type,
                            filter_threshold=0.2,
                            depth_confidence=-1,
                            width_confidence=-1,
                            ).eval().to(DEVICE)

    if R is not None:
        # a rotation multiple of 90 degrees exists between both images
        # lightglue matching needs both images with the same orientation
        # happy idea: can we circumvent this by simply rotating the SIFT kp coordinates?
        H, W, phi_deg = R
        if features_type == "sift":
            features_j_ = rotate_opencv_sift_keypoint_coordinates(features_j, H, W, phi_deg)
        else:
            features_j_ = rotate_feature_keypoint_coordinates(features_j, H, W, phi_deg)
    else:
        features_j_ = features_j

    feats0 = lightglue_feature_format(features_i, device=DEVICE, features_type=features_type)
    feats1 = lightglue_feature_format(features_j_, device=DEVICE, features_type=features_type)

    # run matching
    with torch.inference_mode():
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
    del matches01, feats0, feats1

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

    max_matches = 300 if max_matches is None else max_matches # max_matches = None may generate a lot of redundant matches
    if (max_matches is not None) and (n_matches_final > max_matches):
        sorted_indices = np.argsort(-scores_ij.ravel()) # sort from major confidence prediction to minor
        scores_ij = scores_ij[sorted_indices]
        matches_ij = matches_ij[sorted_indices]
        scores_ij = scores_ij[:max_matches]
        matches_ij = matches_ij[:max_matches]
        n_matches_final = max_matches

    return matches_ij, n_matches, n_matches_final


def superpoint_detect(geotiff_path, mask_path=None, offset=None, tracks_config=None):
    """
    Detect SuperPoint keypoints in a single input grayscale image using LightGlue's SuperPoint extractor.

    Output rows follow the bundle_adjust feature convention:
    (col, row, score, dummy orientation, superpoint descriptor)
    """
    import torch
    from lightglue import SuperPoint
    from lightglue.utils import numpy_image_to_torch
    from bundle_adjust.feature_tracks import ft_utils

    config = ft_utils.init_feature_tracks_config(tracks_config)
    max_kp = None if tracks_config is None else config["FT_kp_max"]
    resize = config.get("FT_superpoint_resize", None)

    found_existing_file = False
    if not config["FT_reset"] and "in_dir" in config.keys():
        npy_path_in = os.path.join(config["in_dir"], "features/{}.npy".format(get_id(geotiff_path)))
        if os.path.exists(npy_path_in):
            features_i = np.load(npy_path_in)
            found_existing_file = features_i.ndim == 2 and features_i.shape[1] == 260

    if not found_existing_file:
        im = loader.load_image(geotiff_path, offset=offset, equalize=True).astype(np.uint8)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        extractor = SuperPoint(max_num_keypoints=max_kp).eval().to(device)
        tensor = numpy_image_to_torch(im).to(device)
        with torch.no_grad():
            feats = extractor.extract(tensor, resize=resize)

        keypoints = feats["keypoints"][0].detach().cpu().numpy()
        scores = feats["keypoint_scores"][0].detach().cpu().numpy()
        descriptors = feats["descriptors"][0].detach().cpu().numpy()

        features_i = np.zeros((keypoints.shape[0], 260), dtype=np.float32)
        if keypoints.shape[0] > 0:
            features_i[:, :2] = keypoints
            features_i[:, 2] = scores
            features_i[:, 3] = 0.0
            features_i[:, 4:] = descriptors

        del feats, tensor, extractor
        torch.cuda.empty_cache()

    if features_i.shape[0] > 0:
        features_i = features_i[~np.isnan(features_i[:, 0])]

    if mask_path is not None and features_i.shape[0] > 0:
        mask = np.load(mask_path)
        pts2d_colrow = np.round(features_i[:, :2]).astype(int)
        h, w = mask.shape
        pts2d_colrow[:, 0] = np.clip(pts2d_colrow[:, 0], 0, w - 1)
        pts2d_colrow[:, 1] = np.clip(pts2d_colrow[:, 1], 0, h - 1)
        true_if_obs_inside_aoi = mask[pts2d_colrow[:, 1], pts2d_colrow[:, 0]] > 0
        features_i = features_i[true_if_obs_inside_aoi, :]

    if features_i.shape[0] > 0:
        features_i = np.array(sorted(features_i.tolist(), key=lambda kp: kp[2], reverse=True), dtype=np.float32)
    else:
        features_i = np.zeros((0, 260), dtype=np.float32)
    if max_kp is not None:
        features_i_final = np.zeros((max_kp, 260), dtype=np.float32)
        features_i_final[:] = np.nan
        features_i_final[: min(features_i.shape[0], max_kp)] = features_i[:max_kp]
    else:
        features_i_final = features_i
    n_kp = int(np.sum(~np.isnan(features_i_final[:, 0])))

    if config["FT_save"] and "out_dir" in config.keys():
        npy_path_out = os.path.join(config["out_dir"], "features/{}.npy".format(get_id(geotiff_path)))
        os.makedirs(os.path.dirname(npy_path_out), exist_ok=True)
        np.save(npy_path_out, features_i_final)

    return features_i_final, n_kp


def detect_superpoint_features_image_sequence(geotiff_paths, mask_paths=None, offsets=None, tracks_config=None):
    """
    Detect SuperPoint keypoints in each image of a collection of input grayscale images.
    """

    n_img = len(geotiff_paths)
    features = []
    for i in range(n_img):
        mask_i = None if mask_paths is None else mask_paths[i]
        offset_i = None if offsets is None else offsets[i]
        features_i, n_kp = superpoint_detect(geotiff_paths[i], mask_i, offset_i, tracks_config)
        features.append(features_i)
        flush_print("{} SuperPoint keypoints in image {}".format(n_kp, i))
    return features



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

def rotate_feature_keypoint_coordinates(features, H, W, phi_deg):
    kp_coordinates = np.vstack([features[:, 0], features[:, 1]]).T
    A, _ = affine_image_rotation(W, H, phi_deg=phi_deg)
    rot_features = features.copy()
    rot_features[:, :2] = rotate_points(kp_coordinates, A) # update the point coordinates
    return rot_features

def rotate_opencv_sift_keypoint_coordinates(sift_features, H, W, phi_deg):
    #sift_features --> Nx132 array with N sift keypoint descriptors
    #                  each row/keypoint is represented by 132 values:
    #                  (col, row, scale, orientation) in columns 0-3 and (sift_descriptor) in the following 128 columns
    #H, W --> shape of the original image (not the rotated one)
    #phi_deg --> rotation angle in degrees, multiple of 90
    rot_sift_features = rotate_feature_keypoint_coordinates(sift_features, H, W, phi_deg)
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

def compute_rotations_for_lightglue_alignment(input_pairs, images, lightglue_matching=True):
    """
    LightGlue matching requires images to have consistent orientation
    This function computes possible 90 deg rotations to enforce consistent image orientation between image pairs
    """
    lightglue_correct_orientation = True
    if lightglue_matching and lightglue_correct_orientation:
        # TODO this is currently limited to multiples of 90 deg, observed in the DFC2019 data
        R  = []
        for pair in input_pairs:
            i, j = pair[0], pair[1]
            h, w = images[i].offset["height"], images[i].offset["width"]
            _, phi_deg, _ = suggest_quarter_rotation_from_rpc_scales(images[i].rpc, images[j].rpc)
            R.append(np.array([h, w, phi_deg]))
    else:
        R = None
    return R

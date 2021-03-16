import numpy as np

from bundle_adjust import ba_utils


def ba_offsets_initialize_3d_points(C, mycrops, myrpcs, zmin, zmax, zdelta=0.5):

    print("Initializing 3d pts using RPCs...\n")

    n_pts, n_cam = C.shape[1], int(C.shape[0] / 2)
    output_pts3d = np.zeros((n_pts, 3))

    for i in range(n_pts):
        # idx of images with an observation of feature track i
        im_ind = [k for k, q in enumerate(range(n_cam)) if not np.isnan(C[q * 2, i])]
        n_obs = len(im_ind)

        # for each altitude z in the range of altitudes
        min_sigma = np.inf
        for alt in np.arange(zmin, zmax, zdelta):
            X_i = []
            for k in im_ind:
                # get coordinates of the projection of feature track i in current image
                col_obs, row_obs = (
                    C[k * 2, i] + mycrops[k]["col0"],
                    C[k * 2 + 1, i] + mycrops[k]["row0"],
                )
                # backproject to the plane at height z using rpc
                lon, lat = myrpcs[k].localization(col_obs, row_obs, alt)
                # convert the coordinates of the candidate 3D point from geodetic to geocentric
                x, y, z = ba_utils.latlon_to_ecef_custom(lat, lon, alt)
                # add the 3D point to the list of candidates for track i
                X_i.append(np.hstack([x, y, z]))

            # compute the mean and the std of the list of candidate 3D points
            X_i = np.array(X_i)
            Xhat_i = np.mean(X_i, axis=0)
            sigma_i = np.sqrt(
                np.sum(((X_i[:, 0] - Xhat_i[0]) ** 2) / n_obs)
                + np.sum(((X_i[:, 1] - Xhat_i[1]) ** 2) / n_obs)
            )
            # pick the point with the smaller std as the correct 3D point of feature track i
            if sigma_i < min_sigma:
                min_sigma = sigma_i.copy()
                pt3d = np.array([Xhat_i[0], Xhat_i[1], Xhat_i[2]])
        output_pts3d[i, :] = pt3d
        print("\rpoint {}/{} done".format(i, n_pts), end="\r")
    print("\n")

    return output_pts3d


def ba_offsets_ransac_init_rpc_offset(
    cam_idx, pts3d, C, myrpcs, mycrops, ransac_thr=3.0, ransac_p=0.95, debug=False
):

    """
    Given a set of possible offsets for the view with index 'im_ind', this function
    uses Adaptive RANSAC to select a single offset with the largest support of inliers from the set
    """

    # get all 3d pts that are seen in cam with index cam_idx
    obs_indices = np.arange(C.shape[1])[~np.isnan(C[cam_idx * 2, :])]
    pts3d_seen = pts3d[obs_indices]
    n_obs = pts3d_seen.shape[0]

    # project the 3d points and get their observed location in the images
    pts2d_rpc_prj = np.zeros((n_obs, 2))
    pts2d_img_obs = np.zeros((n_obs, 2))
    for pt_idx, obs_idx in enumerate(obs_indices.tolist()):
        pt3d = pts3d_seen[pt_idx, :]
        lat, lon, alt = ba_utils.ecef_to_latlon_custom(pt3d[0], pt3d[1], pt3d[2])
        col_prj, row_prj = myrpcs[cam_idx].projection(lon, lat, alt)
        pts2d_rpc_prj[pt_idx, :] = np.array([col_prj, row_prj])
        col_obs = C[cam_idx * 2, obs_idx] + mycrops[cam_idx]["col0"]
        row_obs = C[cam_idx * 2 + 1, obs_idx] + mycrops[cam_idx]["row0"]
        pts2d_img_obs[pt_idx, :] = np.array([col_obs, row_obs])

    # get all possible offsets for selected camera (i.e. 1 per point)
    all_offsets = pts2d_img_obs - pts2d_rpc_prj

    # apply ransac to choose the best offset
    import random

    best_inliers, best_offset = [], np.zeros(2)
    it, max_it = 0, np.inf
    p = ransac_p  # probability that at least one of the random samples is an outlier
    d_thr = ransac_thr  # distance threshold to accept inliers

    while it < max_it:
        # pick random sample in image 'im_ind' and initialize offset
        i = random.choice(np.arange(n_obs).tolist())  # selected obs
        offset_i = all_offsets[i, :].copy()

        # compute inliers (i.e. points with reprojection error less than thr after applying offset i)
        pts2d_new_prj = pts2d_rpc_prj + np.repeat(np.array([offset_i]), n_obs, axis=0)
        reprj_err = np.linalg.norm(pts2d_img_obs - pts2d_new_prj, axis=1)
        inliers_i = np.arange(n_obs)[reprj_err < d_thr].tolist()

        n_inliers = len(inliers_i)  # count number of inliers
        w = n_inliers / n_obs  # estimate w - prob that a random point is an inlier
        max_it = np.log10(1 - p) / np.log10(
            1 - w + 1e-16
        )  # update number of iterations

        # test if this is the best model so far (if it is the case, save current offset)
        if n_inliers > len(best_inliers):
            best_inliers = inliers_i.copy()
            best_offset = offset_i.copy()

        it += 1

    if debug:

        print("Initializing offset for image {} with RANSAC\n".format(cam_idx))

        print("- {} points seen".format(n_obs))
        print("- Best offset is {}".format(best_offset))
        print(
            "- Voted by {}% of observations\n".format(len(best_inliers) / n_obs * 100)
        )

        # check that the projection after applying the slected offset is closer to the real observation
        init_reprj_err, new_reprj_err = [], []
        for pt_idx in range(n_obs):
            pts2d_new_prj = pts2d_rpc_prj[pt_idx, :] + best_offset
            init_reprj_err.append(
                np.linalg.norm(pts2d_img_obs[pt_idx, :] - pts2d_rpc_prj[pt_idx, :])
            )
            new_reprj_err.append(
                np.linalg.norm(pts2d_img_obs[pt_idx, :] - pts2d_new_prj)
            )
            # print('- original keypoint coordinates ({}, {})'.format(*np.around(pts2d_img_obs[pt_idx,:],6)))
            # print('- projection using original rpc ({}, {})'.format(*np.around(pts2d_rpc_prj[pt_idx,:],6)))
            # print('- projection using init. offset ({}, {})\n'.format(*np.around(pts2d_new_prj,6)))

        print(
            "- Median RPC reprojection error     : {}".format(np.median(init_reprj_err))
        )
        print(
            "- Median RPC+offset reprojection error : {}\n".format(
                np.median(new_reprj_err)
            )
        )

    return best_offset


def ba_offsets_set_ba_params(C, pts3d, myrpcs, mycrops):

    n_cam = int(C.shape[0] / 2)

    # initialize an offset per each rpc using adaptive ransac
    print("Initializing RPC correction offsets...")
    offsets = np.zeros((n_cam, 2))
    for cam_idx in range(n_cam):
        print("\rRPC {} / {}".format(cam_idx + 1, n_cam), end="\r")
        offsets[cam_idx, :] = ba_offsets_ransac_init_rpc_offset(
            cam_idx,
            pts3d,
            C,
            myrpcs,
            mycrops,
            ransac_thr=3.0,
            ransac_p=0.95,
            debug=False,
        )
    print("\nDone!\n")

    print("Defining the rest of BA parameters...")
    # define camera_ind, points_ind, points_2d as needed in bundle adjustment to refine rpc offsets
    n_pts = pts3d.shape[0]
    point_ind, camera_ind, points_2d = [], [], []
    for i in range(n_pts):
        im_ind = [k for k, j in enumerate(range(n_cam)) if not np.isnan(C[j * 2, i])]
        for j in im_ind:
            point_ind.append(i)
            camera_ind.append(j)
            col_obs, row_obs = (
                C[j * 2, i] + mycrops[j]["col0"],
                C[j * 2 + 1, i] + mycrops[j]["row0"],
            )
            points_2d.append(np.array([col_obs, row_obs]))
    pts_ind, cam_ind, pts_2d, pts_3d = (
        np.array(point_ind),
        np.array(camera_ind),
        np.vstack(points_2d),
        pts3d.copy(),
    )

    print("Done!\n")
    print("{} cameras in total, all will be adjusted".format(n_cam))
    print("2 parameters per camera and {} 3d points to be optimized".format(n_pts))
    return offsets, pts_ind, cam_ind, pts_2d, pts_3d, n_cam, n_pts


def ba_offsets_fun(params, cam_ind, pts_ind, pts_2d, n_cam, n_pts, myrpcs):

    # get 3d points
    pts_3d_ba = params[n_cam * 2 :].reshape((n_pts, 3))
    x, y, z = pts_3d_ba[:, 0], pts_3d_ba[:, 1], pts_3d_ba[:, 2]

    # get offsets
    cam_params_opt = params[: n_cam * 2].reshape((n_cam, 2))

    # project 3d points using the current camera params
    lat, lon, alt = ba_utils.ecef_to_latlon_custom(x[pts_ind], y[pts_ind], z[pts_ind])
    col_prj, row_prj = np.zeros(pts_ind.shape[0]), np.zeros(pts_ind.shape[0])
    for c_idx in np.unique(cam_ind).tolist():
        where_c_idx = cam_ind == c_idx
        col_prj[where_c_idx], row_prj[where_c_idx] = myrpcs[c_idx].projection(
            lon[where_c_idx], lat[where_c_idx], alt[where_c_idx]
        )
    col_prj, row_prj = (
        col_prj + cam_params_opt[cam_ind, 0],
        row_prj + cam_params_opt[cam_ind, 1],
    )
    points_proj = np.vstack((col_prj, row_prj)).T

    # compute reprojection errors
    err = (points_proj - pts_2d).ravel()
    return err


def ba_offsets_bundle_adjustment_sparsity(cam_ind, pts_ind, n_cam, n_pts):

    from scipy.sparse import lil_matrix

    n_params = 2
    m = cam_ind.size * 2
    n = n_cam * n_params + n_pts * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(cam_ind.size)
    for s in range(n_params):
        A[2 * i, cam_ind * n_params + s] = 1
        A[2 * i + 1, cam_ind * n_params + s] = 1

    for s in range(3):
        A[2 * i, n_cam * n_params + pts_ind * 3 + s] = 1
        A[2 * i + 1, n_cam * n_params + pts_ind * 3 + s] = 1

    return A


def ba_offsets_initialize_3d_points_warp(
    C, mycrops, myrpcs, aoi, pairs_to_triangulate, cam_model="Affine"
):
    from bundle_adjust.ba_core import approximate_rpcs_as_proj_matrices
    from bundle_adjust.ba_triangulate import initialize_3d_points

    P_crop = approximate_rpcs_as_proj_matrices(
        myrpcs, mycrops, aoi, cam_model=cam_model
    )
    print("Initializing {} 3d points...".format(C.shape[1]))
    output_pts3d = initialize_3d_points(P_crop, C, pairs_to_triangulate, cam_model)
    print("...Done!\n")
    return output_pts3d


def rpc_affine_approx_with_correction_offset(rpc, p, offset):
    """
    Compute the first order Taylor approximation of an RPC projection function.

    Args:
        rpc: instance of the rpc_model.RPCModel class
        p: x, y, z coordinates

    Return:
        array of shape (3, 4) representing the affine camera matrix equal to the
        first order Taylor approximation of the RPC projection function at point p.
    """
    import ad

    p = ad.adnumber(p)
    # project and then apply the offset translation in the image space
    lat, lon, alt = ba_utils.ecef_to_latlon_custom_ad(*p)
    col, row = rpc.projection(lon, lat, alt)
    q = (col + offset[0], row + offset[1])
    J = ad.jacobian(q, p)

    A = np.zeros((3, 4))
    A[:2, :3] = J
    A[:2, 3] = np.array(q) - np.dot(J, p)
    A[2, 3] = 1
    return A

"""
this file contains some useful functions 
to handle the output of the bundle adjustment 
from the ames stereo pipeline
https://github.com/NeoGeographyToolkit/StereoPipeline
main dependency:
https://github.com/visionworkbench/visionworkbench
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import rpcm
from PIL import Image

from bundle_adjust import ba_core, ba_rotate, ba_utils


def save_pickle(fname, data):
    import pickle

    pickle_out = open(fname, "wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()


def load_pickle(fname):
    import pickle

    pickle_in = open(fname, "rb")
    data = pickle.load(pickle_in)
    pickle_in.close()
    return data


def get_id(fname):
    return os.path.splitext(os.path.basename(fname))[0]


def write_P_to_ASP_tsai_file(fname, P):

    K, R, vecT, oC = ba_core.decompose_perspective_camera(P)
    fx, fy, skew, cx, cy, pitch = K[0, 0], K[1, 1], K[0, 1], K[0, 2], K[1, 2], K[2, 2]
    k1, k2, k3 = 0.0, 0.0, 0.0
    p1, p2 = 0.0, 0.0
    R = np.linalg.inv(R)

    with open(fname, "w") as f_out:
        f_out.write("VERSION_4\n")
        f_out.write("PINHOLE\n")
        f_out.write("fu = {}\n".format(fx))
        f_out.write("fv = {}\n".format(fy))
        f_out.write("cu = {}\n".format(cx))
        f_out.write("cv = {}\n".format(cy))
        f_out.write("u_direction = 1 0 0\n")
        f_out.write("v_direction = 0 1 0\n")
        f_out.write("w_direction = 0 0 1\n")
        f_out.write("C = {} {} {}\n".format(oC[0], oC[1], oC[2]))
        f_out.write(
            "R = {} {} {} {} {} {} {} {} {}\n".format(
                R[0, 0],
                R[0, 1],
                R[0, 2],
                R[1, 0],
                R[1, 1],
                R[1, 2],
                R[2, 0],
                R[2, 1],
                R[2, 2],
            )
        )
        f_out.write("pitch = {}\n".format(pitch))

        # extra parameters (usually we consider a skew factor)
        f_out.write("AdjustableTSAI\n")
        f_out.write("Radial Coeff: Vector3({}, {}, {})\n".format(k1, k2, k3))
        f_out.write("Tangential Coeff: Vector2({}, {})\n".format(p1, p2))
        f_out.write("Alpha: {}\n".format(skew))


def read_P_from_ASP_tsai_file(fname):
    def get_numerical_values_from_line(line):
        import re

        match_number = re.compile("-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?")
        return list(map(np.float64, re.findall(match_number, line)))

    with open(fname) as f:
        lines = f.readlines()
        fx = np.array(get_numerical_values_from_line(lines[2]))[0]
        fy = np.array(get_numerical_values_from_line(lines[3]))[0]
        cx = np.array(get_numerical_values_from_line(lines[4]))[0]
        cy = np.array(get_numerical_values_from_line(lines[5]))[0]
        oC = np.array(get_numerical_values_from_line(lines[9]))
        R = np.array(get_numerical_values_from_line(lines[10])).reshape((3, 3))
        pitch = np.array(get_numerical_values_from_line(lines[11]))[0]
        skew = get_numerical_values_from_line(lines[15])[0]
        K = np.zeros((3, 3)).astype(np.float64)
        K[0, 0], K[1, 1], K[0, 1], K[0, 2], K[1, 2], K[2, 2] = (
            fx,
            fy,
            skew,
            cx,
            cy,
            pitch,
        )
        P = K @ np.linalg.inv(R) @ np.hstack((np.eye(3), -oC[:, np.newaxis]))

    return P / P[2, 3]


def read_ASP_points_file(fname):
    import xml.etree.ElementTree as ET

    tree = ET.parse(fname)
    kml_points = tree.findall(".//{http://www.opengis.net/kml/2.2}Point")
    lon, lat, alt = [], [], []
    for attributes in kml_points:
        for subAttribute in attributes:
            if subAttribute.tag == "{http://www.opengis.net/kml/2.2}coordinates":
                tmp = get_doubles_from_line(subAttribute.text)
                lon.append(tmp[0])
                lat.append(tmp[1])
                alt.append(tmp[2])

    x, y, z = ba_utils.latlon_to_ecef_custom(
        np.array(lat), np.array(lon), np.array(alt)
    )
    pts_3d_ecef = np.vstack([x, y, z]).T

    return pts_3d_ecef


def read_AMES_match_binary_file(fname, verbose=True):
    with open(fname, "r") as f:
        size1, size2 = np.fromfile(f, dtype=np.uint64, count=2)

        if verbose:
            print("{} matches found".format(size1, size2))

        interest_points = []
        for i in range(size1 + size2):
            x, y = np.fromfile(f, dtype=np.float32, count=2)
            ix, iy = np.fromfile(f, dtype=np.int32, count=2)
            orientation, scale, interest = np.fromfile(f, dtype=np.float32, count=3)
            polarity = np.fromfile(f, dtype=np.bool_, count=1)[0]
            octave, scale_lvl = np.fromfile(f, dtype=np.uint32, count=2)
            descriptor_size = np.fromfile(f, dtype=np.uint64, count=1)[0]
            descriptor = np.fromfile(f, dtype=np.float32, count=descriptor_size)
            # maybe in the descriptor we shoud use dtype=float64 since in the C++ code they use double
            # but I got memory allocation errors and the numbers look stranger than with float32

            ip = {
                "x": x,
                "y": y,
                "ix": ix,
                "iy": iy,
                "orientation": orientation,
                "scale": scale,
                "interest": interest,
                "polarity": polarity,
                "octave": octave,
                "scale_lvl": scale_lvl,
                "size": descriptor_size,
                "descriptor": descriptor,
            }

            interest_points.append(ip)

        ip_1 = interest_points[:size1]  # interest points image 1
        ip_2 = interest_points[size1:]  # interest points image 2

    # I am only interested in the 2D image coordinates of the matched points
    pts_1 = np.array([np.array([ip["x"], ip["y"]]) for ip in ip_1])
    pts_2 = np.array([np.array([ip["x"], ip["y"]]) for ip in ip_2])
    des_1 = np.array([ip["descriptor"] for ip in ip_1])
    des_2 = np.array([ip["descriptor"] for ip in ip_2])

    return [pts_1, pts_2, des_1, des_2]


def plot_matches(fname1, fname2, pts1, pts2, color="r", scale=4):

    from bundle_adjust.ba_timeseries import custom_equalization

    fig = plt.figure(figsize=(30, 20))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.imshow(custom_equalization(np.array(Image.open(fname1))), cmap="gray")
    ax2.imshow(custom_equalization(np.array(Image.open(fname2))), cmap="gray")

    ax1.scatter(x=pts1[:, 0], y=pts1[:, 1], c=color, s=scale)
    ax2.scatter(x=pts2[:, 0], y=pts2[:, 1], c=color, s=scale)
    plt.show()


def mean_error_lists_of_points(pts1, pts2):
    return np.round(np.mean(np.linalg.norm(pts1 - pts2, axis=1)), 3)


def custom_get_rpc_rotation_center(rpc, col=0, row=0):

    """
    This function tries to emulate the function point_and_dir used in src/asp/Camera/RPCModel.cc
    It is used to compute the rotation center around which the correcting rotation in the .adjust
    files must rotate. According to the authors such center is usually a point in the ray crossing
    the pixel (0,0) but there can be exceptions.
    """

    # For an RPC model there is no defined origin so it and the ray need to be computed.

    # Center of valid region to bottom of valid region (normalized)
    vert_scale_factor = 0.9  # The virtual center should be above the terrain
    height_up = rpc.alt_offset + rpc.alt_offset * vert_scale_factor
    height_dn = rpc.alt_offset - rpc.alt_offset * vert_scale_factor

    # Given the pixel and elevation, estimate lon-lat.
    # Use m_lonlatheight_offset as initial guess for lonlat_up,
    # and then use lonlat_up as initial guess for lonlat_dn.
    lon_up, lat_up = rpc.localization_iterative(col, row, height_up)
    lon_dn, lat_dn = rpc.localization_iterative(col, row, height_dn)

    geo_up = np.array([lon_up, lat_up, height_up])
    geo_dn = np.array([lon_dn, lat_dn, height_dn])

    geo_up = np.array([lat_up, lon_up, height_up])
    geo_dn = np.array([lat_dn, lon_dn, height_dn])

    P_up = np.array(ba_utils.latlon_to_ecef_custom(*geo_up))
    P_dn = np.array(ba_utils.latlon_to_ecef_custom(*geo_dn))

    v_dir = P_dn - P_up
    v_dir_norm = v_dir / np.linalg.norm(v_dir)

    # Set the origin location very far in the opposite direction of the pointing vector,
    # to put it high above the terrain.
    long_scale_up = 10000
    # This is a distance in meters approx from the top of the llh valid cube
    m_rotation_center = P_up - v_dir_norm * float(long_scale_up)

    return m_rotation_center


def get_doubles_from_line(line):
    ### extracts numbers from a line containing only spaces, \n and numbers
    ### advantage: no need to worry about matching all kinds of notations (decimal, scientific, etc.) with regex
    ### get_numerical_values_from_line caused problems in numbers like 4.6789e06 but read properly 4.6789e6
    tmp = [x for x in line.replace(",", " ").replace("\n", " ").split(" ") if x != ""]
    return list(map(np.float64, tmp))


def read_ASP_adjust_file(fname):
    with open(fname) as f:
        lines = f.readlines()
        m_translation = get_doubles_from_line(lines[0])
        m_rotation = get_doubles_from_line(lines[1])
        m_rotation_center = get_doubles_from_line(lines[4])

        # m_rotation_inv = get_doubles_from_line(lines[2])
        # R = np.array(get_doubles_from_line(lines[3])).reshape(3,3)

    return m_translation, m_rotation, m_rotation_center


def ames_project_adjusted_points(
    rpc_init,
    m_rotation,
    m_translation,
    pts_3d,
    m_rotation_center=None,
    m_rotation_inv=None,
):

    n_pts = pts_3d.shape[0]

    if m_rotation_center is None:
        m_rotation_center = custom_get_rpc_rotation_center(rpc_init)
    m_rotation_center_adj = np.tile(np.array([m_rotation_center]).T, (1, n_pts))
    m_rotation_adj = ba_rotate.quaternion_to_R(*m_rotation)
    m_translation_adj = np.tile(np.array([m_translation]).T, (1, n_pts))

    # m_rotation_inv_adj = ba_rotations.quaternion_to_R(*m_rotation_inv)

    """
    we need to apply
    
    Vector3 AdjustedCameraModel::adjusted_point(Vector3 const& point) const {
      Vector3 offset_pt = point-m_rotation_center-m_translation;
      Vector3 new_pt = m_rotation_inverse.rotate(offset_pt) + m_rotation_center;
      return new_pt;
    }
    
    to adjust the input ecef point coordinates previous to the original projection
    """

    # correct 3d locations
    R = np.linalg.inv(m_rotation_adj)
    # R = m_rotation_inv_adj
    new_pts = R @ (pts_3d.T - m_rotation_center_adj - m_translation_adj)
    new_pts += m_rotation_center_adj

    # apply original projection
    x, y, z = new_pts[0, :], new_pts[1, :], new_pts[2, :]
    lat, lon, alt = ba_utils.ecef_to_latlon_custom(x, y, z)
    col, row = rpc_init.projection(lon, lat, alt)

    return np.vstack([col, row]).T


def define_grid_from_3d_points(pts_3d_ecef):

    margin, n_samples = 500, 20
    x, y, z = pts_3d_ecef[:, 0], pts_3d_ecef[:, 1], pts_3d_ecef[:, 2]
    x_grid_coords = np.linspace(
        np.percentile(x, 5) - margin, np.percentile(x, 95) + margin, n_samples
    )
    y_grid_coords = np.linspace(
        np.percentile(y, 5) - margin, np.percentile(y, 95) + margin, n_samples
    )
    z_grid_coords = np.linspace(
        np.percentile(z, 5) - margin, np.percentile(z, 95) + margin, n_samples
    )
    x_grid, y_grid, z_grid = np.meshgrid(x_grid_coords, y_grid_coords, z_grid_coords)
    world_points = np.vstack((x_grid.ravel(), y_grid.ravel(), z_grid.ravel())).T

    return world_points


def ames_fit_corrected_rpc(
    image_fname, adjust_fname, pts_3d_ecef, rpc=None, fit_proj_matrix=True, verbose=True
):

    from bundle_adjust import rpc_fit, rpc_utils

    if rpc is None:
        rpc = rpcm.rpc_from_geotiff(image_fname)

    h, w = np.array(Image.open(image_fname)).shape
    world_points = define_grid_from_3d_points(pts_3d_ecef)

    lat, lon, alt = ba_utils.ecef_to_latlon_custom(
        world_points[:, 0], world_points[:, 1], world_points[:, 2]
    )
    world_points_lonlat = np.vstack([lon, lat, alt]).T

    m_translation, m_rotation, m_center = read_ASP_adjust_file(adjust_fname)
    image_points = ames_project_adjusted_points(
        rpc, m_rotation, m_translation, world_points, m_rotation_center=m_center
    )

    # image_points = ba_utils.apply_rpc_projection(rpc, world_points)

    # initialize rpc to fit
    import copy

    rpc_init = copy.copy(rpc)

    rows, cols = h, w
    rpc_init.row_offset = float(rows) / 2
    rpc_init.col_offset = float(cols) / 2
    rpc_init.lat_offset = min(lat) + (max(lat) - min(lat)) / 2
    rpc_init.lon_offset = min(lon) + (max(lon) - min(lon)) / 2
    rpc_init.alt_offset = min(alt) + (max(alt) - min(alt)) / 2
    rpc_init.row_scale = float(rows) / 2
    rpc_init.col_scale = float(cols) / 2
    rpc_init.lat_scale = (max(lat) - min(lat)) / 2
    rpc_init.lon_scale = (max(lon) - min(lon)) / 2
    rpc_init.alt_scale = (max(alt) - min(alt)) / 2

    rpc_calib = rpc_fit.weighted_lsq(rpc_init, image_points, world_points_lonlat)
    rmse_err = rpc_fit.calculate_RMSE_row_col(
        rpc_calib, world_points_lonlat, image_points
    )

    if fit_proj_matrix:

        cols, lins = image_points[:, 0], image_points[:, 1]
        x, y, z = (
            world_points[:, 0].tolist(),
            world_points[:, 1].tolist(),
            world_points[:, 2].tolist(),
        )

        P = rpc_utils.camera_matrix(world_points, image_points)

        if verbose:
            # compute the projection error made by the computed matrix P, on the
            # used learning points
            colPROJ = np.zeros(len(x))
            linPROJ = np.zeros(len(x))
            for i in range(len(x)):
                v = np.dot(P, [[x[i]], [y[i]], [z[i]], [1]])
                colPROJ[i] = v[0] / v[2]
                linPROJ[i] = v[1] / v[2]

            d_col, d_lin = cols - colPROJ, lins - linPROJ

            _, f = plt.subplots(1, 2, figsize=(10, 3))
            f[0].hist(np.sort(d_col), bins=40)
            f[1].hist(np.sort(d_lin), bins=40)
            plt.show()

            print("approximate_rpc_as_projective: (min, max, mean)")
            print("distance on cols:", np.min(d_col), np.max(d_col), np.mean(d_col))
            print("distance on rows:", np.min(d_lin), np.max(d_lin), np.mean(d_lin))
    else:
        P = None

    return rpc_calib, rmse_err, P


def check_pair(fname1, fname2, P1, P2, pts1, pts2):

    from bundle_adjust import ba_triangulate

    _, _, _, oC1 = ba_core.decompose_perspective_camera(P1)
    _, _, _, oC2 = ba_core.decompose_perspective_camera(P2)
    baseline = np.linalg.norm(oC1 - oC2)
    print(
        "baseline: {:.3f} {}".format(baseline, "GOOD" if baseline > 125000 else "BAD")
    )

    pts_3d = ba_triangulate.triangulate_list_of_matches(pts1.T, pts2.T, P1, P2)
    reproj_pts1 = ba_utils.apply_projection_matrix(P1, pts_3d)
    reproj_pts2 = ba_utils.apply_projection_matrix(P2, pts_3d)

    plot_reproj_pair_detail(fname1, fname2, pts1, pts2, reproj_pts1, reproj_pts2)


def plot_reproj_pair(fname1, fname2, features1, features2, reproj_pts1, reproj_pts2):

    err_1 = np.linalg.norm(reproj_pts1 - features1, axis=1)
    err_2 = np.linalg.norm(reproj_pts2 - features2, axis=1)
    err_1[err_1 < 1e-3] = 0
    err_2[err_2 < 1e-3] = 0

    print(
        "average reprojection error: {:.3f} / {:.3f}".format(
            np.mean(err_1), np.mean(err_2)
        )
    )

    sort_1 = np.argsort(err_1)[::-1]
    sort_2 = np.argsort(err_2)[::-1]

    features1, features2 = features1[sort_1], features2[sort_2]
    reproj_pts1, reproj_pts2 = reproj_pts1[sort_1], reproj_pts2[sort_2]
    err_1, err_2 = err_1[sort_1], err_2[sort_2]

    # reprojection error histograms for the selected image

    fig = plt.figure(figsize=(10, 3))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.title.set_text("Reprojection errors image 1")
    ax2.title.set_text("Reprojection errors image 2")
    ax1.plot(np.sort(err_1))
    ax1.axvline(x=err_1[err_1 < 1.0].shape[0], color="r", linestyle="-")
    ax2.plot(np.sort(err_2))
    ax2.axvline(x=err_2[err_2 < 1.0].shape[0], color="r", linestyle="-")
    plt.show()

    # plot images
    fig = plt.figure(figsize=(20, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.title.set_text("Reprojection errors image 1")
    ax2.title.set_text("Reprojection errors image 2")
    ax1.imshow(np.array(Image.open(fname1)), cmap="gray")
    ax2.imshow(np.array(Image.open(fname2)), cmap="gray")
    for k in range(min(3000, features1.shape[0])):
        # for k in range(min(10,features1.shape[0])):
        # before bundle adjustment
        ax1.plot(
            [features1[k, 0], reproj_pts1[k, 0]],
            [features1[k, 1], reproj_pts1[k, 1]],
            "r-",
            lw=3,
        )
        ax1.plot(*features1[k], "yx")
        # after bundle adjustment
        ax2.plot(
            [features2[k, 0], reproj_pts2[k, 0]],
            [features2[k, 1], reproj_pts2[k, 1]],
            "r-",
            lw=3,
        )
        ax2.plot(*features2[k], "yx")
    plt.show()


def plot_reproj_pair_detail(
    fname1, fname2, features1, features2, reproj_pts1, reproj_pts2
):

    from bundle_adjust.ba_timeseries import custom_equalization

    err_1 = np.linalg.norm(reproj_pts1 - features1, axis=1)
    err_2 = np.linalg.norm(reproj_pts2 - features2, axis=1)
    avg_residuals = (
        np.mean(
            np.hstack([abs(reproj_pts1 - features1), abs(reproj_pts2 - features2)]),
            axis=1,
        )
        / 2.0
    )

    err_1[err_1 < 1e-3] = 0
    err_2[err_2 < 1e-3] = 0

    print(
        "average reprojection error: {:.3f} / {:.3f}".format(
            np.mean(err_1), np.mean(err_2)
        )
    )
    print(
        "{} matches with reprojection error above 1".format(
            np.sum(1 * (((err_1 + err_2) / 2.0) > 1.0))
        )
    )

    sort_1 = np.argsort(err_1)[::-1]
    sort_2 = np.argsort(err_2)[::-1]

    features1, features2 = features1[sort_1], features2[sort_2]
    reproj_pts1, reproj_pts2 = reproj_pts1[sort_1], reproj_pts2[sort_2]
    err_1, err_2 = err_1[sort_1], err_2[sort_2]

    # reprojection error histograms for the selected image
    fig = plt.figure(figsize=(20, 3))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax1.title.set_text("Reprojection errors image 1 (L2 norm)")
    ax2.title.set_text("Reprojection errors image 2 (L2 norm)")
    ax3.title.set_text("Mean residual per observation (L1 norm)")
    ax1.plot(np.sort(err_1))
    ax1.axvline(x=err_1[err_1 < 1.0].shape[0], color="r", linestyle="-")
    ax2.plot(np.sort(err_2))
    ax2.axvline(x=err_2[err_2 < 1.0].shape[0], color="r", linestyle="-")
    ax3.plot(np.sort(avg_residuals))
    plt.show()

    fig = plt.figure(figsize=(30, 20))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.title.set_text("Reprojection errors image 1")
    ax2.title.set_text("Reprojection errors image 2")
    ax1.imshow(custom_equalization(np.array(Image.open(fname1))), cmap="gray")
    ax2.imshow(custom_equalization(np.array(Image.open(fname2))), cmap="gray")
    for k in range(min(3000, features1.shape[0])):
        # for k in range(min(10,features1.shape[0])):

        # before bundle adjustment
        ax1.plot(
            [features1[k, 0], reproj_pts1[k, 0]],
            [features1[k, 1], reproj_pts1[k, 1]],
            "r-",
            lw=3,
        )
        ax1.plot(*features1[k], "yx")
        # after bundle adjustment
        ax2.plot(
            [features2[k, 0], reproj_pts2[k, 0]],
            [features2[k, 1], reproj_pts2[k, 1]],
            "r-",
            lw=3,
        )
        ax2.plot(*features2[k], "yx")
    plt.show()

    return avg_residuals


def read_point_log_csv(fname):

    import csv

    with open(fname) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=" ", quotechar="|")
        lon, lat, height_above_datum, mean_residual, observations, outlier_flag = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for i, row in enumerate(spamreader):
            if i < 2:
                continue
            lon.append(float(row[0].replace(",", "")))
            lat.append(float(row[1].replace(",", "")))
            height_above_datum.append(float(row[2].replace(",", "")))
            mean_residual.append(float(row[3].replace(",", "")))
            observations.append(float(row[4].replace(",", "")))
            outlier_flag.append(True if row[5].replace(",", "") == "outlier" else False)

    """
    import pyproj
    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    x, y, z = pyproj.transform(lla, ecef, lon, lat, height_above_datum, radians=False)
    """

    x, y, z = ba_utils.latlon_to_ecef_custom(
        np.array(lat), np.array(lon), np.array(height_above_datum)
    )
    pts_3d_ecef = np.vstack([x, y, z]).T

    return pts_3d_ecef, mean_residual, observations, outlier_flag


def read_cnet_csv(fname):

    import csv

    with open(fname) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=" ", quotechar="|")
        lon, lat, alt, obs_2d = [], [], [], []
        for row in spamreader:

            # 3d coords
            lat.append(float(row[1]))
            lon.append(float(row[2]))
            alt.append(float(row[3]))

            # 2d projections
            geotif_fnames = row[7::5]
            col = list(map(np.float32, row[8::5]))
            lin = list(map(np.float32, row[9::5]))

            obs_2d.append(
                {
                    "proj": np.array([[x, y] for x, y in zip(col, lin)]),
                    "fnames": geotif_fnames,
                }
            )

    # the 3d coords are already corrected (use original rpcs to reproject them)
    x, y, z = ba_utils.latlon_to_ecef_custom(
        np.array(lat), np.array(lon), np.array(alt)
    )
    pts_3d_ecef = np.vstack([x, y, z]).T

    return pts_3d_ecef, obs_2d


def read_residuals_raw_pixels(fname):

    output = []
    with open(fname) as f:
        lines = f.readlines()
        line_counter = 0
        while line_counter < len(lines):
            image_fname = lines[line_counter].split(" ")[0].replace(",", "")
            n_points = int(lines[line_counter].split(" ")[1].replace("\n", ""))
            residuals = np.array(
                [
                    np.array(get_doubles_from_line(lines[line_counter + i + 1]))
                    for i in range(n_points)
                ]
            )
            line_counter += residuals.shape[0] + 1
            output.append({"fname": image_fname, "residuals": residuals})
    return output


def lonlat_limits_from_image_footprints(image_paths):

    from bundle_adjust import data_loader

    lonlat_geojson = data_loader.load_aoi_from_geotiffs(image_paths)
    lons, lats = np.array(lonlat_geojson["coordinates"][0]).T

    print("--lon-lat-limit <min_lon min_lat max_lon max_lat>")
    print(
        "--lon-lat-limit {:.6f} {:.6f} {:.6f} {:.6f}".format(
            lons.min(), lats.min(), lons.max(), lats.max()
        )
    )

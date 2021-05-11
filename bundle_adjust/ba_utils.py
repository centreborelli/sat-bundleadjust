"""
* Bundle Adjustment (BA) for 3D Reconstruction from Multi-Date Satellite Images
* Based on https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
* by Roger Mari <roger.mari@ens-paris-saclay.fr>
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from bundle_adjust import loader
from bundle_adjust import cam_utils


def rpc_rpcm_to_geotiff_format(input_dict):
    output_dict = {}

    output_dict["LINE_OFF"] = str(input_dict["row_offset"])
    output_dict["SAMP_OFF"] = str(input_dict["col_offset"])
    output_dict["LAT_OFF"] = str(input_dict["lat_offset"])
    output_dict["LONG_OFF"] = str(input_dict["lon_offset"])
    output_dict["HEIGHT_OFF"] = str(input_dict["alt_offset"])

    output_dict["LINE_SCALE"] = str(input_dict["row_scale"])
    output_dict["SAMP_SCALE"] = str(input_dict["col_scale"])
    output_dict["LAT_SCALE"] = str(input_dict["lat_scale"])
    output_dict["LONG_SCALE"] = str(input_dict["lon_scale"])
    output_dict["HEIGHT_SCALE"] = str(input_dict["alt_scale"])

    output_dict["LINE_NUM_COEFF"] = str(input_dict["row_num"])[1:-1].replace(",", "")
    output_dict["LINE_DEN_COEFF"] = str(input_dict["row_den"])[1:-1].replace(",", "")
    output_dict["SAMP_NUM_COEFF"] = str(input_dict["col_num"])[1:-1].replace(",", "")
    output_dict["SAMP_DEN_COEFF"] = str(input_dict["col_den"])[1:-1].replace(",", "")
    if "lon_num" in input_dict:
        output_dict["LON_NUM_COEFF"] = str(input_dict["lon_num"])[1:-1].replace(",", "")
        output_dict["LON_DEN_COEFF"] = str(input_dict["lon_den"])[1:-1].replace(",", "")
        output_dict["LAT_NUM_COEFF"] = str(input_dict["lat_num"])[1:-1].replace(",", "")
        output_dict["LAT_DEN_COEFF"] = str(input_dict["lat_den"])[1:-1].replace(",", "")

    return output_dict


def reestimate_lonlat_geojson_after_rpc_correction(initial_rpc, corrected_rpc, lonlat_geojson):

    import srtm4

    from bundle_adjust import geo_utils

    aoi_lons_init, aoi_lats_init = np.array(lonlat_geojson["coordinates"][0]).T
    alt = srtm4.srtm4(np.mean(aoi_lons_init), np.mean(aoi_lats_init))
    aoi_cols_init, aoi_rows_init = initial_rpc.projection(aoi_lons_init, aoi_lats_init, alt)
    aoi_lons_ba, aoi_lats_ba = corrected_rpc.localization(aoi_cols_init, aoi_rows_init, alt)
    lonlat_coords = np.vstack((aoi_lons_ba, aoi_lats_ba)).T
    lonlat_geojson = geo_utils.geojson_polygon(lonlat_coords)

    return lonlat_geojson


def update_geotiff_rpc(geotiff_path, rpc_model):
    from osgeo import gdal, gdalconst

    geotiff_dataset = gdal.Open(geotiff_path, gdalconst.GA_Update)
    geotiff_dataset.SetMetadata(rpc_rpcm_to_geotiff_format(rpc_model.__dict__), "RPC")
    del geotiff_dataset


def reproject_pts3d(cam_init, cam_ba, cam_model, obs2d, pts3d_init, pts3d_ba, image_fname=None, verbose=False):

    if image_fname is not None and not os.path.exists(image_fname):
        image_fname = None

    # open image if available
    image = loader.load_image_crops([image_fname], verbose=False)[0] if (image_fname is not None) else None
    # reprojections before bundle adjustment
    pts2d_init = project_pts3d(cam_init, cam_model, pts3d_init)
    # reprojections after bundle adjustment
    pts2d_ba = project_pts3d(cam_ba, cam_model, pts3d_ba)
    # compute average residuals and reprojection errors
    avg_residuals = np.mean(abs(pts2d_ba - obs2d), axis=1) / 2.0
    err_init = np.linalg.norm(pts2d_init - obs2d, axis=1)
    err_ba = np.linalg.norm(pts2d_ba - obs2d, axis=1)

    if verbose:

        print("path to image: {}".format(image_fname))
        args = [np.mean(err_init), np.median(err_init)]
        print("Reprojection error before BA (mean / median): {:.2f} / {:.2f}".format(*args))
        args = [np.mean(err_ba), np.median(err_ba)]
        print("Reprojection error after  BA (mean / median): {:.2f} / {:.2f}\n".format(*args))
        # reprojection error histograms for the selected image
        fig = plt.figure(figsize=(10, 3))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.title.set_text("Reprojection error before BA")
        ax2.title.set_text("Reprojection error after  BA")
        ax1.hist(err_init, bins=40)
        ax2.hist(err_ba, bins=40)
        # ax2.hist(err_ba, bins=40, range=(err_init.min(), err_init.max()))
        plt.show()

        plot = True
        if image is not None and plot:
            # warning: this is slow...
            # green crosses represent the observations from feature tracks seen in the image,
            # red vectors are the distance to the reprojected point locations.
            fig = plt.figure(figsize=(20, 6))
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            ax1.title.set_text("Before BA")
            ax2.title.set_text("After  BA")
            ax1.imshow(loader.custom_equalization(image), cmap="gray")
            ax2.imshow(loader.custom_equalization(image), cmap="gray")
            for k in range(min(3000, obs2d.shape[0])):
                # before bundle adjustment
                ax1.plot([obs2d[k, 0], pts2d_init[k, 0]], [obs2d[k, 1], pts2d_init[k, 1]], "r-", lw=3)
                ax1.plot(*obs2d[k], "yx")
                # after bundle adjustment
                ax2.plot([obs2d[k, 0], pts2d_ba[k, 0]], [obs2d[k, 1], pts2d_ba[k, 1]], "r-", lw=3)
                ax2.plot(*obs2d[k], "yx")
            plt.show()

    return pts2d_init, pts2d_ba, err_init, err_ba, avg_residuals


def project_pts3d(camera, cam_model, pts3d):
    """
    Project 3d points according to camera model
    Args:
        camera: either a projection matrix or a rpc model
        cam_model: accepted values are 'rpc', 'perspective' or 'affine'
        pts3d: Nx3 array of 3d points in ECEF coordinates
    Returns:
        pts2d: Nx2 array containing the 2d projections of pts3d
    """
    pts2d = cam_utils.apply_rpc_projection(camera, pts3d) if cam_model == "rpc" else cam_utils.apply_projection_matrix(camera, pts3d)
    return pts2d

def compute_relative_motion_between_projection_matrices(P1, P2, verbose=False):
    """
    Compute the relative motion between the extrinsic matrices of 2 perspective projection matrices
    This is useful to express the position of one camera in terms of the position of another camera
    Source: https://math.stackexchange.com/questions/709622/relative-camera-matrix-pose-from-global-camera-matrixes
    Args:
        P1: the projection matrix whose extrinsic matrix [R1 | t1] we want to express w.r.t a reference camera
        P2: the reference projection matrix, with extrinsic matrix [R2 | t2]
    Returns:
        ext21: a 4x4 matrix such that [R1 | t1] = [R2 | t2] @ ext21
    """

    # decompose input cameras
    k1, r1, t1, o1 = decompose_perspective_camera(P1)
    k2, r2, t2, o2 = decompose_perspective_camera(P2)
    # build extrinsic matrices
    ext1 = np.vstack([np.hstack([r1, t1[:, np.newaxis]]), np.array([0, 0, 0, 1], dtype=np.float32)])
    ext2 = np.vstack([np.hstack([r2, t2[:, np.newaxis]]), np.array([0, 0, 0, 1], dtype=np.float32)])
    # compute relative rotation and translation vector from camera 2 to camera 1
    r21 = r2.T @ r1  # i.e. r2 @ r21 = r1
    t21 = r2.T @ (t1 - t2)[:, np.newaxis]
    # build relative extrinsic matrix
    ext21 = np.vstack([np.hstack([r21, t21]), np.array([0, 0, 0, 1], dtype=np.float32)])
    if verbose:
        print("[R1 | t1] = [R2 | t2] @ [R21 | t21] ?", np.allclose(ext1, ext2 @ ext21))  # sanity check
        print("P1 = K1 @ [R2 | t2] @ [R21 | t21] ?", np.allclose(P1, k1 @ ext2[:3, :] @ ext21))  # sanity check
        deg = np.rad2deg(np.arccos((np.trace(r21) - 1) / 2))
        print("Found a rotation of {:.3f} degrees between both cameras\n".format(deg))
    return ext21


def rescale_projection_matrix(P, alpha):
    """
    Scale a projection matrix following an image resize
    Args:
        P: projection matrix to scale
        alpha: resize factor
               e.g. 2 if the image is upsampled by a factor of 2
                    1/2 if the image is downsampled by a factor of 2
    Returns:
        P_scaled: the scaled version of P by a factor alpha
    """
    s = float(alpha)
    P_scaled = np.array([[s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, 1.0]]) @ P
    return P_scaled


def rescale_RPC(rpc, alpha):
    """
    Scale a rpc model following an image resize
    Args:
        rpc: rpc model to scale
        alpha: resize factor
               e.g. 2 if the image is upsampled by a factor of 2
                    1/2 if the image is downsampled by a factor of 2
    Returns:
        rpc_scaled: the scaled version of P by a factor alpha
    """
    import copy

    rpc_scaled = copy.copy(rpc)
    rpc_scaled.row_scale *= float(alpha)
    rpc_scaled.col_scale *= float(alpha)
    rpc_scaled.row_offset *= float(alpha)
    rpc_scaled.col_offset *= float(alpha)
    return rpc_scaled


# compute the union of all pair intersections in a list of lonlat_geojson
def get_aoi_where_at_least_two_lonlat_geojson_overlap(lonlat_geojson_list):

    from bundle_adjust import geo_utils

    from itertools import combinations

    from shapely.geometry import shape
    from shapely.ops import cascaded_union

    utm_zone = geo_utils.utm_zonestring_from_lonlat_geojson(lonlat_geojson_list[0])
    utm_geojson_list = [geo_utils.utm_geojson_from_lonlat_geojson(x) for x in lonlat_geojson_list]

    geoms = [shape(g) for g in utm_geojson_list]
    geoms = [a.intersection(b) for a, b in combinations(geoms, 2)]
    combined_borders_shapely = cascaded_union([geom if geom.is_valid else geom.buffer(0) for geom in geoms])
    vertices = np.array(combined_borders_shapely.boundary.coords.xy).T[:-1, :]
    utm_geojson = geo_utils.geojson_polygon(vertices)
    return geo_utils.lonlat_geojson_from_utm_geojson(utm_geojson, utm_zone)


def epsg_from_utm_zone(utm_zone, datum="WGS84"):
    """
    convert from geodetic (lat, lon, alt) to geocentric coordinates (x, y, z)
    """
    from pyproj import CRS

    args = [utm_zone[:2], "+south" if utm_zone[-1] == "S" else "+north", datum]
    crs = CRS.from_proj4("+proj=utm +zone={} {} +datum={}".format(*args))
    return crs.to_epsg()


# display lonlat_geojson list over map
def display_lonlat_geojson_list_over_map(lonlat_geojson_list, zoom_factor=14):
    from bundle_adjust import vistools

    mymap = vistools.clickablemap(zoom=zoom_factor)
    for aoi in lonlat_geojson_list:
        mymap.add_GeoJSON(aoi)
    mymap.center = lonlat_geojson_list[int(len(lonlat_geojson_list) / 2)]["center"][::-1]
    display(mymap)


def load_pairs_from_same_date_and_next_dates(timeline, timeline_indices, next_dates=1, intra_date=True):
    """
    Given some timeline_indices of a certain timeline, this function defines those pairs of images
    composed by (1) nodes that belong to the same acquisition date
                (2) nodes between each acquisition date and the next N dates
    """
    timeline_indices = np.array(timeline_indices)

    def count_cams(timeline_indices):
        return np.sum([timeline[t_idx]["n_images"] for t_idx in timeline_indices])

    # get pairs within the current date and between this date and the next
    init_pairs, cams_so_far, dates_left = [], 0, len(timeline_indices)
    for k, t_idx in enumerate(timeline_indices):
        cams_current_date = timeline[t_idx]["n_images"]
        if intra_date:
            # (1) pairs within the current date
            for cam_i in np.arange(cams_so_far, cams_so_far + cams_current_date):
                for cam_j in np.arange(cam_i + 1, cams_so_far + cams_current_date):
                    init_pairs.append((int(cam_i), int(cam_j)))
        # (2) pairs between the current date and the next N dates
        dates_left -= 1
        for next_date in np.arange(1, min(next_dates + 1, dates_left + 1)):
            next_date_t_idx = timeline_indices[k + next_date]
            cams_next_date = timeline[next_date_t_idx]["n_images"]
            cams_until_next_date = count_cams(timeline_indices[: k + next_date])
            for cam_i in np.arange(cams_so_far, cams_so_far + cams_current_date):
                for cam_j in np.arange(cams_until_next_date, cams_until_next_date + cams_next_date):
                    init_pairs.append((int(cam_i), int(cam_j)))
        cams_so_far += cams_current_date
    return init_pairs







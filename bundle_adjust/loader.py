"""
A Generic Bundle Adjustment Methodology for Indirect RPC Model Refinement of Satellite Imagery
author: Roger Mari <roger.mari@ens-paris-saclay.fr>
year: 2021

This script consists of a series of functions dedicated to load and store data on the disk
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

import rasterio
import warnings
import json
import rpcm

from bundle_adjust import cam_utils, geo_utils

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

def flush_print(input_string):
    print(input_string, flush=True)


def display_dict(d):
    """
    Displays the input dictionary d
    """
    max_k_len = len(sorted(d.keys(), key=lambda i: len(i))[::-1][0])
    for k in d.keys():
        print("    - {}:{}{}".format(k, "".join([" "] * (max_k_len - len(k) + 2)), d[k]))
    print("\n")


def read_image_size(im_fname):
    """
    Reads image width and height without opening the file
    useful when dealing with huge images
    """
    with rasterio.open(im_fname) as f:
        h, w = f.height, f.width
    return h, w


def get_time_in_hours_mins_secs(input_seconds):
    """
    Takes a float representing a time measure in seconds
    Returns a string with the time measure expressed in hours:minutes:seconds
    """
    hours, rem = divmod(input_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)


def add_suffix_to_fname(src_fname, suffix):
    """
    Adds a string suffix at the end of the src_fname path to file, keeping the file extension
    """
    src_basename = os.path.basename(src_fname)
    file_id, file_extension = os.path.splitext(src_basename)
    dst_fname = src_fname.replace(src_basename, file_id + suffix + file_extension)
    return dst_fname


def get_id(fname):
    """
    Gets the basename without extension of a path to file
    """
    return os.path.splitext(os.path.basename(fname))[0]


def save_dict_to_json(input_dict, output_json_fname):
    """
    Saves a python dictionary to a .json file
    """
    with open(output_json_fname, "w") as f:
        json.dump(input_dict, f, indent=2)


def load_dict_from_json(input_json_fname):
    """
    Reads a .json file into a python dictionary
    """
    with open(input_json_fname) as f:
        output_dict = json.load(f)
    return output_dict


def load_geotiff_lonlat_footprints(geotiff_paths, rpcs=None, crop_offsets=None):
    """
    Takes a list of geotiff paths and, optionally, their rpcs and crop offsets (in case the images are crops)
    If the rpcs are not specified in the input, they will be read from the geotiffs
    Outputs a list of geojson polygons delimiting the geographic footprint of the images in lon-lat coordinates
    """
    if crop_offsets is None:
        crop_offsets = []
        for path_to_geotiff in geotiff_paths:
            h, w = read_image_size(path_to_geotiff)
            crop_offsets.append({"col0": 0.0, "row0": 0.0, "width": w, "height": h})
    if rpcs is None:
        rpcs = [rpcm.rpc_from_geotiff(path_to_geotiff) for path_to_geotiff in geotiff_paths]

    lonlat_geotiff_footprints = []
    # get srtm4 of the middle point in each geotiff crop
    import srtm4

    lonslats = np.array([[rpc.lon_offset, rpc.lat_offset] for rpc in rpcs])
    alts = srtm4.srtm4(lonslats[:, 0], lonslats[:, 1])
    fails = 0
    import warnings

    warnings.filterwarnings("ignore")
    for im_idx, (rpc, offset) in enumerate(zip(rpcs, crop_offsets)):
        try:
            lonlat_geotiff_footprints.append(geo_utils.lonlat_geojson_from_geotiff_crop(rpc, offset, z=alts[im_idx]))
        except:
            fails += 1
    if fails > 0:
        args = [fails, len(geotiff_paths)]
        print("\nWARNING: {}/{} fails loading geotiff footprints (rpc localization max iter error)\n".format(*args))
    return lonlat_geotiff_footprints, alts


def load_aoi_from_multiple_images(images, verbose=False):
    """
    Reads all footprints of a series of geotiff files and returns a geojson, in lon-lat coordinates,
    consisting of the union of all footprints
    """
    if verbose:
        print("Defined aoi from union of all geotiff footprints")
    return geo_utils.combine_lonlat_geojson_borders([im.lonlat_geojson for im in images])


def mask_from_shapely_polygons(polygons, im_size):
    """
    Computes a binary mask from a list of shapely polygons or multipolygon list
    with 1 inside the polygons and 0 outside. im_size is a tuple = (height, width)
    Note: polygon coords have to be specified within the range of max rows and cols determined by im_size (h, w)
    """
    import cv2

    img_mask = np.zeros(im_size, np.uint8)
    # function to round and convert coordinates to int
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
    interiors = [int_coords(pi.coords) for poly in polygons for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask


def get_binary_mask_from_aoi_lonlat_within_image(height, width, geotiff_rpc, aoi_lonlat):
    """
    Computes a binary mask of an area of interest within the limits of a geotiff image
    with 1 in those points inside the area of interest and 0 in those points outisde of it
    """
    import srtm4

    lons, lats = np.array(aoi_lonlat["coordinates"][0]).T
    alts = srtm4.srtm4(lons, lats)
    cols, rows = geotiff_rpc.projection(lons, lats, alts)
    poly_verts_colrow = np.vstack((cols, rows)).T
    geojson_poly = geo_utils.geojson_polygon(poly_verts_colrow)
    shapely_poly = geo_utils.geojson_to_shapely_polygon(geojson_poly)
    mask = mask_from_shapely_polygons([shapely_poly], (height, width))
    return mask


def custom_equalization(im, mask=None, clip=True, percentiles=5):
    """
    Given the numpy array im, returns a numpy array im_eq with the equalized version of the image between 0-255.
    Optional boolean array mask can be used to indicate a subset of values to use for the equalization.
    Optional boolean clip combined with the integer value in percentiles can be used to restrict the
    input range of values taken into account between the x-th and the (100-x)-th percentiles.
    """
    valid_domain = mask > 0 if (mask is not None) else np.isfinite(im)  # compute mask
    if clip:
        mi, ma = np.percentile(im[valid_domain], (percentiles, 100 - percentiles))
    else:
        mi, ma = im[valid_domain].min(), im[valid_domain].max()
    im = np.minimum(np.maximum(im, mi), ma)  # clip
    im = (im - mi) / (ma - mi) * 255.0  # scale
    return im


def load_image(path_to_geotiff, offset=None, equalize=False):
    """
    Reads an input geotiff
    It is possible to specify a window to read a specific region of the image
    """
    if offset is None:
        with rasterio.open(path_to_geotiff) as src:
            im = src.read().astype(float)
    else:
        y0, x0, h, w = offset["row0"], offset["col0"], offset["height"], offset["width"]
        with rasterio.open(path_to_geotiff) as src:
            im = src.read(window=((y0, y0 + h), (x0, x0 + w))).squeeze().astype(float)

    # we only work with 1-band images
    if len(im.shape) > 2:
        im = np.mean(im, axis=np.argmin(im.shape)) # use average of all bands (assume number of bands = smallest dim)
    if equalize:
        im = custom_equalization(im)
    return im


def load_offsets(rpcs, aoi):
    """
    Loads an offset for each rpc model, which delimits the image region corresponding to the area of interest
    an offset is a dictionary with fields "col0", "row0", "width", "height"
    i.e. aoi_as_seen_in_image = image[row0: row0 + height, col0 : col0 + width]
    """

    offsets = []
    for rpc in rpcs:
        # get the altitude of the center of the AOI
        import srtm4

        lon, lat = aoi["center"]
        alt = srtm4.srtm4(lon, lat)
        # project aoi and crop
        lons, lats = np.array(aoi["coordinates"][0]).T
        x, y = rpc.projection(lons, lats, alt)
        x_min, x_max, y_min, y_max = min(x), max(x), min(y), max(y)
        x0, y0, w, h = int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)
        offsets.append({"col0": x0, "row0": y0, "width": w, "height": h})
    return offsets


def save_rpcs(filenames, rpcs):
    """
    Writes a series of rpc models to the specified filenames
    """
    for fn, rpc in zip(filenames, rpcs):
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        rpc.write_to_file(fn)


def load_rpcs_from_dir(image_fnames_list, rpc_dir, suffix="", extension="rpc", verbose=True):
    """
    Loads rpcs from rpc files stored in a common directory
    """
    rpcs = []
    for im_idx, fname in enumerate(image_fnames_list):
        rpc_basename = "{}.{}".format(get_id(add_suffix_to_fname(fname, suffix)), extension)
        path_to_rpc = os.path.join(rpc_dir, rpc_basename)
        rpcs.append(rpcm.rpc_from_rpc_file(path_to_rpc))
    if verbose:
        flush_print("Loaded {} rpcs".format(len(image_fnames_list)))
    return rpcs


def save_projection_matrices(filenames, projection_matrices, crop_offsets):
    """
    Writes a series of projection matrices and their corresponding crop offsets to the specified filenames
    """
    for fn, P, offset in zip(filenames, projection_matrices, crop_offsets):
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        to_write = {
            "P": [P[0, :].tolist(), P[1, :].tolist(), P[2, :].tolist()],
            "height": int(offset["height"]),
            "width": int(offset["width"]),
            "col_offset": int(offset["col0"]),
            "row_offset": int(offset["row0"]),
        }
        save_dict_to_json(to_write, fn)


def load_matrices_from_dir(image_fnames_list, P_dir, suffix="pinhole_adj", verbose=True):
    """
    Loads projection matrices from json files stored in a common directory
    """
    proj_matrices = []
    for im_idx, fname in enumerate(image_fnames_list):
        path_to_P = os.path.join(P_dir, "{}_{}.json".format(get_id(fname), suffix))
        P = load_dict_from_json(path_to_P)["P"]
        proj_matrices.append(P / P[2, 3])
    if verbose:
        print("Loaded {} projection matrices".format(len(image_fnames_list)))
    return proj_matrices


def load_offsets_from_dir(image_fnames_list, P_dir, suffix="pinhole_adj", verbose=True):
    """
    Loads offsets from json files stored in a common directory
    """
    crop_offsets = []
    for im_idx, fname in enumerate(image_fnames_list):
        path_to_P = os.path.join(P_dir, "{}_{}.json".format(get_id(fname), suffix))
        d = load_dict_from_json(path_to_P)
        crop_offsets.append(
            {
                "col0": d["col_offset"],
                "row0": d["col_offset"],
                "width": d["width"],
                "height": d["height"],
            }
        )
    if verbose:
        print("Loaded {} crop offsets".format(len(image_fnames_list)))
    return crop_offsets


def save_list_of_pairs(path_to_npy, list_of_pairs):
    """
    Save a list of pairs to a .npy file
    list of pairs is a list of tuples, but is saved as a 2d array with 2 columns (one row per pair)
    """
    np.save(path_to_npy, np.array(list_of_pairs))


def load_list_of_pairs(path_to_npy):
    """
    Opposite operation of save_list_of_pairs
    """
    array_t = np.load(path_to_npy).T.astype(int)
    return list(zip(array_t[0], array_t[1]))


def save_list_of_paths(path_to_txt, list_of_paths):
    """
    Save a list of strings to a txt (one string per line)
    """
    with open(path_to_txt, "w") as f:
        for p in list_of_paths:
            f.write("%s\n" % p)


def load_list_of_paths(path_to_txt):
    """
    Read a list of strings from a txt (one string per line)
    """
    with open(path_to_txt, "r") as f:
        content = f.readlines()
    return [x.strip() for x in content]


def save_geojson(path_to_json, geojson):
    """
    Save a geojson polygon to a .json file
    """
    geojson_to_save = {}
    geojson_to_save["coordinates"] = geojson["coordinates"]
    geojson_to_save["type"] = "Polygon"
    save_dict_to_json(geojson_to_save, path_to_json)


def load_geojson(path_to_json):
    """
    Read a geojson polygon from a .json file
    """
    d = load_dict_from_json(path_to_json)
    return geo_utils.geojson_polygon(np.array(d["coordinates"][0]))


def read_point_cloud_ply(filename):
    """
    Reads a point cloud from a ply file
    the header of the file is expected to be as below, with the vertices coordinates listed after

    ply
    format ascii 1.0
    element vertex 541636
    property float x
    property float y
    property float z
    end_header
    """
    import re

    with open(filename, "r") as f_in:
        lines = f_in.readlines()
        content = [x.strip() for x in lines]
        n_pts = len(content) - 7
        pt_cloud = np.zeros((n_pts, 3))
        for i in range(n_pts):
            coords = re.findall(r"[-+]?\d*\.\d+|\d+", content[i + 7])
            pt_cloud[i, :] = np.array([float(coords[0]), float(coords[1]), float(coords[2])])
    return pt_cloud


def write_point_cloud_ply(filename, point_cloud, color=np.array([None, None, None])):
    """
    Writes a point cloud of N 3d points to a ply file
    The color of the points can be specified using color (3 valued vector with the rgb values)
    """
    with open(filename, "w") as f_out:
        n_points = point_cloud.shape[0]
        # write output ply file with the point cloud
        f_out.write("ply\n")
        f_out.write("format ascii 1.0\n")
        f_out.write("element vertex {}\n".format(n_points))
        f_out.write("property float x\nproperty float y\nproperty float z\n")
        if not (color[0] is None and color[1] is None and color[2] is None):
            f_out.write("property uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar alpha\n")
            f_out.write("element face 0\nproperty list uchar int vertex_indices\n")
        f_out.write("end_header\n")
        # write 3d points
        for i in range(n_points):
            p_3d = point_cloud[i, :]
            f_out.write("{} {} {}".format(p_3d[0], p_3d[1], p_3d[2]))
            if not (color[0] is None and color[1] is None and color[2] is None):
                f_out.write(" {} {} {} 255".format(color[0], color[1], color[2]))
            f_out.write("\n")


def save_predefined_matches(input_dir, output_dir):
    """
    Converts the results of pairwise matching using FeatureTracksPipeline to the predefined matches format
    The predefined matches format stores (1) the image coordinates and the scale of detected keypoints
    (orientation and sift descriptor are discarded) + (2) the matches.npy file + (3) the filenames.txt file
    """
    import glob

    predefined_matches_dir = os.path.join(output_dir, "predefined_matches")
    features_fnames = glob.glob(input_dir + "/features/*.npy")
    os.makedirs(predefined_matches_dir + "/keypoints", exist_ok=True)
    for fn in features_fnames:
        features_light = np.load(fn)[:, :3]  # we take only the first 3 columns corresponding to (col, row, scale)
        np.save(fn.replace(input_dir + "/features/", predefined_matches_dir + "/keypoints/"), features_light)
    shutil.copyfile(os.path.join(input_dir, "matches.npy"), predefined_matches_dir + "/matches.npy")
    shutil.copyfile(os.path.join(input_dir, "filenames.txt"), predefined_matches_dir + "/filenames.txt")


#--- functions that generate output illustrations ---


def write_georeferenced_raster_utm_bbox(img_path, raster, utm_bbx, epsg, resolution):
    """
    Writes a georeferenced raster to a tif image file
    The bounds are fixed by a utm bounding box
    it is also necessary to know the resolution of the image and the epsg code
    """
    import rasterio
    from rasterio.crs import CRS

    west, south, east, north = utm_bbx["xmin"], utm_bbx["ymin"], utm_bbx["xmax"], utm_bbx["ymax"]
    height, width = geo_utils.utm_bbox_shape(utm_bbx, resolution)

    profile = dict()
    profile["driver"] = "GTiff"
    profile["nodata"] = np.nan
    profile["dtype"] = np.float32
    profile["height"] = height
    profile["width"] = width
    profile["count"] = 1
    profile["crs"] = CRS.from_epsg(epsg)
    profile["transform"] = rasterio.transform.from_bounds(west, south, east, north, width, height)
    with rasterio.open(img_path, "w", **profile) as f:
        f.write(raster.astype(np.float32), 1)


def draw_image_footprints(img_path, lonlat_footprints, aoi_lonlat, plot=False):
    """
    Saves a png image with the contours of the image footprints and the contours of the aoi
    this is useful to check the disposition of the input images and how they cover the aoi
    """
    utm_footprints = [geo_utils.utm_geojson_from_lonlat_geojson(x) for x in lonlat_footprints]
    utm_geojson = geo_utils.combine_utm_geojson_borders(utm_footprints)
    easts, norths = np.array(utm_geojson["coordinates"][0]).T
    utm_bbx = {"xmin": easts.min(), "xmax": easts.max(), "ymin": norths.min(), "ymax": norths.max()}
    resolution = 1.0
    aoi_utm_geojson = geo_utils.utm_geojson_from_lonlat_geojson(aoi_lonlat)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.invert_yaxis()
    ax.axis("equal")
    ax.axis("off")
    for f in utm_footprints:
        tmp = np.array(f["coordinates"][0])
        tmp = geo_utils.compute_relative_utm_coords_inside_utm_bbx(tmp, utm_bbx, resolution)
        tmp = geo_utils.geojson_to_shapely_polygon(geo_utils.geojson_polygon(tmp))
        plt.plot(*tmp.exterior.xy, color="black", linewidth=1.0)
    tmp = np.array(aoi_utm_geojson["coordinates"][0])
    tmp = geo_utils.compute_relative_utm_coords_inside_utm_bbx(tmp, utm_bbx, resolution)
    tmp = geo_utils.geojson_to_shapely_polygon(geo_utils.geojson_polygon(tmp))
    plt.plot(*tmp.exterior.xy, color="red", linewidth=3.0)
    if plot:
        plt.show()
    else:
        plt.savefig(img_path, bbox_inches="tight")

"""
Bundle Adjustment for 3D Reconstruction from Multi-Date Satellite Images
This script contains functions used to load and write data of all sorts:
geotiffs, rpcs, projection matrices, areas of interest, etc.
It was created to make it easier to load and write stuff to and from Scene objects to bundle adjust
by Roger Mari <roger.mari@ens-paris-saclay.fr>
"""

import datetime
import glob
import json
import os
import pickle
import sys

import numpy as np
import rasterio
import rpcm

from bundle_adjust import camera_utils, geotools


def display_dict(d):
    """
    Displays the input dictionary d
    """
    max_k_len = len(sorted(d.keys(), key=lambda k: len(k))[::-1][0])
    for k in d.keys():
        print(
            "    - {}:{}{}".format(k, "".join([" "] * (max_k_len - len(k) + 2)), d[k])
        )
    print("\n")


def save_pickle(fname, data):
    """
    Saves variables into a pickle file
    """

    pickle_out = open(fname, "wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()


def load_pickle(fname):
    """
    Saves variables into a pickle file
    """

    pickle_in = open(fname, "rb")
    return pickle.load(pickle_in)


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
    dst_fname = src_fname.replace(
        "/" + src_basename, "/" + file_id + "_" + suffix + file_extension
    )
    return dst_fname


def get_id(fname):
    """
    Gets the basename without extension of a path to file
    """
    return os.path.splitext(os.path.basename(fname))[0]


def get_acquisition_date(geotiff_path):
    """
    Reads the acquisition date of a geotiff
    """
    with rasterio.open(geotiff_path) as src:
        if "TIFFTAG_DATETIME" in src.tags().keys():
            date_string = src.tags()["TIFFTAG_DATETIME"]
            dt = datetime.datetime.strptime(date_string, "%Y:%m:%d %H:%M:%S")
        else:
            # temporary fix in case the previous tag is missing
            # get datetime from skysat geotiff identifier
            date_string = os.path.basename(geotiff_path)[:15]
            dt = datetime.datetime.strptime(date_string, "%Y%m%d_%H%M%S")
    return dt


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


def load_scene_from_s2p_configs(
    geotiff_dir, s2p_configs_dir, output_dir, rpc_src="s2p_configs", geotiff_label=None
):
    """
    This function loads the timeline and area of interest of a Scene to bundle adjust from a series of s2p config.json
    Args:
        geotiff_dir (string): path containing all the geotiff images to be considered
        s2p_configs_dir (string): path containing all s2p initial config.json files
        output_dir (string): path where all initial rpcs will be written
        rpc_src (string): indicates where to read the initial rpcs from, i.e. 'geotiff', 'json', or 'txt' files
        geotiff_label (string): if specified, only the geotiff filenames containing this string are considered
    Returns:
        timeline: list of dicts, where each dict groups a set of geotiffs with a common acquisition date
        aoi_lonlat: geojson delimiting the area of interest of the scene in lon lat coordinates
    """

    all_config_fnames = []
    all_images_fnames = []
    all_images_rpcs = []
    all_images_datetimes = []

    # get all image fnames used by s2p and their rpcs

    geotiff_paths = sorted(glob.glob(os.path.join(geotiff_dir, '**/*.tif'), recursive=True))
    if geotiff_label is None:
        geotiff_basenames = [os.path.basename(fn) for fn in geotiff_paths]
    else:
        geotiff_basenames = [
            os.path.basename(fn) for fn in geotiff_paths if geotiff_label in fn
        ]

    config_fnames = glob.glob(
        os.path.join(s2p_configs_dir, "**/config.json"), recursive=True
    )

    seen_images = []
    for fname in config_fnames:
        d = load_dict_from_json(fname)

        # check both images listed in the config.json are available in the geotiff_dir
        basename_l = os.path.basename(d["images"][0]["img"])
        basename_r = os.path.basename(d["images"][1]["img"])
        if basename_l not in geotiff_basenames or basename_r not in geotiff_basenames:
            continue

        # load image filenames and rpcs from config.json
        for view in d["images"]:
            img_basename = os.path.basename(view["img"])
            if img_basename not in seen_images:
                seen_images.append(img_basename)

                img_geotiff_path = glob.glob(
                    "{}/**/{}".format(geotiff_dir, img_basename), recursive=True
                )[0]

                # load rpc
                if rpc_src == "s2p_configs":
                    rpc = rpcm.RPCModel(view["rpc"], dict_format="rpcm")
                elif rpc_src == "geotiff":
                    rpc = rpcm.rpc_from_geotiff(img_geotiff_path)
                elif rpc_src == 'txt':
                    rpc = rpcm.rpc_from_rpc_file(os.path.join(geotiff_dir, get_id(img_geotiff_path) + '_RPC.TXT'))
                else:
                    raise ValueError("Unknown rpc_src value: {}".format(rpc_src))

                all_images_fnames.append(img_geotiff_path)
                all_images_rpcs.append(rpc)
                all_images_datetimes.append(get_acquisition_date(img_geotiff_path))
        all_config_fnames.append(fname)

    # copy initial rpcs to a folder in the output directory so it is easier to access them
    all_rpc_fnames = [
        os.path.join(output_dir, "RPC_init/{}_RPC.txt".format(get_id(fn)))
        for fn in all_images_fnames
    ]
    save_rpcs(all_rpc_fnames, all_images_rpcs)

    # define timeline and aoi
    timeline = group_files_by_date(all_images_datetimes, all_images_fnames)
    aoi_lonlat = load_aoi_from_s2p_configs(all_config_fnames)

    return timeline, aoi_lonlat


def load_scene_from_geotiff_dir(
    geotiff_dir, output_dir, rpc_src="geotiff", geotiff_label=None
):
    """
    This function loads the timeline and area of interest of a Scene to bundle adjust from a geotiff_directory
    Args:
        geotiff_dir (string): path containing all the geotiff images to be considered
        output_dir (string): path where all initial rpcs will be written
        rpc_src (string): indicates where to read the initial rpcs from, i.e. 'geotiff', 'json', or 'txt' files
        geotiff_label (string): if specified, only the geotiff filenames containing this string are considered
    Returns:
        timeline: list of dicts, where each dict groups a set of geotiffs with a common acquisition date
        aoi_lonlat: geojson delimiting the area of interest of the scene in lon lat coordinates
    """

    all_images_fnames = []
    all_images_rpcs = []
    all_images_datetimes = []

    geotiff_paths = sorted(glob.glob(os.path.join(geotiff_dir, '**/*.tif'), recursive=True))
    if geotiff_label is not None:
        geotiff_paths = [
            os.path.basename(fn) for fn in geotiff_paths if geotiff_label in fn
        ]

    for tif_fname in geotiff_paths:

        f_id = get_id(tif_fname)
        tif_dir = os.path.dirname(tif_fname)

        # load rpc
        if rpc_src == "geotiff":
            rpc = rpcm.rpc_from_geotiff(tif_fname)
        elif rpc_src == "json":
            with open(os.path.join(tif_dir, f_id + ".json")) as f:
                d = json.load(f)
            rpc = rpcm.RPCModel(d, dict_format="rpcm")
        elif rpc_src == "txt":
            rpc = rpcm.rpc_from_rpc_file(os.path.join(tif_dir, f_id + "_RPC.TXT"))
        else:
            raise ValueError("Unknown rpc_src value: {}".format(rpc_src))

        all_images_fnames.append(tif_fname)
        all_images_rpcs.append(rpc)
        all_images_datetimes.append(get_acquisition_date(tif_fname))

    # copy initial rpcs
    all_rpc_fnames = [
        os.path.join(output_dir, "RPC_init/{}_RPC.txt".format(get_id(fn)))
        for fn in all_images_fnames
    ]
    save_rpcs(all_rpc_fnames, all_images_rpcs)

    # define timeline and aoi
    timeline = group_files_by_date(all_images_datetimes, all_images_fnames)
    aoi_lonlat = load_aoi_from_geotiffs(all_images_fnames, rpcs=all_images_rpcs)

    return timeline, aoi_lonlat


def load_aoi_from_geotiffs(geotiff_paths, rpcs=None, crop_offsets=None):
    """
    Reads all footprints of a series of geotiff files and returns a geojson, in lon lat coordinates,
    consisting of the union of all footprints
    """
    if crop_offsets is None:
        crop_offsets = []
        for path_to_geotiff in geotiff_paths:
            h, w = read_image_size(path_to_geotiff)
            crop_offsets.append({"col0": 0.0, "row0": 0.0, "width": w, "height": h})
    if rpcs is None:
        rpcs = [
            rpcm.rpc_from_geotiff(path_to_geotiff) for path_to_geotiff in geotiff_paths
        ]

    n = len(geotiff_paths)
    lonlat_geotiff_footprints = []
    for path_to_geotiff, rpc, offset, im_idx in zip(
        geotiff_paths, rpcs, crop_offsets, np.arange(n)
    ):
        lonlat_geotiff_footprints.append(
            geotools.lonlat_geojson_from_geotiff_crop(rpc, offset)
        )
        if sys.stdout.name == "stdout":
            print(
                "\rDefining aoi from union of all geotiff footprints... {}/{}".format(
                    im_idx + 1, n
                ),
                end="\r",
            )
    print(
        "\rDefining aoi from union of all geotiff footprints... {}/{}\n".format(
            im_idx + 1, n
        )
    )

    return geotools.combine_lonlat_geojson_borders(lonlat_geotiff_footprints)


def load_aoi_from_s2p_configs(s2p_config_fnames):
    """
    Reads all the roi_geojson (i.e. areas of interest defined in lon lat coordinates)
    within an input list of s2p config.json files and returns the roi_geojson resulting from the union of all
    """
    n = len(s2p_config_fnames)
    config_aois = []
    for config_idx, fname in enumerate(s2p_config_fnames):
        d = load_dict_from_json(fname)
        current_aoi = d["roi_geojson"]
        current_aoi["center"] = np.mean(
            current_aoi["coordinates"][0][:4], axis=0
        ).tolist()
        config_aois.append(current_aoi)
        if sys.stdout.name == "stdout":
            print(
                "\rDefining aoi from s2p config.json files... {}/{}".format(
                    config_idx + 1, n
                ),
                end="\r",
            )
    print(
        "\rDefining aoi from s2p config.json files... {}/{}\n".format(
            config_idx + 1, n
        ),
        flush=True,
    )
    return geotools.combine_lonlat_geojson_borders(config_aois)


def load_pairs_from_same_date_and_next_dates(
    timeline, timeline_indices, next_dates=1, intra_date=True
):
    """
    Given some timeline_indices of a certain timeline, this function defines those pairs of images
    composed by (1) nodes that belong to the same acquisition date
                (2) nodes between each acquisition date and the next N dates
    """
    timeline_indices = np.array(timeline_indices)

    def count_cams(timeline_indices):
        return np.sum([timeline[t_idx]["n_images"] for t_idx in timeline_indices])

    # get pairs within the current date and between this date and the next
    init_pairs, cam_so_far, dates_left = [], 0, len(timeline_indices)
    for k, t_idx in enumerate(timeline_indices):
        cam_current_date = timeline[t_idx]["n_images"]
        if intra_date:
            # (1) pairs within the current date
            for cam_i in np.arange(cam_so_far, cam_so_far + cam_current_date):
                for cam_j in np.arange(cam_i + 1, cam_so_far + cam_current_date):
                    init_pairs.append((int(cam_i), int(cam_j)))
        # (2) pairs between the current date and the next N dates
        dates_left -= 1
        for next_date in np.arange(1, min(next_dates + 1, dates_left + 1)):
            next_date_t_idx = timeline_indices[k + next_date]
            cam_next_date = timeline[next_date_t_idx]["n_images"]
            for cam_i in np.arange(cam_so_far, cam_so_far + cam_current_date):
                for cam_j in np.arange(
                    count_cams(timeline_indices[: k + next_date]),
                    count_cams(timeline_indices[: k + next_date]) + cam_next_date,
                ):
                    init_pairs.append((int(cam_i), int(cam_j)))
        cam_so_far += cam_current_date
    return init_pairs


def group_files_by_date(datetimes, image_fnames):
    """
    This function picks a list of image fnames and their acquisition dates,
    and returns the timeline of a scene to bundle adjust (class Scene fromfrom ba_timeseries.py)
    Each timeline instance is a group of images with a common acquisition date (i.e. less than 30 mins difference)
    """

    # sort images according to the acquisition date
    sorted_indices = np.argsort(datetimes)
    sorted_datetimes = np.array(datetimes)[sorted_indices].tolist()
    sorted_fnames = np.array(image_fnames)[sorted_indices].tolist()
    margin = 30  # maximum acquisition time difference allowed, in minutes, within a timeline instance

    # build timeline
    d = {}
    dates_already_seen = []
    for im_idx, fname in enumerate(sorted_fnames):

        new_date = True

        diff_wrt_prev_dates_in_mins = [
            abs((x - sorted_datetimes[im_idx]).total_seconds() / 60.0)
            for x in dates_already_seen
        ]

        if len(diff_wrt_prev_dates_in_mins) > 0:
            min_pos = np.argmin(diff_wrt_prev_dates_in_mins)

            # if this image was acquired within 30 mins of difference w.r.t to an already seen date,
            # then it is part of the same acquisition
            if diff_wrt_prev_dates_in_mins[min_pos] < margin:
                ref_date_id = dates_already_seen[min_pos].strftime("%Y%m%d_%H%M%S")
                d[ref_date_id].append(im_idx)
                new_date = False

        if new_date:
            date_id = sorted_datetimes[im_idx].strftime("%Y%m%d_%H%M%S")
            d[date_id] = [im_idx]
            dates_already_seen.append(sorted_datetimes[im_idx])

    timeline = []
    for k in d.keys():
        current_datetime = sorted_datetimes[d[k][0]]
        im_fnames_current_datetime = np.array(sorted_fnames)[d[k]].tolist()
        timeline.append(
            {
                "datetime": current_datetime,
                "id": k.split("/")[-1],
                "fnames": im_fnames_current_datetime,
                "n_images": len(d[k]),
                "adjusted": False,
                "image_weights": [],
            }
        )
    return timeline


def get_timeline_attributes(timeline, timeline_indices, attributes):
    """
    Displays the value of certain attributes at some indices of the timeline in a scene to bundle adjust
    """

    max_lens = np.zeros(len(attributes)).tolist()
    for idx in timeline_indices:
        to_display = ""
        for a_idx, a in enumerate(attributes):
            string_len = len("{}".format(timeline[idx][a]))
            if max_lens[a_idx] < string_len:
                max_lens[a_idx] = string_len
    max_len_idx = max([len(str(idx)) for idx in timeline_indices])
    index_str = "index"
    margin = max_len_idx - len(index_str)
    header_values = [index_str + " " * margin] if margin > 0 else [index_str]
    for a_idx, a in enumerate(attributes):
        margin = max_lens[a_idx] - len(a)
        header_values.append(a + " " * margin if margin > 0 else a)
    header_row = "  |  ".join(header_values)
    print(header_row)
    print("_" * len(header_row) + "\n")
    for idx in timeline_indices:
        margin = len(header_values[0]) - len(str(idx))
        to_display = [str(idx) + " " * margin if margin > 0 else str(idx)]
        for a_idx, a in enumerate(attributes):
            a_value = "{}".format(timeline[idx][a])
            margin = len(header_values[a_idx + 1]) - len(a_value)
            to_display.append(a_value + " " * margin if margin > 0 else a_value)
        print("  |  ".join(to_display))

    if "n_images" in attributes:  # add total number of images
        print("_" * len(header_row) + "\n")
        to_display = [" " * len(header_values[0])]
        for a_idx, a in enumerate(attributes):
            if a == "n_images":
                a_value = "{} total".format(
                    sum([timeline[idx]["n_images"] for idx in timeline_indices])
                )
                margin = len(header_values[a_idx + 1]) - len(a_value)
                to_display.append(a_value + " " * margin if margin > 0 else a_value)
            else:
                to_display.append(" " * len(header_values[a_idx + 1]))
        print("     ".join(to_display))
    print("\n")


def mask_from_shapely_polygons(polygons, im_size):
    """
    Builds a numpy binary array from a list of shapely polygons or multipolygon list
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


def get_binary_mask_from_aoi_lonlat_within_utm_bbx(utm_bbx, resolution, aoi_lonlat):
    """
    Gets a binary mask within a grid of specific resolution delimited by utm_bbx
    with 1 in those points inside the area of interest and 0 in those points outisde of it
    """

    height = int(np.floor((utm_bbx["ymax"] - utm_bbx["ymin"]) / resolution) + 1)
    width = int(np.floor((utm_bbx["xmax"] - utm_bbx["xmin"]) / resolution) + 1)

    lonlat_coords = np.array(aoi_lonlat["coordinates"][0])
    lats, lons = lonlat_coords[:, 1], lonlat_coords[:, 0]
    easts, norths = geotools.utm_from_latlon(lats, lons)

    offset = np.zeros(len(norths)).astype(np.float32)
    offset[norths < 0] = 10e6
    rows = (height - ((norths + offset) - utm_bbx["ymin"]) / resolution).astype(int)
    cols = ((easts - utm_bbx["xmin"]) / resolution).astype(int)
    poly_verts_colrow = np.vstack([cols, rows]).T

    from shapely.geometry import shape

    shapely_poly = shape(
        {"type": "Polygon", "coordinates": [poly_verts_colrow.tolist()]}
    )
    mask = mask_from_shapely_polygons([shapely_poly], (height, width))

    return mask


def get_binary_mask_from_aoi_lonlat_within_image(
    geotiff_fname, geotiff_rpc, aoi_lonlat
):
    """
    Gets a binary mask within the limits of a geotiff image
    with 1 in those points inside the area of interest and 0 in those points outisde of it
    """

    h, w = read_image_size(geotiff_fname)

    lonlat_coords = np.array(aoi_lonlat["coordinates"][0])
    lats, lons = lonlat_coords[:, 1], lonlat_coords[:, 0]
    poly_verts_colrow = np.array(
        [geotiff_rpc.projection(lon, lat, 0.0) for lon, lat in zip(lons, lats)]
    )

    from shapely.geometry import shape

    shapely_poly = shape(
        {"type": "Polygon", "coordinates": [poly_verts_colrow.tolist()]}
    )
    mask = mask_from_shapely_polygons([shapely_poly], (h, w))

    return mask


def apply_mask_to_raster(raster, mask):
    """
    Applies a mask to the numpy array raster
    mask contains 1 in those positions where data is to be kept and 0 in the rest
    """
    output = raster.copy()
    output[~mask.astype(bool)] = np.nan
    return output


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


def load_image_crops(
    geotiff_fnames,
    rpcs=None,
    aoi=None,
    crop_aoi=False,
    compute_aoi_mask=False,
    verbose=True,
):
    """
    Loads the geotiff or the geotiff crops of interest for each image in the list geotiff_fnames
    """

    compute_masks = compute_aoi_mask and rpcs is not None and aoi is not None

    crops = []
    for im_idx, path_to_geotiff in enumerate(geotiff_fnames):
        if aoi is not None and crop_aoi:
            # get the altitude of the center of the AOI
            import srtm4

            lon, lat = aoi["center"]
            alt = srtm4.srtm4(lon, lat)
            # project aoi and crop
            lons, lats = np.array(aoi["coordinates"][0]).T
            x, y = rpcs[im_idx].projection(lons, lats, alt)
            x_min, x_max, y_min, y_max = min(x), max(x), min(y), max(y)
            x0, y0, w, h = x_min, y_min, x_max - x_min, y_max - y_min
            with rasterio.open(path_to_geotiff) as src:
                im = (
                    src.read(window=((y0, y0 + h), (x0, x0 + w)))
                    .squeeze()
                    .astype(np.float)
                )
        else:
            with rasterio.open(path_to_geotiff) as src:
                im = src.read()[0, :, :].astype(np.float)
            x0, y0 = 0.0, 0.0

        crops.append(
            {
                "crop": im,
                "col0": x0,
                "row0": y0,
                "width": im.shape[1],
                "height": im.shape[0],
            }
        )
        if compute_masks:
            mask = get_binary_mask_from_aoi_lonlat_within_image(
                path_to_geotiff, rpcs[im_idx], aoi
            )
            crops[-1]["mask"] = mask
        if verbose and sys.stdout.name == "stdout":
            print(
                "\rLoading geotiff crops... {}/{}".format(
                    im_idx + 1, len(geotiff_fnames)
                ),
                end="\r",
            )
    if verbose:
        print(
            "\rLoading geotiff crops... {}/{}".format(im_idx + 1, len(geotiff_fnames)),
            flush=True,
        )
    return crops


def save_rpcs(filenames, rpcs):
    """
    Writes a series of rpc models to the specified filenames
    """
    for fn, rpc in zip(filenames, rpcs):
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        rpc.write_to_file(fn)


def load_rpcs_from_dir(image_fnames_list, rpc_dir, suffix="RPC_adj", verbose=True):
    """
    Loads rpcs from rpc files stored in a common directory
    """
    rpcs = []
    for im_idx, fname in enumerate(image_fnames_list):
        path_to_rpc = os.path.join(rpc_dir, "{}_{}.txt".format(get_id(fname), suffix))
        rpcs.append(rpcm.rpc_from_rpc_file(path_to_rpc))
        if verbose and sys.stdout.name == "stdout":
            print(
                "\rLoading rpcs... {}/{}".format(im_idx + 1, len(image_fnames_list)),
                end="\r",
            )
    if verbose:
        print(
            "\rLoading rpcs... {}/{}".format(im_idx + 1, len(image_fnames_list)),
            flush=True,
        )
    return rpcs


def save_projection_matrices(filenames, projection_matrices, crop_offsets):
    """
    Writes a series of projection matrices and their corresponding crop offsets to the specified filenames
    """
    for fn, P, offset in zip(filenames, projection_matrices, crop_offsets):
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        to_write = {
            # 'P_camera'
            # 'P_extrinsic'
            # 'P_intrinsic'
            "P": [P[0, :].tolist(), P[1, :].tolist(), P[2, :].tolist()],
            # 'exterior_orientation'
            "height": int(offset["height"]),
            "width": int(offset["width"]),
            "col_offset": int(offset["col0"]),
            "row_offset": int(offset["row0"]),
        }
        save_dict_to_json(to_write, fn)


def load_matrices_from_dir(
    image_fnames_list, P_dir, suffix="pinhole_adj", verbose=True
):
    """
    Loads projection matrices from json files stored in a common directory
    """
    proj_matrices = []
    for im_idx, fname in enumerate(image_fnames_list):
        path_to_P = os.path.join(P_dir, "{}_{}.json".format(get_id(fname), suffix))
        P = load_dict_from_json(path_to_P)["P"]
        proj_matrices.append(P / P[2, 3])
    if verbose:
        print("Projection matrices loaded from {}".format(P_dir))
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
    return crop_offsets


def epsg_from_utm_zone(utm_zone, datum="WGS84"):
    """
    Returns the epsg code given the string of a utm zone
    """
    from pyproj import CRS

    args = [utm_zone[:2], "+south" if utm_zone[-1] == "S" else "+north", datum]
    crs = CRS.from_proj4("+proj=utm +zone={} {} +datum={}".format(*args))
    return crs.to_epsg()


def load_s2p_configs_from_image_filenames(
    im_fnames, s2p_configs_dir, geotiff_label=None
):
    """
    Returns all config.json fnames in the s2p_configs_dir
    where both images are part of the im_fnames list
    """
    if geotiff_label is None:
        bnames = [os.path.basename(fn) for fn in im_fnames]
    else:
        bnames = [os.path.basename(fn) for fn in im_fnames if geotiff_label in fn]
    config_fnames = glob.glob(
        os.path.join(s2p_configs_dir, "**/config.json"), recursive=True
    )
    selected_config_fnames = []
    for s2p_config_fn in config_fnames:
        d = load_dict_from_json(s2p_config_fn)
        if (
            os.path.basename(d["images"][0]["img"]) in bnames
            and os.path.basename(d["images"][1]["img"]) in bnames
        ):
            selected_config_fnames.append(s2p_config_fn)
    return selected_config_fnames


def load_s2p_dsm_fnames_from_dir(s2p_dir):
    """
    Returns the filenames of all dsms within a directory containing s2p outputs
    dsm.tif in tiles directories are ignored
    """
    all_dsm_fnames = glob.glob(s2p_dir + "/**/dsm.tif", recursive=True)
    dsm_fnames = [fn for fn in all_dsm_fnames if "/tiles/" not in fn]
    return dsm_fnames


def read_geotiff_metadata(geotiff_fname):
    """
    Reads geotiff metadata
    """
    # reconstructed dsms have to present the following parameters
    with rasterio.open(geotiff_fname) as src:
        # dsm_data = src.read()[0,:,:]
        dsm_metadata = src
    xmin = dsm_metadata.bounds.left
    ymin = dsm_metadata.bounds.bottom
    xmax = dsm_metadata.bounds.right
    ymax = dsm_metadata.bounds.top
    epsg = dsm_metadata.crs
    resolution = dsm_metadata.res[0]
    h, w = dsm_metadata.shape
    utm_bbx = {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax}
    return utm_bbx, epsg, resolution, h, w


def approx_affine_projection_matrices(
    input_rpcs, crop_offsets, aoi_lonlat, verbose=False
):
    """
    Approximates a list of rpcs as affine projection matrices
    """
    import srtm4

    projection_matrices, n_cam = [], len(input_rpcs)
    for im_idx, (rpc, offset) in enumerate(zip(input_rpcs, crop_offsets)):
        lon, lat = aoi_lonlat["center"][0], aoi_lonlat["center"][1]
        alt = srtm4.srtm4(lon, lat)
        x, y, z = geotools.latlon_to_ecef_custom(lat, lon, alt)
        projection_matrices.append(
            camera_utils.approx_rpc_as_affine_projection_matrix(rpc, x, y, z, offset)
        )
        if verbose and sys.stdout.name == "stdout":
            print(
                "\rAffine projection matrix approximation... {}/{}".format(
                    im_idx + 1, n_cam
                ),
                end="\r",
            )
    if verbose:
        print(
            "\rAffine projection matrix approximation... {}/{}".format(
                im_idx + 1, n_cam
            ),
            flush=True,
        )
    errors = np.zeros(n_cam).tolist()
    return projection_matrices, errors


def approx_perspective_projection_matrices(input_rpcs, crop_offsets, verbose=False):
    """
    Approximates a list of rpcs as perspective projection matrices
    """
    projection_matrices, errors, n_cam = [], [], len(input_rpcs)
    for im_idx, (rpc, crop) in enumerate(zip(input_rpcs, crop_offsets)):
        P, e = camera_utils.approx_rpc_as_perspective_projection_matrix(rpc, crop)
        projection_matrices.append(P)
        errors.append(e)
        if verbose and sys.stdout.name == "stdout":
            print(
                "\rPerspective projection matrix approximation... {}/{}".format(
                    im_idx + 1, n_cam
                ),
                end="\r",
            )
    if verbose:
        print(
            "\rPerspective projection matrix approximation... {}/{}".format(
                im_idx + 1, n_cam
            ),
            flush=True,
        )
    return projection_matrices, errors


def save_list_of_pairs(path_to_npy, list_of_pairs):
    # list of pairs is a list of tuples, but is saved as a 2d array with 2 columns (one row per pair)
    np.save(path_to_npy, np.array(list_of_pairs))

def load_list_of_pairs(path_to_npy):
    # opposite operation of save_list_of_pairs
    array_t = np.load(path_to_npy).T.astype(int)
    return list(zip(array_t[0], array_t[1]))

def save_list_of_paths(path_to_txt, list_of_paths):
    with open(path_to_txt, 'w') as f:
        for p in list_of_paths:
            f.write("%s\n" % p)

def load_list_of_paths(path_to_txt):
    with open(path_to_txt, 'r') as f:
        content = f.readlines()
    return [x.strip() for x in content]

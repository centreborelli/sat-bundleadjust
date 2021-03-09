"""
* Bundle Adjustment (BA) for 3D Reconstruction from Multi-Date Satellite Images
* Based on https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
* by Roger Mari <roger.mari@ens-paris-saclay.fr>
"""

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import srtm4
from PIL import Image
from shapely.geometry import shape


def read_point_cloud_ply(filename):
    """
    to read a point cloud from a ply file
    the header of the file is expected to be as in the e.g., with vertices coords starting the line after end_header

    ply
    format ascii 1.0
    element vertex 541636
    property float x
    property float y
    property float z
    end_header
    """

    with open(filename, "r") as f_in:
        lines = f_in.readlines()
        content = [x.strip() for x in lines]
        n_pts = len(content) - 7
        pt_cloud = np.zeros((n_pts, 3))
        for i in range(n_pts):
            coords = re.findall(r"[-+]?\d*\.\d+|\d+", content[i + 7])
            pt_cloud[i, :] = np.array(
                [float(coords[0]), float(coords[1]), float(coords[2])]
            )
    return pt_cloud


def read_point_cloud_txt(filename):
    """
    to read a point cloud from a txt file
    where each line has 3 floats representing the x y z coordinates of a 3D point
    """

    with open(filename, "r") as f_in:
        lines = f_in.readlines()
        content = [x.strip() for x in lines]
        n_pts = len(content)
        pt_cloud = np.zeros((n_pts, 3))
        for i in range(n_pts):
            coords = re.findall(r"[-+]?\d*\.\d+|\d+", content[i])
            pt_cloud[i, :] = np.array(
                [float(coords[0]), float(coords[1]), float(coords[2])]
            )
    return pt_cloud


def write_point_cloud_ply(filename, point_cloud, color=np.array([255, 255, 255])):
    with open(filename, "w") as f_out:
        n_points = point_cloud.shape[0]
        # write output ply file with the point cloud
        f_out.write("ply\n")
        f_out.write("format ascii 1.0\n")
        f_out.write("element vertex {}\n".format(n_points))
        f_out.write("property float x\nproperty float y\nproperty float z\n")
        if not (color[0] == 255 and color[1] == 255 and color[2] == 255):
            f_out.write(
                "property uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar alpha\n"
            )
            f_out.write("element face 0\nproperty list uchar int vertex_indices\n")
        f_out.write("end_header\n")
        # write 3d points
        for i in range(n_points):
            p_3d = point_cloud[i, :]
            f_out.write("{} {} {}".format(p_3d[0], p_3d[1], p_3d[2]))
            if not (color[0] == 255 and color[1] == 255 and color[2] == 255):
                f_out.write(" {} {} {} 255".format(color[0], color[1], color[2]))
            f_out.write("\n")


def write_ply_cam(input_P, crop, filename, s=100.0):
    h, w = crop["crop"].shape
    with open(filename, "w") as f_out:
        f_out.write("ply\n")
        f_out.write("format ascii 1.0\n")
        f_out.write("element vertex 5\n")
        f_out.write("property float x\n")
        f_out.write("property float y\n")
        f_out.write("property float z\n")
        f_out.write("element edge 8\n")
        f_out.write("property int vertex1\n")
        f_out.write("property int vertex2\n")
        f_out.write("end_header\n")

        K, R, t, oC = decompose_perspective_camera(input_P)
        KRinv = np.linalg.inv(K @ R)

        p4 = oC + KRinv @ np.array([-w / 2 * s, -h / 2 * s, 1]).T
        p3 = oC + KRinv @ np.array([w / 2 * s, -h / 2 * s, 1]).T
        p2 = oC + KRinv @ np.array([w / 2 * s, h / 2 * s, 1]).T
        p1 = oC + KRinv @ np.array([-w / 2 * s, h / 2 * s, 1]).T
        p5 = oC

        # write 3d points
        f_out.write(
            "{} {} {}\n".format(p1[0] - 2611000, p1[1] - 4322000, p1[2] - 3506000)
        )
        f_out.write(
            "{} {} {}\n".format(p2[0] - 2611000, p2[1] - 4322000, p2[2] - 3506000)
        )
        f_out.write(
            "{} {} {}\n".format(p3[0] - 2611000, p3[1] - 4322000, p3[2] - 3506000)
        )
        f_out.write(
            "{} {} {}\n".format(p4[0] - 2611000, p4[1] - 4322000, p4[2] - 3506000)
        )
        f_out.write(
            "{} {} {}\n".format(p5[0] - 2611000, p5[1] - 4322000, p5[2] - 3506000)
        )

        # write edges
        f_out.write("0 1\n1 2\n2 3\n3 0\n0 4\n1 4\n2 4\n3 4")


def save_geotiff(filename, input_im, epsg_code, x, y, r=0.5):
    # (x,y) - geographic coordinates of the top left pixel
    # r - pixel resolution in meters
    # code epsg (e.g. buenos aires is in the UTM zone 21 south - epsg: 32721)
    h, w = input_im.shape
    profile = {
        "driver": "GTiff",
        "count": 1,
        "width": w,
        "height": h,
        "dtype": rasterio.dtypes.float64,
        "crs": "epsg:32614",  # UTM zone 14 north
        "transform": rasterio.transform.from_origin(x - r / 2, y + r / 2, r, r),
    }
    with rasterio.open(filename, "w", **profile) as dst:
        dst.write(np.asarray([input_im]))


def plot_connectivity_graph(C, min_matches, save_pgf=False):

    import networkx as nx

    G, _, _, _, _ = build_connectivity_graph(C, min_matches=min_matches)

    if save_pgf:
        fig_width_pt = 229.5  # CVPR
        inches_per_pt = 1.0 / 72.27  # Convert pt to inches
        golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
        fig_width = fig_width_pt * inches_per_pt  # width in inches
        fig_height = fig_width * golden_mean  # height in inches
        fig_size = [fig_width, fig_height]
        params = {
            "backend": "pgf",
            "axes.labelsize": 8,
            "font.size": 8,
            "legend.fontsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 8,
            "text.usetex": True,
            "figure.figsize": fig_size,
        }
        plt.rcParams.update(params)

    fig = plt.gcf()
    fig.set_size_inches(8, 8)

    # draw all nodes in a circle
    G_pos = nx.circular_layout(G)

    # draw nodes
    nx.draw_networkx_nodes(
        G, G_pos, node_size=600, node_color="#FFFFFF", edgecolors="#000000"
    )

    # paint subgroup of nodes
    # nx.draw_networkx_nodes(G, G_pos, nodelist=[41,42, 43, 44, 45], node_size=600, node_color='#FF6161',edgecolors='#000000')

    # draw edges and labels
    nx.draw_networkx_edges(G, G_pos)
    nx.draw_networkx_labels(G, G_pos, font_size=12, font_family="sans-serif")

    # show graph and save it as .pgf
    plt.axis("off")
    if save_pgf:
        plt.savefig("graph.pgf", pad_inches=0, bbox_inches="tight", dpi=200)
    plt.show()


def build_connectivity_graph(C, min_matches, verbose=True):
    def connected_component_subgraphs(G):
        for c in nx.connected_components(G):
            yield G.subgraph(c)

    # (1) Build connectivity matrix A, where position (i,j) contains the number of matches between images i and j
    n_cam = int(C.shape[0] / 2)
    A, n_correspondences_filt, tmp_pairs = np.zeros((n_cam, n_cam)), [], []
    for im1 in range(n_cam):
        for im2 in range(im1 + 1, n_cam):
            n_matches = np.sum(
                1
                * np.logical_and(
                    1 * ~np.isnan(C[2 * im1, :]), 1 * ~np.isnan(C[2 * im2, :])
                )
            )
            n_correspondences_filt.append(n_matches)
            tmp_pairs.append((im1, im2))
            A[im1, im2] = n_matches
            A[im2, im1] = n_matches

    # (2) Filter graph edges according to the threshold on the number of matches
    pairs_to_draw = []
    matches_per_pair = []
    for i in range(len(tmp_pairs)):
        if n_correspondences_filt[i] >= min_matches:
            pairs_to_draw.append(tmp_pairs[i])
            matches_per_pair.append(n_correspondences_filt[i])

    # (3) Draw the graph and save it as a .pgf image
    import networkx as nx

    # create networkx graph
    G = nx.Graph()
    # add edges
    for edge in pairs_to_draw:
        G.add_edge(edge[0], edge[1])

    # get list of connected components (to see if there is any disconnected subgroup)
    G_cc = list(connected_component_subgraphs(G))
    n_cc = len(G_cc)
    missing_cams = list(set(np.arange(n_cam)) - set(G_cc[0].nodes))

    obs_per_cam = np.sum(1 * ~np.isnan(C), axis=1)[::2]

    if verbose:
        print(
            "Connectivity graph: {} missing cameras: {}".format(
                len(missing_cams), missing_cams
            )
        )
        print("                    {} connected components".format(n_cc))
        print("                    {} edges".format(len(pairs_to_draw)))
        print(
            "                    {} min n_matches in an edge".format(
                min(matches_per_pair)
            )
        )
        print("                    {} min obs per camera\n".format(min(obs_per_cam)))

    return G, n_cc, pairs_to_draw, matches_per_pair, missing_cams


def plot_dsm(fname, vmin=None, vmax=None, color_bar="jet", save_pgf=False):

    f_size = (13, 10)
    fig = plt.figure()
    params = {
        "backend": "pgf",
        "axes.labelsize": 22,
        "ytick.labelleft": False,
        "font.size": 22,
        "legend.fontsize": 22,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "xtick.top": False,
        "xtick.bottom": False,
        "xtick.labelbottom": False,
        "ytick.left": False,
        "ytick.right": False,
        "text.usetex": True,  # use TeX for text
        "font.family": "serif",
        "legend.loc": "upper left",
        "legend.fontsize": 22,
    }
    plt.rcParams.update(params)
    plt.figure(figsize=f_size)

    im = np.array(Image.open(fname))
    vmin_in = np.min(im.squeeze()) if vmin is None else vmin
    vmax_in = np.max(im.squeeze()) if vmax is None else vmin
    plt.imshow(im.squeeze(), cmap=color_bar, vmin=vmin, vmax=vmax)
    plt.axis("equal")
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=40)
    if save_pgf:
        plt.savefig(
            os.path.splitext(fname) + ".pgf", pad_inches=0, bbox_inches="tight", dpi=200
        )
    plt.show()


def display_ba_error_particular_view(
    P_before, P_after, pts3d_before, pts3d_after, pts2d, image
):

    n_pts = pts3d_before.shape[0]

    # reprojections before bundle adjustment
    proj = P_before @ np.hstack((pts3d_before, np.ones((n_pts, 1)))).T
    pts_reproj_before = (proj[:2, :] / proj[-1, :]).T

    # reprojections after bundle adjustment
    proj = P_after @ np.hstack((pts3d_after, np.ones((n_pts, 1)))).T
    pts_reproj_after = (proj[:2, :] / proj[-1, :]).T

    err_before = np.sum(abs(pts_reproj_before - pts2d), axis=1)
    err_after = np.sum(abs(pts_reproj_after - pts2d), axis=1)

    print("Mean abs reproj error before BA: {:.4f}".format(np.mean(err_before)))
    print("Mean abs reproj error after  BA: {:.4f}".format(np.mean(err_after)))

    # reprojection error histograms for the selected image
    fig = plt.figure(figsize=(10, 3))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.title.set_text("Reprojection error before BA")
    ax2.title.set_text("Reprojection error after  BA")
    ax1.hist(err_before, bins=40)
    ax2.hist(err_after, bins=40)

    plt.show()

    # warning: this is slow...

    fig = plt.figure(figsize=(20, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.title.set_text("Before BA")
    ax2.title.set_text("After  BA")
    ax1.imshow(image, cmap="gray")
    ax2.imshow(image, cmap="gray")
    for k in range(min(1000, n_pts)):
        # before bundle adjustment
        ax1.plot(
            [pts2d[k, 0], pts_reproj_before[k, 0]],
            [pts2d[k, 1], pts_reproj_before[k, 1]],
            "r-",
        )
        ax1.plot(*pts2d[k], "yx")
        # after bundle adjustment
        ax2.plot(
            [pts2d[k, 0], pts_reproj_after[k, 0]],
            [pts2d[k, 1], pts_reproj_after[k, 1]],
            "r-",
        )
        ax2.plot(*pts2d[k], "yx")
    plt.show()


def get_image_footprints(myrpcs, crop_offsets, z=None):

    from bundle_adjust import geotools

    footprints = []
    for cam_idx, (rpc, offset) in enumerate(zip(myrpcs, crop_offsets)):
        if z is None:
            z = srtm4.srtm4(rpc.lon_offset, rpc.lat_offset)
        footprint_lonlat_geojson = geotools.lonlat_geojson_from_geotiff_crop(
            rpc, offset, z
        )
        footprint_utm_geojson = geotools.utm_geojson_from_lonlat_geojson(
            footprint_lonlat_geojson
        )
        footprints.append({"poly": shape(footprint_utm_geojson), "z": z})
        # print('\r{} / {} done'.format(cam_idx+1, len(myrpcs)), end = '\r')
    return footprints


def save_ply_pts_projected_over_geotiff_as_svg(
    geotiff_fname, ply_fname, output_svg_fname, verbose=False
):

    # points in the ply file are assumed to be in ecef coordinates
    from bundle_adjust import geotools
    from bundle_adjust.data_loader import read_geotiff_metadata

    from .feature_tracks.ft_utils import save_pts2d_as_svg

    utm_bbx, _, resolution, height, width = read_geotiff_metadata(geotiff_fname)
    xyz = read_point_cloud_ply(ply_fname)

    lats, lons, h = geotools.ecef_to_latlon_custom(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    easts, norths = geotools.utm_from_lonlat(lons, lats)

    offset = np.zeros(len(norths)).astype(np.float32)
    offset[norths < 0] = 10e6
    cols = ((easts - utm_bbx["xmin"]) / resolution).astype(int)
    rows = (height - ((norths + offset) - utm_bbx["ymin"]) / resolution).astype(int)
    pts2d_ba_all = np.vstack([cols, rows]).T

    # keep only those points inside the geotiff limits
    valid_cols = np.logical_and(cols < width, cols >= 0)
    valid_rows = np.logical_and(rows < height, rows >= 0)
    pts2d_ba = pts2d_ba_all[np.logical_and(valid_cols, valid_rows), :]

    if verbose:
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(np.array(Image.open(geotiff_fname)), cmap="gray")
        plt.scatter(x=pts2d_ba[:, 0], y=pts2d_ba[:, 1], c="r", s=4)
        plt.show()

    save_pts2d_as_svg(output_svg_fname, pts2d_ba, c="yellow")


def close_small_holes_from_dsm(
    input_geotiff_fname, output_geotiff_fname, imscript_bin_dir
):

    # small hole interpolation by closing
    args = [input_geotiff_fname, output_geotiff_fname, imscript_bin_dir]
    cmd = '{2}/morsi square closing {0} | {2}/plambda {0} - "x isfinite x y isfinite y nan if if" -o {1}'.format(
        *args
    )
    os.system(cmd)

    # read imscript result
    with rasterio.open(output_geotiff_fname) as imscript_data:
        cdsm_array = imscript_data.read(1)
    os.system("rm {}".format(output_geotiff_fname))

    # write imscript result with same metadata as input geotiff
    with rasterio.open(input_geotiff_fname) as src_data:
        kwds = src_data.profile
        with rasterio.open(output_geotiff_fname, "w", **kwds) as dst_data:
            dst_data.write(cdsm_array.astype(rasterio.float32), 1)


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


def run_plyflatten(
    ply_list, resolution, utm_bbx, output_file, aoi_lonlat=None, std=False, cnt=False
):
    from plyflatten import plyflatten_from_plyfiles_list

    # compute roi from utm bounding box
    xoff = np.floor(utm_bbx["xmin"] / resolution) * resolution
    # xsize = int(1 + np.floor((utm_bbx['xmax'] - xoff) / resolution))
    xsize = int(np.floor((utm_bbx["xmax"] - utm_bbx["xmin"]) / resolution) + 1)  # width
    yoff = np.ceil(utm_bbx["ymax"] / resolution) * resolution
    # ysize = int(1 - np.floor((utm_bbx['ymin']  - yoff) / resolution))
    ysize = int(
        np.floor((utm_bbx["ymax"] - utm_bbx["ymin"]) / resolution) + 1
    )  # height
    roi = (xoff, yoff, xsize, ysize)

    # run plyflatten
    raster, profile = plyflatten_from_plyfiles_list(
        ply_list, resolution, roi=roi, std=std
    )

    # if aoi_lonlat is not None, then mask those parts outside the area of interest with NaN values
    from bundle_adjust.data_loader import (
        apply_mask_to_raster, get_binary_mask_from_aoi_lonlat_within_utm_bbx)

    if aoi_lonlat is None:
        mask = np.ones(raster.shape[:2], dtype=np.float32)
    else:
        mask = get_binary_mask_from_aoi_lonlat_within_utm_bbx(
            utm_bbx, resolution, aoi_lonlat
        )

    # write dsm
    import rasterio

    profile["dtype"] = raster.dtype
    profile["height"] = raster.shape[0]
    profile["width"] = raster.shape[1]
    profile["count"] = 1
    profile["driver"] = "GTiff"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with rasterio.open(output_file, "w", **profile) as f:
        f.write(apply_mask_to_raster(raster[:, :, 0], mask), 1)

    # write std if flag was enabled
    if std:

        # if you are using the version of pyflatten that writes an extra layer on top of std with the number of counts
        if raster.shape[2] % 2 == 1:
            if cnt:
                cnt_path = os.path.join(
                    os.path.dirname(output_file), "cnt/" + os.path.basename(output_file)
                )
                os.makedirs(os.path.dirname(cnt_path), exist_ok=True)
                with rasterio.open(cnt_path, "w", **profile) as f:
                    f.write(apply_mask_to_raster(raster[:, :, -1], mask), 1)
            raster = raster[:, :, :-1]

        # write remaining extra layers with the std
        std_path = os.path.join(
            os.path.dirname(output_file), "std/" + os.path.basename(output_file)
        )
        os.makedirs(os.path.dirname(std_path), exist_ok=True)
        n = raster.shape[2]
        assert n % 2 == 0
        with rasterio.open(std_path, "w", **profile) as f:
            f.write(apply_mask_to_raster(raster[:, :, n // 2], mask), 1)


def merge_s2p_ply_and_assign_color_v2(ply_fnames, out_ply, color):
    from s2p import ply
    ply_comments = ply.read_3d_point_cloud_from_ply(ply_fnames[0])[1]
    super_xyz = np.vstack([ply.read_3d_point_cloud_from_ply(fn)[0] for fn in ply_fnames])

    if color == 'dark_blue':
        rgb = np.array([0, 125, 255])
    elif color == 'orange':
        rgb = np.array([255, 125, 0])
    elif color == 'red':
        rgb = np.array([220, 0, 0])
    elif color == 'light_blue':
        rgb = np.array([0, 220, 255])
    elif color == 'green':
        rgb = np.array([0, 200, 0])
    else:
        print('color not recognized! using gray!')
        rgb = np.array([125, 125, 125])
    v_colors = np.tile(rgb, (super_xyz.shape[0],1)) *0.3 + super_xyz[:, 3:6] *0.7
    ply.write_3d_point_cloud_to_ply(out_ply, super_xyz[:, :3], colors=v_colors.astype('uint8'), comments=ply_comments)


def merge_s2p_ply_and_assign_color(ply_fnames, out_ply, color):
    from s2p import ply
    ply_comments = ply.read_3d_point_cloud_from_ply(ply_fnames[0])[1]
    super_xyz = np.vstack([ply.read_3d_point_cloud_from_ply(fn)[0] for fn in ply_fnames])

    if color == 'dark_blue':
        rgb = np.array([0, 125, 255])
    elif color == 'orange':
        rgb = np.array([255, 125, 0])
    elif color == 'red':
        rgb = np.array([220, 0, 0])
    elif color == 'light_blue':
        rgb = np.array([0, 220, 255])
    elif color == 'green':
        rgb = np.array([0, 200, 0])
    else:
        print('color not recognized! using gray!')
        rgb = np.array([125, 125, 125])
    v_colors = np.tile(rgb, (super_xyz.shape[0],1))
    ply.write_3d_point_cloud_to_ply(out_ply, super_xyz[:, :3], colors=v_colors.astype('uint8'), comments=ply_comments)


def merge_s2p_ply(ply_fnames, out_ply):
    from s2p import ply

    ply_comments = ply.read_3d_point_cloud_from_ply(ply_fnames[0])[1]
    super_xyz = np.vstack(
        [ply.read_3d_point_cloud_from_ply(fn)[0] for fn in ply_fnames]
    )
    ply.write_3d_point_cloud_to_ply(
        out_ply,
        super_xyz[:, :3],
        colors=super_xyz[:, 3:6].astype("uint8"),
        comments=ply_comments,
    )


def get_mask_and_its_polygon(aoi_path, dsm_res):
    from bundle_adjust import data_loader as loader
    from bundle_adjust import geotools

    dsm_res = float(dsm_res)
    corrected_aoi_lonlat = loader.load_pickle(aoi_path)
    corrected_utm_bbx = geotools.utm_bbox_from_aoi_lonlat(corrected_aoi_lonlat)
    mask = loader.get_binary_mask_from_aoi_lonlat_within_utm_bbx(
        corrected_utm_bbx, dsm_res, corrected_aoi_lonlat
    )

    utm_bbx = corrected_utm_bbx
    resolution = dsm_res
    aoi_lonlat = corrected_aoi_lonlat
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

    return shapely_poly, mask


def reestimate_lonlat_geojson_after_rpc_correction(
    initial_rpc, corrected_rpc, lonlat_geojson
):

    import srtm4

    from bundle_adjust import geotools

    aoi_lons_init, aoi_lats_init = np.array(lonlat_geojson["coordinates"][0]).T
    alt = srtm4.srtm4(np.mean(aoi_lons_init), np.mean(aoi_lats_init))
    aoi_cols_init, aoi_rows_init = initial_rpc.projection(
        aoi_lons_init, aoi_lats_init, alt
    )
    aoi_lons_ba, aoi_lats_ba = corrected_rpc.localization(
        aoi_cols_init, aoi_rows_init, alt
    )
    lonlat_coords = np.vstack((aoi_lons_ba, aoi_lats_ba)).T
    lonlat_geojson = geotools.geojson_polygon(lonlat_coords)

    return lonlat_geojson


def plot_heatmap_reprojection_error(
    err, pts3d_ecef, cam_ind, pts_ind, aoi_lonlat, resolution, thr=1.0, scale=2
):

    from bundle_adjust import ba_core
    from bundle_adjust import data_loader as loader
    from bundle_adjust import geotools

    lats, lons, alts = geotools.ecef_to_latlon_custom(
        pts3d_ecef[:, 0], pts3d_ecef[:, 1], pts3d_ecef[:, 2]
    )
    easts, norths = geotools.utm_from_lonlat(lons, lats)
    norths[norths < 0] += 10e6
    utm_bbx = geotools.utm_bbox_from_aoi_lonlat(aoi_lonlat)

    track_err = ba_core.compute_mean_reprojection_error_per_track(err, pts_ind, cam_ind)

    height = int(np.floor((utm_bbx["ymax"] - utm_bbx["ymin"]) / resolution) + 1)
    width = int(np.floor((utm_bbx["xmax"] - utm_bbx["xmin"]) / resolution) + 1)
    cols = ((easts - utm_bbx["xmin"]) / resolution).astype(int)
    rows = (height - (norths - utm_bbx["ymin"]) / resolution).astype(int)
    pts2d_ba_all = np.vstack([cols, rows]).T
    # keep only those points inside the geotiff limits
    valid_cols = np.logical_and(cols < width, cols >= 0)
    valid_rows = np.logical_and(rows < height, rows >= 0)
    valid_pts = np.logical_and(valid_cols, valid_rows)

    pts2d_ba = pts2d_ba_all[valid_pts, :]
    track_err = track_err[valid_pts]

    fig = plt.figure(figsize=(10, 5))
    plt.plot(sorted(track_err))
    plt.title("average reprojection error per track")
    plt.plot()

    if thr is not None:
        n_pts = pts3d_ecef.shape[0]
        n_outliers = np.sum(1 * (track_err > thr))
        args = [n_outliers, n_outliers / n_pts * 100, thr]
        print(
            "{} points ({:.2f}%) with reprojection error above {:.2f} pixels".format(
                *args
            )
        )
        track_err[track_err > thr] = thr

    # aoi contour
    lonlat_coords = np.array(aoi_lonlat["coordinates"][0])
    lats, lons = lonlat_coords[:, 1], lonlat_coords[:, 0]
    easts, norths = geotools.utm_from_lonlat(lons, lats)
    norths[norths < 0] += 10e6
    rows = (height - (norths - utm_bbx["ymin"]) / resolution).astype(int)
    cols = ((easts - utm_bbx["xmin"]) / resolution).astype(int)
    poly_verts_colrow = np.vstack([cols, rows]).T

    from shapely.geometry import shape

    shapely_poly = shape(
        {"type": "Polygon", "coordinates": [poly_verts_colrow.tolist()]}
    )

    scale_factor = width / 10
    fig = plt.figure(
        figsize=((width + int(width * 0.1)) / scale_factor, height / scale_factor)
    )
    plt.plot(*shapely_poly.exterior.xy, color="black")
    sc = plt.scatter(x=pts2d_ba[:, 0], y=pts2d_ba[:, 1], c=track_err * 255, s=scale)
    ax = plt.gca()
    ax.set_ylim((height, 0))
    ax.set_xlim((0, width))
    max_v, min_v = max(track_err * 255), 0
    mid_v = min_v + (max_v - min_v) / 2
    adj_ticks = [min_v, mid_v, max_v]
    adj_ticks_labels = [
        "{:.2f}".format(min_v / 255),
        "{:.2f}".format(mid_v / 255),
        "{:.2f}".format(max_v / 255),
    ]
    cb_frac, cb_pad, cb_asp = 0.045, 0.0, 7
    cbar = plt.colorbar(
        sc, fraction=cb_frac, pad=cb_pad, aspect=cb_asp, ticks=adj_ticks
    )
    cbar.set_ticklabels(adj_ticks_labels)
    plt.axis('off')
    plt.show()


def update_geotiff_rpc(geotiff_path, rpc_model):
    from osgeo import gdal, gdalconst
    geotiff_dataset = gdal.Open(geotiff_path, gdalconst.GA_Update)
    geotiff_dataset.SetMetadata(rpc_rpcm_to_geotiff_format(rpc_model.__dict__ ), 'RPC')
    del(geotiff_dataset)

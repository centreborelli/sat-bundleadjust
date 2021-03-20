"""
* Bundle Adjustment (BA) for 3D Reconstruction from Multi-Date Satellite Images
* Based on https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
* by Roger Mari <roger.mari@ens-paris-saclay.fr>
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import re
from bundle_adjust import loader as loader


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
            pt_cloud[i, :] = np.array([float(coords[0]), float(coords[1]), float(coords[2])])
    return pt_cloud


def write_point_cloud_ply(filename, point_cloud, color=np.array([None, None, None])):
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
    nx.draw_networkx_nodes(G, G_pos, node_size=600, node_color="#FFFFFF", edgecolors="#000000")

    # paint subgroup of nodes
    # nx.draw_networkx_nodes(G, G_pos, nodelist=[41,42, 43, 44, 45], node_size=600,
    #                        node_color="#FF6161", edgecolors="#000000")

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
    n_cam = C.shape[0] // 2
    A, n_correspondences_filt, tmp_pairs = np.zeros((n_cam, n_cam)), [], []
    not_nan_C = ~np.isnan(C)
    for im1 in range(n_cam):
        for im2 in range(im1 + 1, n_cam):
            n_matches = np.sum(not_nan_C[2 * im1] & not_nan_C[2 * im2])
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
        print("Connectivity graph: {} missing cameras: {}".format(len(missing_cams), missing_cams))
        print("                    {} connected components".format(n_cc))
        print("                    {} edges".format(len(pairs_to_draw)))
        print("                    {} min n_matches in an edge".format(min(matches_per_pair)))
        print("                    {} min obs per camera\n".format(min(obs_per_cam)))

    return G, n_cc, pairs_to_draw, matches_per_pair, missing_cams


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

    from bundle_adjust import geotools

    aoi_lons_init, aoi_lats_init = np.array(lonlat_geojson["coordinates"][0]).T
    alt = srtm4.srtm4(np.mean(aoi_lons_init), np.mean(aoi_lats_init))
    aoi_cols_init, aoi_rows_init = initial_rpc.projection(aoi_lons_init, aoi_lats_init, alt)
    aoi_lons_ba, aoi_lats_ba = corrected_rpc.localization(aoi_cols_init, aoi_rows_init, alt)
    lonlat_coords = np.vstack((aoi_lons_ba, aoi_lats_ba)).T
    lonlat_geojson = geotools.geojson_polygon(lonlat_coords)

    return lonlat_geojson


def update_geotiff_rpc(geotiff_path, rpc_model):
    from osgeo import gdal, gdalconst

    geotiff_dataset = gdal.Open(geotiff_path, gdalconst.GA_Update)
    geotiff_dataset.SetMetadata(rpc_rpcm_to_geotiff_format(rpc_model.__dict__), "RPC")
    del geotiff_dataset


def reproject_pts3d(cam_init, cam_ba, cam_model, obs2d, pts3d_init, pts3d_ba, image_fname=None, verbose=False):

    if image_fname is not None and not os.path.exists(image_fname):
        image_fname = None

    from bundle_adjust.camera_utils import project_pts3d

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

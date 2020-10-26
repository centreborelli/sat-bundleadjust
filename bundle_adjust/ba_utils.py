"""
* Bundle Adjustment (BA) for 3D Reconstruction from Multi-Date Satellite Images
* Based on https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
* by Roger Mari <mari@cmla.ens-cachan.fr>
"""

import numpy as np
import matplotlib.pyplot as plt

import re
import os
from PIL import Image
import srtm4
from shapely.geometry import shape
import rasterio

def get_predefined_pairs(fname, site, order, myimages):
    pairs = []
    with open(fname) as f:
        if order in ['oracle', 'sift']:
            for i in range(50):
                current_str = f.readline()
                a = [int(s) for s in current_str.split() if s.isdigit()]
                p, q = a[0]-1, a[1]-1
                pairs.append((p,q))
        else:
            # reads pairs from the heuristics order
            myimages_fn = [os.path.basename(i) for i in myimages]
            if site == 'IARPA':
                while len(pairs) < 50:
                    current_str = f.readline().split(' ')
                    im1_fn, im2_fn = os.path.basename(current_str[0]), os.path.basename(current_str[1])
                    if im1_fn in myimages_fn and im2_fn in myimages_fn:
                        p, q = myimages_fn.index(im1_fn), myimages_fn.index(im2_fn)
                        pairs.append((p,q))
            else:
                while len(pairs) < 50:
                    current_str = f.readline().split(' ')
                    im1_fn, im2_fn = os.path.basename(current_str[0]+'.tif'), os.path.basename(current_str[1]+'.tif')
                    if im1_fn in myimages_fn and im2_fn in myimages_fn:
                        p, q = myimages_fn.index(im1_fn), myimages_fn.index(im2_fn)
                        pairs.append((p,q))            
    return pairs


def read_point_cloud_ply(filename):
    '''
    to read a point cloud from a ply file
    the header of the file is expected to be as in the e.g., with vertices coords starting the line after end_header
    
    ply
    format ascii 1.0
    element vertex 541636
    property float x
    property float y
    property float z
    end_header
    
    '''
    
    with open(filename, 'r') as f_in:
        lines = f_in.readlines()
        content = [x.strip() for x in lines]
        n_pts = len(content)-7
        pt_cloud = np.zeros((n_pts,3))
        for i in range(n_pts):
            coords = re.findall(r"[-+]?\d*\.\d+|\d+", content[i+7])
            pt_cloud[i,:] = np.array([float(coords[0]),float(coords[1]),float(coords[2])])
    return pt_cloud

def read_point_cloud_txt(filename):
    '''
    to read a point cloud from a txt file
    where each line has 3 floats representing the x y z coordinates of a 3D point
    '''
    
    with open(filename, 'r') as f_in:
        lines = f_in.readlines()
        content = [x.strip() for x in lines]
        n_pts = len(content)
        pt_cloud = np.zeros((n_pts,3))
        for i in range(n_pts):
            coords = re.findall(r"[-+]?\d*\.\d+|\d+", content[i])
            pt_cloud[i,:] = np.array([float(coords[0]),float(coords[1]),float(coords[2])])
    return pt_cloud

def write_point_cloud_ply(filename, point_cloud, color=np.array([255,255,255])):
    with open(filename, 'w') as f_out:
        n_points = point_cloud.shape[0]
        # write output ply file with the point cloud
        f_out.write('ply\n')
        f_out.write('format ascii 1.0\n')
        f_out.write('element vertex {}\n'.format(n_points))
        f_out.write('property float x\nproperty float y\nproperty float z\n')
        if not (color[0] == 255 and color[1] == 255 and color[2] == 255):
            f_out.write('property uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar alpha\n')
            f_out.write('element face 0\nproperty list uchar int vertex_indices\n')
        f_out.write('end_header\n')
        # write 3d points
        for i in range(n_points):
            p_3d = point_cloud[i,:]
            f_out.write('{} {} {}'.format(p_3d[0], p_3d[1], p_3d[2]))
            if not (color[0] == 255 and color[1] == 255 and color[2] == 255):
                f_out.write(' {} {} {} 255'.format(color[0], color[1], color[2]))
            f_out.write('\n')

def write_ply_cam(input_P, crop, filename, s=100.):
    
    h, w = crop['crop'].shape
    with open(filename, 'w') as f_out:
        f_out.write('ply\n')
        f_out.write('format ascii 1.0\n')
        f_out.write('element vertex 5\n')
        f_out.write('property float x\n')
        f_out.write('property float y\n')
        f_out.write('property float z\n')
        f_out.write('element edge 8\n')
        f_out.write('property int vertex1\n')
        f_out.write('property int vertex2\n')
        f_out.write('end_header\n')

        K, R, t, oC = decompose_perspective_camera(input_P)
        KRinv = np.linalg.inv(K @ R)
             
        p4 = oC + KRinv @ np.array([-w/2 * s, -h/2 * s,1]).T
        p3 = oC + KRinv @ np.array([ w/2 * s, -h/2 * s,1]).T
        p2 = oC + KRinv @ np.array([ w/2 * s,  h/2 * s,1]).T
        p1 = oC + KRinv @ np.array([-w/2 * s,  h/2 * s,1]).T
        p5 = oC
        
        # write 3d points
        f_out.write('{} {} {}\n'.format(p1[0] - 2611000, p1[1] - 4322000, p1[2] - 3506000))
        f_out.write('{} {} {}\n'.format(p2[0] - 2611000, p2[1] - 4322000, p2[2] - 3506000))
        f_out.write('{} {} {}\n'.format(p3[0] - 2611000, p3[1] - 4322000, p3[2] - 3506000))
        f_out.write('{} {} {}\n'.format(p4[0] - 2611000, p4[1] - 4322000, p4[2] - 3506000))
        f_out.write('{} {} {}\n'.format(p5[0] - 2611000, p5[1] - 4322000, p5[2] - 3506000))

        # write edges
        f_out.write('0 1\n1 2\n2 3\n3 0\n0 4\n1 4\n2 4\n3 4')       


def save_geotiff(filename, input_im, epsg_code, x, y, r=0.5):
    # (x,y) - geographic coordinates of the top left pixel
    # r - pixel resolution in meters
    # code epsg (e.g. buenos aires is in the UTM zone 21 south - epsg: 32721)
    h, w = input_im.shape
    profile = {'driver': 'GTiff',
               'count':  1,
               'width':  w,
               'height': h,
               'dtype': rasterio.dtypes.float64,
               'crs': 'epsg:32614',  # UTM zone 14 north
               'transform': rasterio.transform.from_origin(x - r / 2, y + r / 2, r, r)}
    with rasterio.open(filename, 'w', **profile) as dst:
        dst.write(np.asarray([input_im]))
        
def latlon_to_ecef_custom(lat, lon, alt):
    '''
    to convert from geodetic (lat, lon, alt) to geocentric coordinates (x, y, z)
    '''
    rad_lat = lat * (np.pi / 180.0)
    rad_lon = lon * (np.pi / 180.0)

    a = 6378137.0
    finv = 298.257223563
    f = 1 / finv
    e2 = 1 - (1 - f) * (1 - f)
    v = a / np.sqrt(1 - e2 * np.sin(rad_lat) * np.sin(rad_lat))

    x = (v + alt) * np.cos(rad_lat) * np.cos(rad_lon)
    y = (v + alt) * np.cos(rad_lat) * np.sin(rad_lon)
    z = (v * (1 - e2) + alt) * np.sin(rad_lat)

    return x, y, z

def ecef_to_latlon_custom(x, y, z):
    '''
    to convert from geocentric coordinates (x, y, z) to geodetic (lat, lon, alt)
    '''
    a = 6378137.0
    e = 8.1819190842622e-2

    asq = a ** 2
    esq = e ** 2

    b   = np.sqrt(asq * (1 - esq))
    bsq = b ** 2

    ep  = np.sqrt((asq - bsq)/bsq)
    p   = np.sqrt( (x ** 2) + (y ** 2) )
    th  = np.arctan2(a * z, b * p)

    lon = np.arctan2(y, x)
    lat = np.arctan2( (z + (ep ** 2) * b * (np.sin(th) ** 3) ), (p - esq * a * (np.cos(th) ** 3) ) )
    N = a / ( np.sqrt(1 - esq * (np.sin(lat) ** 2) ) )
    alt = p / np.cos(lat) - N 

    lon = lon * 180 / np.pi
    lat = lat * 180 / np.pi

    return lat, lon, alt

def ecef_to_latlon_custom_ad(x, y, z):
    # the 'ad' package is unable to differentiate numpy trigonometry functions (sin, tan, etc.)
    # also, 'ad.admath' can't handle lists/arrays, so x, y, z are expected to be floats here
    from ad import admath as math
    a = 6378137.0
    e = 8.1819190842622e-2

    asq = a ** 2
    esq = e ** 2

    b   = math.sqrt(asq * (1 - esq))
    bsq = b ** 2

    ep  = math.sqrt((asq - bsq)/bsq)
    p   = math.sqrt( (x ** 2) + (y ** 2) )
    th  = math.atan2(a * z, b * p)

    lon = math.atan2(y, x)
    lat = math.atan2( (z + (ep ** 2) * b * (math.sin(th) ** 3) ), (p - esq * a * (math.cos(th) ** 3) ) )
    N = a / ( math.sqrt(1 - esq * (math.sin(lat) ** 2) ) )
    alt = p / math.cos(lat) - N 

    lon = lon * 180 / math.pi
    lat = lat * 180 / math.pi

    return lat, lon, alt

def plot_connectivity_graph(C, min_matches, save_pgf=False):

    import networkx as nx
    G, _, _, _ = build_connectivity_graph(C, min_matches=min_matches)

    if save_pgf:
        fig_width_pt = 229.5  # CVPR
        inches_per_pt = 1.0 / 72.27  # Convert pt to inches
        golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
        fig_width = fig_width_pt * inches_per_pt  # width in inches
        fig_height = fig_width * golden_mean  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'backend': 'pgf', 'axes.labelsize': 8, 'font.size': 8, 'legend.fontsize': 8,
                  'xtick.labelsize': 7, 'ytick.labelsize': 8, 'text.usetex': True, 'figure.figsize': fig_size}
        plt.rcParams.update(params)

    fig = plt.gcf()
    fig.set_size_inches(8, 8)

    # draw all nodes in a circle
    G_pos = nx.circular_layout(G)

    # draw nodes
    nx.draw_networkx_nodes(G, G_pos, node_size=600, node_color='#FFFFFF', edgecolors='#000000')

    # paint subgroup of nodes
    # nx.draw_networkx_nodes(G, G_pos, nodelist=[41,42, 43, 44, 45], node_size=600, node_color='#FF6161',edgecolors='#000000')

    # draw edges and labels
    nx.draw_networkx_edges(G, G_pos)
    nx.draw_networkx_labels(G, G_pos, font_size=12, font_family='sans-serif')

    # show graph and save it as .pgf
    plt.axis('off')
    if save_pgf:
        plt.savefig('graph.pgf', pad_inches=0, bbox_inches='tight', dpi=200)
    plt.show()


def build_connectivity_graph(C, min_matches, verbose=True):

    def connected_component_subgraphs(G):
        for c in nx.connected_components(G):
            yield G.subgraph(c)

    # (1) Build connectivity matrix A, where position (i,j) contains the number of matches between images i and j
    n_cam = int(C.shape[0]/2)
    A, n_correspondences_filt, tmp_pairs = np.zeros((n_cam,n_cam)), [], []
    for im1 in range(n_cam):
        for im2 in range(im1+1,n_cam):
            n_matches = np.sum(1*np.logical_and(1*~np.isnan(C[2*im1, :]), 1*~np.isnan(C[2*im2, :])))
            n_correspondences_filt.append(n_matches)
            tmp_pairs.append((im1,im2))
            A[im1,im2] = n_matches
            A[im2,im1] = n_matches
 
    # (2) Filter graph edges according to the threshold on the number of matches
    pairs_to_draw = []
    matches_per_pair = []
    for i in range(len(tmp_pairs)):
        if n_correspondences_filt[i] > min_matches:
            pairs_to_draw.append(tmp_pairs[i])
            matches_per_pair.append(n_correspondences_filt[i])

    # (3) Draw the graph and save it as a .pgf image
    import networkx as nx

    # create networkx graph
    G=nx.Graph()
    # add edges
    for edge in pairs_to_draw:
        G.add_edge(edge[0], edge[1])

    # get list of connected components (to see if there is any disconnected subgroup)
    G_cc = list(connected_component_subgraphs(G))
    n_cc = len(G_cc)
    missing_cams = list(set(np.arange(n_cam)) - set(G_cc[0].nodes))

    if verbose:
        print('Connectivity graph: {} missing cameras: {}'.format(len(missing_cams), missing_cams))
        print('                    {} connected components'.format(n_cc))
        print('                    {} edges'.format(len(pairs_to_draw)))
        print('                    {} min n_matches in an edge\n'.format(min(matches_per_pair)))

    return G, n_cc, pairs_to_draw, matches_per_pair





def plot_dsm(fname, vmin=None, vmax=None, color_bar='jet', save_pgf=False):
    
    f_size = (13,10)
    fig = plt.figure()
    params = {'backend': 'pgf',
              'axes.labelsize': 22,
              'ytick.labelleft': False,
              'font.size': 22,
              'legend.fontsize': 22,
              'xtick.labelsize': 22,
              'ytick.labelsize': 22,
              'xtick.top': False,
              'xtick.bottom': False,
              'xtick.labelbottom': False,
              'ytick.left': False,
              'ytick.right': False,   
              'text.usetex': True, # use TeX for text
              'font.family': 'serif',
              'legend.loc': 'upper left',
              'legend.fontsize': 22}
    plt.rcParams.update(params)
    plt.figure(figsize=f_size)
    
    im = np.array(Image.open(fname))
    vmin_in = np.min(im.squeeze()) if vmin is None else vmin
    vmax_in = np.max(im.squeeze()) if vmax is None else vmin
    plt.imshow(im.squeeze(), cmap=color_bar, vmin=vmin, vmax=vmax)
    plt.axis('equal')
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=40)
    if save_pgf:
        plt.savefig(os.path.splitext(fname) + '.pgf', pad_inches=0, bbox_inches='tight', dpi=200)
    plt.show()


def display_ba_error_particular_view(P_before, P_after, pts3d_before, pts3d_after, pts2d, image):

    n_pts = pts3d_before.shape[0]

    # reprojections before bundle adjustment
    proj = P_before @ np.hstack((pts3d_before, np.ones((n_pts,1)))).T
    pts_reproj_before = (proj[:2,:]/proj[-1,:]).T

    # reprojections after bundle adjustment
    proj = P_after @ np.hstack((pts3d_after, np.ones((n_pts,1)))).T
    pts_reproj_after = (proj[:2,:]/proj[-1,:]).T

    err_before = np.sum(abs(pts_reproj_before - pts2d), axis=1)
    err_after = np.sum(abs(pts_reproj_after - pts2d), axis=1)

    print('Mean abs reproj error before BA: {:.4f}'.format(np.mean(err_before)))
    print('Mean abs reproj error after  BA: {:.4f}'.format(np.mean(err_after)))

    # reprojection error histograms for the selected image
    fig = plt.figure(figsize=(10,3))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.title.set_text('Reprojection error before BA')
    ax2.title.set_text('Reprojection error after  BA')
    ax1.hist(err_before, bins=40); 
    ax2.hist(err_after, bins=40);

    plt.show()

    # warning: this is slow...

    fig = plt.figure(figsize=(20,6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.title.set_text('Before BA')
    ax2.title.set_text('After  BA')
    ax1.imshow(image, cmap="gray")
    ax2.imshow(image, cmap="gray")
    for k in range(min(1000,n_pts)):
        # before bundle adjustment
        ax1.plot([pts2d[k,0], pts_reproj_before[k,0] ], [pts2d[k,1], pts_reproj_before[k,1]], 'r-')
        ax1.plot(*pts2d[k], 'yx')
        # after bundle adjustment
        ax2.plot([pts2d[k,0], pts_reproj_after[k,0] ], [pts2d[k,1], pts_reproj_after[k,1]], 'r-')
        ax2.plot(*pts2d[k], 'yx')
    plt.show()


def get_image_footprints(myrpcs, crop_offsets):
    
    from bundle_adjust import geojson_utils
    footprints = []
    for cam_idx, (rpc, offset) in enumerate(zip(myrpcs, crop_offsets)):
        footprint_lonlat_geojson = geojson_utils.lonlat_geojson_from_geotiff_crop(rpc, offset)
        z_footprint = srtm4.srtm4(rpc.lon_offset, rpc.lat_offset)
        footprint_utm_geojson = geojson_utils.utm_geojson_from_lonlat_geojson(footprint_lonlat_geojson)
        footprints.append({'poly': shape(footprint_utm_geojson), 'z': z_footprint})
        #print('\r{} / {} done'.format(cam_idx+1, len(myrpcs)), end = '\r')
    return footprints


def compute_sift_order(C, output_dir):

    n_cam = int(C.shape[0]/2)
    n_correspondences_filt, tmp_pairs = [], []
    for im1 in range(n_cam):
        for im2 in range(im1+1,n_cam):
            obs_im1 = 1*np.invert(np.isnan(C[2*im1,:]))
            obs_im2 = 1*np.invert(np.isnan(C[2*im2,:]))
            n_matches = np.sum(np.sum(np.vstack((obs_im1, obs_im2)), axis=0) == 2)
            n_correspondences_filt.append(n_matches)
            tmp_pairs.append((im1,im2))

    n_correspondences_filt = np.array(n_correspondences_filt)

    sorted_indices = np.flip(np.argsort(n_correspondences_filt))

    output_fname = os.path.join(output_dir, 'sift_order.txt')
    F = open(output_fname,'w')
    for index in sorted_indices:
        F.write('{} {}\n'.format(tmp_pairs[index][0]+1, tmp_pairs[index][1]+1))
    F.close()

    print('sift order successfully saved at {}\n'.format(output_fname))

    



def save_ply_pts_projected_over_geotiff_as_svg(geotiff_fname, ply_fname, output_svg_fname, verbose=False):
    
    # points in the ply file are assumed to be in ecef coordinates
    
    from IS18 import utils
    from bundle_adjust.data_loader import read_geotiff_metadata
    from feature_tracks.ft_utils import save_pts2d_as_svg
    
    utm_bbx, _, resolution, height, width = read_geotiff_metadata(geotiff_fname)
    xyz = read_point_cloud_ply(ply_fname)

    lats, lons, h = ecef_to_latlon_custom(xyz[:,0], xyz[:,1], xyz[:,2])
    easts, norths = utils.utm_from_lonlat(lons, lats)

    offset = np.zeros(len(norths)).astype(np.float32)
    offset[norths < 0] = 10e6
    cols = ( (easts - utm_bbx['xmin'])/resolution ).astype(int)
    rows = ( height - ((norths + offset) - utm_bbx['ymin'])/resolution ).astype(int)
    pts2d_ba_all = np.vstack([cols, rows]).T

    # keep only those points inside the geotiff limits
    valid_cols = np.logical_and(cols < width, cols >= 0)
    valid_rows = np.logical_and(rows < height, rows >= 0)
    pts2d_ba = pts2d_ba_all[np.logical_and(valid_cols, valid_rows), :]

    if verbose:
        fig = plt.figure(figsize=(10,10))
        plt.imshow(np.array(Image.open(geotiff_fname)), cmap="gray")
        plt.scatter(x=pts2d_ba[:,0], y=pts2d_ba[:,1], c='r', s=4)
        plt.show()
        
    save_pts2d_as_svg(output_svg_fname, pts2d_ba, c='yellow')


def close_small_holes_from_dsm(input_geotiff_fname, output_geotiff_fname, imscript_bin_dir):

    # small hole interpolation by closing
    args = [input_geotiff_fname, output_geotiff_fname, imscript_bin_dir]
    cmd = '{2}/morsi square closing {0} | {2}/plambda {0} - "x isfinite x y isfinite y nan if if" -o {1}'.format(*args)
    os.system(cmd)

    # read imscript result
    with rasterio.open(output_geotiff_fname) as imscript_data:
        cdsm_array = imscript_data.read(1)
    os.system('rm {}'.format(output_geotiff_fname))

    # write imscript result with same metadata as input geotiff
    with rasterio.open(input_geotiff_fname) as src_data:
        kwds = src_data.profile
        with rasterio.open(output_geotiff_fname, 'w', **kwds) as dst_data:
            dst_data.write(cdsm_array.astype(rasterio.float32), 1)


def rpc_rpcm_to_geotiff_format(input_dict):
    output_dict = {}

    output_dict['LINE_OFF'] = str(input_dict['row_offset'])
    output_dict['SAMP_OFF'] = str(input_dict['col_offset'])
    output_dict['LAT_OFF'] = str(input_dict['lat_offset'])
    output_dict['LONG_OFF'] = str(input_dict['lon_offset'])
    output_dict['HEIGHT_OFF'] = str(input_dict['alt_offset'])

    output_dict['LINE_SCALE'] = str(input_dict['row_scale'])
    output_dict['SAMP_SCALE'] = str(input_dict['col_scale'])
    output_dict['LAT_SCALE'] = str(input_dict['lat_scale'])
    output_dict['LONG_SCALE'] = str(input_dict['lon_scale'])
    output_dict['HEIGHT_SCALE'] = str(input_dict['alt_scale'])

    output_dict['LINE_NUM_COEFF'] = str(input_dict['row_num'])[1:-1].replace(',', '')
    output_dict['LINE_DEN_COEFF'] = str(input_dict['row_den'])[1:-1].replace(',', '')
    output_dict['SAMP_NUM_COEFF'] = str(input_dict['col_num'])[1:-1].replace(',', '')
    output_dict['SAMP_DEN_COEFF'] = str(input_dict['col_den'])[1:-1].replace(',', '')
    if 'lon_num' in input_dict:
        output_dict['LON_NUM_COEFF'] = str(input_dict['lon_num'])[1:-1].replace(',', '')
        output_dict['LON_DEN_COEFF'] = str(input_dict['lon_den'])[1:-1].replace(',', '')
        output_dict['LAT_NUM_COEFF'] = str(input_dict['lat_num'])[1:-1].replace(',', '')
        output_dict['LAT_DEN_COEFF'] = str(input_dict['lat_den'])[1:-1].replace(',', '')

    return output_dict


def run_plyflatten(ply_list, resolution, utm_bbx, output_file, aoi_lonlat=None, std=False):
    from plyflatten import plyflatten_from_plyfiles_list

    # compute roi from utm bounding box
    xoff = np.floor(utm_bbx['xmin'] / resolution) * resolution
    # xsize = int(1 + np.floor((utm_bbx['xmax'] - xoff) / resolution))
    xsize = int(np.floor((utm_bbx['xmax'] - utm_bbx['xmin']) / resolution) + 1) # width
    yoff = np.ceil(utm_bbx['ymax'] / resolution) * resolution
    # ysize = int(1 - np.floor((utm_bbx['ymin']  - yoff) / resolution))
    ysize = int(np.floor((utm_bbx['ymax'] - utm_bbx['ymin']) / resolution) + 1) # height
    roi = (xoff, yoff, xsize, ysize)

    # run plyflatten
    raster, profile = plyflatten_from_plyfiles_list(ply_list, resolution, roi=roi, std=std)

    # if aoi_lonlat is not None, then mask those parts outside the area of interest with NaN values
    from bundle_adjust.data_loader import apply_mask_to_raster, get_binary_mask_from_aoi_lonlat_within_utm_bbx
    if aoi_lonlat is None:
        mask = np.ones(raster.shape)
    else:
        mask = get_binary_mask_from_aoi_lonlat_within_utm_bbx(utm_bbx, resolution, aoi_lonlat)

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
            cnt_path = os.path.join(os.path.dirname(output_file), 'cnt/'+os.path.basename(output_file))
            os.makedirs(os.path.dirname(cnt_path), exist_ok=True)
            with rasterio.open(cnt_path, "w", **profile) as f:
                f.write(apply_mask_to_raster(raster[:, :, -1], mask), 1)
                raster = raster[:, :, :-1]

        # write remaining extra layers with the std
        std_path = os.path.join(os.path.dirname(output_file), 'std/'+os.path.basename(output_file))
        os.makedirs(os.path.dirname(std_path), exist_ok=True)
        n = raster.shape[2]
        assert n % 2 == 0
        with rasterio.open(std_path, "w", **profile) as f:
            f.write(apply_mask_to_raster(raster[:, :, n // 2], mask), 1)


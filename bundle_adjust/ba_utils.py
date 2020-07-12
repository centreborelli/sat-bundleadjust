"""
* Bundle Adjustment (BA) for 3D Reconstruction from Multi-Date Satellite Images
* Based on https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
* by Roger Mari <mari@cmla.ens-cachan.fr>
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

from IS18 import rectification
from IS18 import vistools
from IS18 import stereo
from IS18 import utils

import cv2
import re
import math
import os
from PIL import Image
import srtm4
import rpcm
from shapely.geometry import Polygon, mapping, shape
import geojson
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

def save_rpc(rpc, filename):
    with open(filename, 'w') as f_out:
        fout.write('LINE_OFF: {}\nSAMPLE_OFF: {}\n'.format(rpc.linOff, rpc.colOff))
        fout.write('LAT_OFF: {}\nLONG_OFF: {}\nHEIGHT_OFF{}\n'.format(rpc.latOff, rpc.lonOff, rpc.altOff))
        fout.write('LINE_SCALE: {}\nSAMPLE_SCALE: {}\n'.format(rpc.linScale, rpc.colScale))
        fout.write('LAT_SCALE: {}\nLONG_SCALE: {}\nHEIGHT_SCALE{}\n'.format(rpc.latScale, rpc.lonScale, rpc.altScale))
        for n in range(1,21):
            fout.write('LINE_NUM_COEFF_{}: {}\n'.format(n, rpc.inverseLinNum[n]))
        for n in range(1,21):
            fout.write('LINE_DEN_COEFF_{}: {}\n'.format(n, rpc.inverseLinDen[n]))
        for n in range(1,21):
            fout.write('SAMP_NUM_COEFF_{}: {}\n'.format(n, rpc.inverseColNum[n]))
        for n in range(1,21):
            fout.write('SAMP_DEN_COEFF_{}: {}\n'.format(n, rpc.inverseColDen[n]))

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

def plot_connectivity_graph(C, thr_matches, save_pgf=False):
    
    def connected_component_subgraphs(G):
        for c in nx.connected_components(G):
            yield G.subgraph(c)
    
    
    # (1) Build connectivity matrix A, where position (i,j) contains the number of matches between images i and j
    n_cam = int(C.shape[0]/2)
    A, n_correspondences_filt, tmp_pairs = np.zeros((n_cam,n_cam)), [], []
    for im1 in range(n_cam):
        for im2 in range(im1+1,n_cam):
            obs_im1 = 1*np.invert(np.isnan(C[2*im1,:]))
            obs_im2 = 1*np.invert(np.isnan(C[2*im2,:]))
            n_matches = np.sum(np.sum(np.vstack((obs_im1, obs_im2)), axis=0) == 2)
            n_correspondences_filt.append(n_matches)
            tmp_pairs.append((im1,im2))
            A[im1,im2] = n_matches
            A[im2,im1] = n_matches
 
    # (2) Filter graph edges according to the threshold on the number of matches
    pairs_to_draw = []
    for i in range(len(tmp_pairs)):
        if n_correspondences_filt[i] > thr_matches:
            pairs_to_draw.append(tmp_pairs[i])

    # (3) Draw the graph and save it as a .pgf image
    import networkx as nx
    
    if save_pgf:
        fig_width_pt = 229.5 # CVPR
        inches_per_pt = 1.0/72.27               # Convert pt to inches
        golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
        fig_width = fig_width_pt*inches_per_pt  # width in inches
        fig_height =fig_width*golden_mean       # height in inches
        fig_size = [fig_width,fig_height]
        params = {'backend': 'pgf', 'axes.labelsize': 8, 'font.size': 8, 'legend.fontsize': 8,
                  'xtick.labelsize': 7, 'ytick.labelsize': 8, 'text.usetex': True, 'figure.figsize': fig_size}
        plt.rcParams.update(params)
    
    fig = plt.gcf()
    fig.set_size_inches(8, 8)

    # create networkx graph
    G=nx.Graph()

    # add edges
    for edge in pairs_to_draw:
        G.add_edge(edge[0], edge[1])
    
    # draw all nodes in a circle
    G_pos = nx.circular_layout(G)
    
    # draw nodes
    nx.draw_networkx_nodes(G, G_pos, node_size=600, node_color='#FFFFFF', edgecolors='#000000')
    
    # paint subgroup of nodes
    #nx.draw_networkx_nodes(G, G_pos, nodelist=[41,42, 43, 44, 45], node_size=600, node_color='#FF6161',edgecolors='#000000')
    
    # draw edges and labels
    nx.draw_networkx_edges(G, G_pos)
    nx.draw_networkx_labels(G, G_pos, font_size=12, font_family='sans-serif')
    
    # get list of connected components (to see if there is any disconnected subgroup)
    G_cc = list(connected_component_subgraphs(G))
    if len(G_cc) > 1:
        print('Attention! Graph G contains {} connected components'.format(len(G_cc)))
    
    # show graph and save it as .pgf
    plt.axis('off')
    if save_pgf:
        plt.savefig('graph.pgf', pad_inches=0, bbox_inches='tight', dpi=200)
    plt.show()

    return A

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

def footprint_from_crop(rpc, x, y, w, h):
    z = srtm4.srtm4(rpc.lon_offset, rpc.lat_offset)
    col0, row0 = x, y
    lons, lats = rpc.localization([col0, col0, col0 + w, col0 + w, col0],
                                  [row0, row0 + h, row0 + h, row0, row0],
                                  [z, z, z, z, z])
    return geojson.Feature(geometry=geojson.Polygon([list(zip(lons, lats))]))

def get_image_footprints(myrpcs, mycrops):
    footprints = []
    for rpc, crop, iter_cont in zip(myrpcs, mycrops, range(len(myrpcs))):
        z_footprint = srtm4.srtm4(rpc.lon_offset, rpc.lat_offset)
        x, y, w, h = crop['col0'], crop['row0'], crop['crop'].shape[1],  crop['crop'].shape[0]
        this_footprint = footprint_from_crop(rpc, x, y, w, h)['geometry']
        this_footprint_lon = np.array(this_footprint["coordinates"][0])[:,0]
        this_footprint_lat = np.array(this_footprint["coordinates"][0])[:,1]
        this_footprint_east, this_footprint_north = utils.utm_from_lonlat(this_footprint_lon, this_footprint_lat)
        this_footprint_utm = np.vstack((this_footprint_east, this_footprint_north)).T
        this_footprint["coordinates"] = [this_footprint_utm.tolist()]
        footprints.append({'poly': shape(this_footprint), 'z': z_footprint})
        #print('\r{} / {} done'.format(iter_cont+1, len(crops)), end = '\r')
    return footprints
    
def rescale_P(input_P, alpha):
    alpha = float(alpha)
    return np.array([[alpha, 0., 0.],[0., alpha, 0.],[0., 0., 1.]]) @ input_P

def rescale_RPC(input_rpc, alpha):
    alpha = float(alpha)
    input_rpc.row_offset *= alpha
    input_rpc.col_offset *= alpha
    input_rpc.row_scale *= alpha
    input_rpc.row_scale *= alpha
    return input_rpc

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

    
def get_image_crops_from_aoi(myimages, aoi, display=False, save_crops=False, output_dir='.'):
    
    # get the altitude of the center of the AOI
    lon, lat = aoi['center']
    alt = srtm4.srtm4(lon, lat)

    # get image crops corresponding to the AOI
    mycrops = []
    for iter_cont, f in enumerate(range(len(myimages))):
        crop, x0, y0 = utils.crop_aoi(myimages[f], aoi, alt)  ### for some reason rpcm.utils.crop_aoi produces bad crops here
        mycrops.append({ 'crop': utils.simple_equalization_8bit(crop), 'col0': x0, 'row0': y0 })
        print('\r{} / {} done'.format(iter_cont+1, len(myimages)), end = '\r')
    print('Finished cropping the AOI in each image')
    if display:
        vistools.display_gallery([f['crop'] for f in mycrops])

    save_crops = True
    if save_crops:
        os.makedirs(output_dir, exist_ok=True)
        for cont, crop in enumerate(mycrops):
            Image.fromarray(crop['crop']).save(output_dir+'/{:02}.tif'.format(cont))
        print('Crops were saved at {}'.format(output_dir))
    return mycrops

def get_aoi_where_at_least_two_crops_overlap(list_aois):
    # INTERSECTION OF PAIRS
    from shapely.ops import cascaded_union
    from shapely.geometry import shape
    from itertools import combinations
    geoms = [shape(g) for g in list_aois]
    geoms = [a.intersection(b) for a, b in combinations(geoms, 2)]
    combined_borders_shapely = cascaded_union([ geom if geom.is_valid else geom.buffer(0) for geom in geoms])
    vertices = (np.array(combined_borders_shapely.boundary.coords.xy).T)[:-1,:]
    output_aoi = {'coordinates': [vertices.tolist()], 'type': 'Polygon'}
    output_aoi['center'] = np.mean(output_aoi['coordinates'][0][:4], axis=0).tolist()
    return output_aoi

def combine_aoi_borders(list_aois):
    # UNION OF ALL AOIS
    from shapely.ops import cascaded_union
    from shapely.geometry import shape
    geoms = [shape(g) for g in list_aois] # convert aois to shapely polygons
    combined_borders_shapely = cascaded_union([ geom if geom.is_valid else geom.buffer(0) for geom in geoms])
    vertices = (np.array(combined_borders_shapely.boundary.coords.xy).T)[:-1,:]
    output_aoi = {'coordinates': [vertices.tolist()], 'type': 'Polygon'}
    output_aoi['center'] = np.mean(output_aoi['coordinates'][0][:4], axis=0).tolist()
    return output_aoi

def display_rois_over_map(list_roi_geojson, zoom_factor = 14):
    mymap = vistools.clickablemap(zoom=zoom_factor)
    for current_aoi in list_roi_geojson:   
        mymap.add_GeoJSON(current_aoi) 
    mymap.center = list_roi_geojson[int(len(list_roi_geojson)/2)]['center'][::-1]
    display(mymap)
'''''''''
this script loads the scenes to be bundle adjusted
such scenes can contain 1 date or multiple dates
'''''''''

import numpy as np
import matplotlib.pyplot as plt

import glob
import os
import json
import rpcm
import datetime
from PIL import Image

from IS18 import utils
from bundle_adjust import ba_utils
from bundle_adjust import geojson_utils


def read_image_size(im_fname):
    
    '''
    this function reads the image width and height without opening the file
    this can be very useful when dealing with huge images
    '''

    import subprocess
    result = subprocess.run(['identify', '-format', '"%wx%h"', im_fname+'[0]'], stdout=subprocess.PIPE)
    im_size = np.array(result.stdout.decode('utf-8').replace('"', '').split('x')).astype(int)
    w, h = im_size[0], im_size[1]
    return h, w
    
    
def add_suffix_to_fname(src_fname, suffix):
    
    '''
    adds a string to a source filename
    '''
    
    src_basename = os.path.basename(src_fname)
    file_id, file_extension = os.path.splitext(src_basename)
    dst_fname = src_fname.replace('/'+src_basename, '/'+file_id+'_'+suffix+file_extension) 
    return dst_fname


def get_id(fname):
     
    '''
    to get file ids (no directory, no extension)
    '''
    
    return os.path.splitext(os.path.basename(fname))[0]


def get_acquisition_date(geotiff_path):
    
    '''
    to read the acquisition date of a geotiff
    '''
    
    with utils.rio_open(geotiff_path) as src:
        if 'TIFFTAG_DATETIME' in src.tags().keys():
            date_string = src.tags()['TIFFTAG_DATETIME']
            dt = datetime.datetime.strptime(date_string, "%Y:%m:%d %H:%M:%S")
        else:
            # temporary fix in case the previous is missing 
            # get datetime from skysat geotiff identifier
            date_string = os.path.basename(geotiff_path)[:15]
            dt = datetime.datetime.strptime(date_string, "%Y%m%d_%H%M%S")
    
    return dt


def save_dict_to_json(input_dict, output_json_fname):
    
    '''
    to save a dict in json format
    '''
    
    with open(output_json_fname, 'w') as f:
        json.dump(input_dict, f, indent = 2)


def load_dict_from_json(input_json_fname):
    
    '''
    to read a json into a python dictionary
    '''
    
    with open(input_json_fname) as f:
        output_dict = json.load(f)
        
    return output_dict


def save_initial_rpcs(all_images_fnames, all_images_rpcs, output_dir):
    
    '''
    copies the initial rpcs in the output directory of the loaded scene
    to make it easier to access them
    '''
    
    init_rpc_dst_dir = os.path.join(output_dir, 'RPC_init')
    os.makedirs(init_rpc_dst_dir, exist_ok=True)
    for tif_fname, rpc in zip(all_images_fnames, all_images_rpcs):
        rpc.write_to_file(os.path.join(init_rpc_dst_dir, get_id(tif_fname) + '_RPC.txt'))


def create_generic_configs(geotiff_fnames, output_dir, dsm_resolution):
    
    # we want to reconstruct overlapping images from the same date 
    
    
    
    config_s2p = {
      "out_dir" : output_dir,
      "images" : [
        {"img" : fname1},
        {"img" : fname2}
        ],
      "roi_geojson" : aoi,
      "horizontal_margin": 20,
      "vertical_margin": 5,
      "tile_size" : 300,
      "disp_range_method" : "sift",
      "msk_erosion": 0,
      "dsm_resolution": dsm_resolution,
      "matching_algorithm": 'mgm_multi'
    }
        
        
def load_scene_from_s2p_configs(images_dir, s2p_configs_dir, output_dir, rpc_src='s2p_configs'):   

    '''
    works for the morenci or RB_ZAF_0001 examples
    in general this is what should be used in the future
    rpc_src can be 's2p_configs', 'geotiff' or the path to a folder containing the rpc files with suffix '_RPC.TXT'
    returns (1) scene timeline  (2) area of interest
    ''' 
    
    all_images_fnames = []
    all_images_rpcs = []
    all_images_datetimes = []

    # get all image fnames used by s2p and their rpcs

    config_fnames = glob.glob(os.path.join(s2p_configs_dir, '**/config.json'), recursive=True)

    seen_images = []
    for fname in config_fnames:
        d = load_dict_from_json(fname)

        for view in d['images']:
            img_basename = os.path.basename(view['img'])
            if img_basename not in seen_images:
                seen_images.append(img_basename)

                img_geotiff_path = glob.glob('{}/**/{}'.format(images_dir, img_basename), recursive=True)
                if len(img_geotiff_path) == 0:
                    # skysat L1A products have different file id from L1B products
                    # e.g. 20190128_204710_ssc6d1_0017_basic_panchromatic_dn.tif for L1B
                    # is equivalent to 20190128_204710_ssc6d1_0017_basic_l1a_panchromatic_dn.tif for L1A
                    tmp = img_basename.replace('basic_panchromatic', 'basic_l1a_panchromatic')
                    img_geotiff_path = glob.glob('{}/**/{}'.format(images_dir, tmp), recursive=True)[0]
                else:
                    img_geotiff_path = img_geotiff_path[0]
                
                # load rpc
                if rpc_src == 's2p_configs':
                    rpc = rpcm.RPCModel(view['rpc'],  dict_format='rpcm')
                elif rpc_src == 'geotiff':
                    rpc = rpcm.rpc_from_geotiff(img_geotiff_path)
                else:
                    rpc = rpcm.rpc_from_rpc_file(os.path.join(images_dir, f_id + '_RPC.TXT')) 

                all_images_fnames.append(img_geotiff_path)
                all_images_rpcs.append(rpc)
                all_images_datetimes.append(get_acquisition_date(img_geotiff_path))

    # copy initial rpcs
    save_initial_rpcs(all_images_fnames, all_images_rpcs, output_dir)
                
    # define timeline and aoi
    timeline = group_files_by_date(all_images_datetimes, all_images_fnames, all_images_rpcs)
    aoi_lonlat = load_aoi_from_s2p_configs(s2p_configs_dir)
    
    return timeline, aoi_lonlat
    

def load_scene_from_geotiff_dir(geotiff_dir, output_dir, rpc_src='geotiff'):
    
    '''
    use it to load skysat_L1A_* stuff
    '''
    
    all_images_fnames = []
    all_images_rpcs = []
    all_images_datetimes = []

    geotiff_paths = glob.glob(os.path.join(geotiff_dir, '**/*.tif'), recursive=True)
    for tif_fname in geotiff_paths:
        
        f_id = get_id(tif_fname)
        
        # load rpc
        if rpc_src == 'geotiff':
            rpc = rpcm.rpc_from_geotiff(tif_fname)
        else:
            rpc = rpcm.rpc_from_rpc_file(os.path.join(images_dir, f_id + '_RPC.TXT')) 
            
        all_images_fnames.append(tif_fname)
        all_images_rpcs.append(rpc)
        all_images_datetimes.append(get_acquisition_date(tif_fname))
        
    # copy initial rpcs            
    save_initial_rpcs(all_images_fnames, all_images_rpcs, output_dir)
        
    # define timeline and aoi
    timeline = group_files_by_date(all_images_datetimes, all_images_fnames, all_images_rpcs)
    aoi_lonlat = load_aoi_from_geotiffs(all_images_fnames, rpcs=all_images_rpcs) 
    
    return timeline, aoi_lonlat
    

def load_aoi_from_geotiffs(geotiff_paths, rpcs=None):
        
    if rpcs is None:
        rpcs = [rpcm.rpc_from_geotiff(path_to_geotiff) for path_to_geotiff in geotiff_paths]
    
    n = len(geotiff_paths)
    lonlat_geotiff_footprints = []
    for rpc, path_to_geotiff, im_idx in zip(rpcs, geotiff_paths, np.arange(n)):
        h, w = read_image_size(path_to_geotiff) 
        crop_offset = {'col0': 0.0, 'row0': 0.0, 'width': w, 'height': h}
        lonlat_geotiff_footprints.append(geojson_utils.lonlat_geojson_from_geotiff_crop(rpc, crop_offset))
        print('\rDefining aoi from the union of all geotiff footprints... {} / {} done'.format(im_idx+1, n), end='\r')
    print('\n')
        
    return geojson_utils.combine_lonlat_geojson_borders(lonlat_geotiff_footprints)

    
def load_aoi_from_s2p_configs(s2p_configs_dir):
    
    config_fnames = glob.glob(os.path.join(s2p_configs_dir, '**/config.json'), recursive=True)
    
    n = len(config_fnames)
    config_aois = []
    for config_idx, fname in enumerate(config_fnames):
        d = load_dict_from_json(fname)
        current_aoi = d['roi_geojson']
        current_aoi['center'] = np.mean(current_aoi['coordinates'][0][:4], axis=0).tolist()
        config_aois.append(current_aoi)        
        print('\rDefining aoi from s2p config.json files... {} / {} done'.format(config_idx+1, n), end='\r')
    print('\n')
    
    return geojson_utils.combine_lonlat_geojson_borders(config_aois)
  
    
def group_files_by_date(datetimes, image_fnames, image_rpcs=None):
    
    '''
    this function picks a list of image fnames, a list with the corresponding acquisition dates,
    a list with the corresponding rpc, and returns the timeline of the set
    i.e. the data is grouped by acquisition date
    '''

    # sort images according to the acquisition date
    sorted_indices = np.argsort(datetimes)
    sorted_datetimes = np.array(datetimes)[sorted_indices].tolist()
    sorted_fnames = np.array(image_fnames)[sorted_indices].tolist()

    '''
    copy_init_rpcs = image_rpcs is not None
    if copy_init_rpcs:
        sorted_rpcs = np.array(image_rpcs)[sorted_indices].tolist()
    '''
    
    margin = 30  # in minutes
    
    # build timeline
    d = {}
    dates_already_seen = []
    for im_idx, fname in enumerate(sorted_fnames):
        
        new_date = True
        
        diff_wrt_prev_dates_in_mins = [abs((x - sorted_datetimes[im_idx]).total_seconds()/60.) for x in dates_already_seen]

        if len(diff_wrt_prev_dates_in_mins) > 0:
            min_pos = np.argmin(diff_wrt_prev_dates_in_mins)
            
            # if this image was acquired within 30 mins of difference w.r.t to an already seen date, 
            # then it is part of the same acquisition
            if diff_wrt_prev_dates_in_mins[min_pos] < 30:
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
        image_fnames_current_datetime = np.array(sorted_fnames)[d[k]].tolist()
        timeline.append({'datetime': current_datetime, 'id': k.split('/')[-1], \
                         'fnames': image_fnames_current_datetime, 'n_images': len(d[k]), \
                         'adjusted': False, 'image_weights': []})


    return timeline


def get_timeline_attributes(timeline, timeline_indices, attributes):
    
    '''
    use this function to display the value of certain attributes at some indices of a given timeline
    '''
    
    max_lens = np.zeros(len(attributes)).tolist()
    for idx in timeline_indices:
        to_display = ''
        for a_idx, a in enumerate(attributes):
            string_len = len('{}'.format(timeline[idx][a]))
            if max_lens[a_idx] < string_len:
                max_lens[a_idx] = string_len
    max_len_idx = max([len(str(idx)) for idx in timeline_indices]) 
    index_str = 'index'
    margin = max_len_idx - len(index_str)
    header_values = [index_str + ' '*margin] if margin > 0 else [index_str]
    for a_idx, a in enumerate(attributes):
        margin = max_lens[a_idx] - len(a)
        header_values.append(a + ' '*margin if margin > 0 else a)
    header_row = '  |  '.join(header_values)
    print(header_row)
    print('_'*len(header_row)+'\n')
    for idx in timeline_indices:
        margin = len(header_values[0]) - len(str(idx))
        to_display = [str(idx) + ' '*margin if margin > 0 else str(idx)]
        for a_idx, a in enumerate(attributes):
            a_value = '{}'.format(timeline[idx][a])
            margin = len(header_values[a_idx+1]) - len(a_value)
            to_display.append(a_value + ' '*margin if margin > 0 else a_value)
        print('  |  '.join(to_display))

    if 'n_images' in attributes: # add total number of images
        print('_'*len(header_row)+'\n')
        to_display = [' '*len(header_values[0])]
        for a_idx, a in enumerate(attributes):
            if a == 'n_images':
                a_value = '{} total'.format(sum([timeline[idx]['n_images'] for idx in timeline_indices]))
                margin = len(header_values[a_idx+1]) - len(a_value)
                to_display.append(a_value + ' '*margin if margin > 0 else a_value)
            else:
                to_display.append(' '*len(header_values[a_idx+1]))
        print('     '.join(to_display))
    print('\n')
    
    
def get_utm_bbox_from_aoi_lonlat(aoi_lonlat):
    
    '''
    gets the limits of the utm bounding box where the aoi is inscribed
    '''
    
    lons = np.array(aoi_lonlat['coordinates'][0])[:,0]
    lats = np.array(aoi_lonlat['coordinates'][0])[:,1]
    easts, norths = utils.utm_from_latlon(lats, lons)
    norths[norths < 0] = norths[norths < 0] + 10000000
    xmin = easts.min()
    xmax = easts.max()
    ymin = norths.min()
    ymax = norths.max()
    
    utm_bbx = {'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax}
    
    return utm_bbx


def mask_from_shapely_polygons(polygons, im_size):
    
    '''
    converts a polygon or multipolygon list back to an image mask ndarray
    '''
    
    import cv2
    
    img_mask = np.zeros(im_size, np.uint8)
    if not polygons:
        return img_mask
    # function to round and convert to int
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
    interiors = [int_coords(pi.coords) for poly in polygons
                 for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    
    return img_mask


def get_binary_mask_from_aoi_lonlat_within_utm_bbx(utm_bbx, resolution, aoi_lonlat):
    
    '''
    gets a binary mask of the utm grid 
    with 1 in those points inside the area of interest and 0 in those points outisde of it
    '''
    
    width = int(np.floor( (utm_bbx['xmax'] - utm_bbx['xmin'])/resolution ) + 1)
    height = int(np.floor( (utm_bbx['ymax'] - utm_bbx['ymin'])/resolution ) + 1)
    
    lonlat_coords = np.array(aoi_lonlat['coordinates'][0])
    lats, lons = lonlat_coords[:,1], lonlat_coords[:,0]
    easts, norths = utils.utm_from_latlon(lats, lons)
    
    offset = np.zeros(len(norths)).astype(np.float32)
    offset[norths < 0] = 10e6
    rows = ( height - ((norths + offset) - utm_bbx['ymin'])/resolution ).astype(int)
    cols = ( (easts - utm_bbx['xmin'])/resolution ).astype(int)
    poly_verts_colrow = np.vstack([cols, rows]).T
   
    from shapely.geometry import shape
    shapely_poly = shape({'type': 'Polygon', 'coordinates': [poly_verts_colrow.tolist()]})
    mask = mask_from_shapely_polygons([shapely_poly], (height, width))
    
    return mask


def get_binary_mask_from_aoi_lonlat_within_image(geotiff_fname, geotiff_rpc, aoi_lonlat):
    
    '''
    gets a binary mask within the limits of a geotiff image
    with 1 in those points inside the area of interest and 0 in those points outisde of it
    '''
    
    im = np.array(Image.open(geotiff_fname)).astype(np.float32)
    h, w = im.shape[:2]
    
    lonlat_coords = np.array(aoi_lonlat['coordinates'][0])
    lats, lons = lonlat_coords[:,1], lonlat_coords[:,0]
    poly_verts_colrow = np.array([geotiff_rpc.projection(lon, lat, 0.) for lon, lat in zip(lons, lats)])
    
    from shapely.geometry import shape
    shapely_poly = shape({'type': 'Polygon', 'coordinates': [poly_verts_colrow.tolist()]})
    mask = mask_from_shapely_polygons([shapely_poly], (h,w))
    
    return mask

    

def load_image_crops(geotiff_fnames, rpcs=None, 
                     aoi=None, crop_aoi=False, get_aoi_mask=False, use_aoi_mask_for_equalization=False, verbose=True):
    
    '''
    loads the crop of interest of each image 
    '''
    
    crops = []
    compute_masks =  (get_aoi_mask and rpcs is not None and aoi is not None)
    
    for im_idx, path_to_geotiff in enumerate(geotiff_fnames):
        
        if aoi is not None and crop_aoi:
            # get the altitude of the center of the AOI
            lon, lat = aoi['center']
            alt = srtm4.srtm4(lon, lat)
            im, x0, y0 = utils.crop_aoi(path_to_geotiff, aoi, alt) 
        else:
            im = np.array(Image.open(path_to_geotiff)).astype(np.float32)
            x0, y0 = 0.0, 0.0
            
        crop = custom_equalization(im)
        
        if compute_masks:
            crop_mask = get_binary_mask_from_aoi_lonlat_within_image(path_to_geotiff, rpcs[im_idx], aoi)
            if use_aoi_mask_for_equalization:
                crop = custom_equalization(im, mask=crop_mask)
        
        crops.append({ 'crop': crop, 'col0': x0, 'row0': y0, 'width': im.shape[1], 'height': im.shape[0]})
        if compute_masks:
            crops[-1]['mask'] = crop_mask
        if verbose:
            print('\rLoading {} image crops / {}'.format(im_idx+1, len(geotiff_fnames)), end='\r')
    if verbose:
        print('\n')
    return crops


def load_rpcs_from_dir(image_fnames_list, rpc_dir, suffix='RPC_adj', verbose=True):
    
    '''
    loads rpcs from rpc files stored in a common directory
    '''
    
    rpcs = []
    for im_idx, fname in enumerate(image_fnames_list):
        image_id = os.path.splitext(os.path.basename(fname))[0]
        path_to_rpc = os.path.join(rpc_dir, image_id + '_{}.txt'.format(suffix))
        
        rpcs.append(rpcm.rpc_from_rpc_file(path_to_rpc))
        
        if verbose:
            print('\rLoading {} image rpcs / {}'.format(im_idx+1, len(image_fnames_list)), end='\r')
    if verbose:
        print('\n')
    return rpcs


def load_matrices_from_dir(image_fnames_list, P_dir, suffix='pinhole_adj', verbose=True):
    
    '''
    loads projection matrices from json files stored in a common directory
    '''
    
    proj_matrices = []
    for im_idx, fname in enumerate(image_fnames_list):
        image_id = os.path.splitext(os.path.basename(fname))[0]
        path_to_P = os.path.join(P_dir, image_id + '_{}.json'.format(suffix))
        with open(path_to_P,'r') as f:
            P_img = np.array(json.load(f)['P'])
        proj_matrices.append(P_img/P_img[2,3])
        if verbose:
            print('\rLoading {} image projection matrices / {}'.format(im_idx+1, len(image_fnames_list)), end='\r')
    if verbose:
        print('\n')
    return proj_matrices


def load_offsets_from_dir(image_fnames_list, P_dir, suffix='pinhole_adj', verbose=True):
    
    '''
    loads offsets from json files stored in a common directory
    '''
    
    crop_offsets = []
    for im_idx, fname in enumerate(image_fnames_list):
        image_id = os.path.splitext(os.path.basename(fname))[0]
        path_to_P = os.path.join(P_dir, image_id + '_{}.json'.format(suffix))
        with open(path_to_P,'r') as f:
            d = json.load(f)
            crop_offsets.append({'col0': d['col_offset'], 'row0': d['col_offset'],
                                 'width': d['width'], 'height': d['height']})
    return crop_offsets


def custom_equalization(im, mask=None, clip=True, percentiles=5):
    
    '''
    im is a numpy array
    returns a numpy array
    '''
    
    if mask is not None:
        valid_domain = mask > 0
    else:
        valid_domain = np.isfinite(im)
        
    if clip:
        mi, ma = np.percentile(im[valid_domain], (percentiles,100-percentiles))
    else:
        mi, ma = im[valid_domain].min(), im[valid_domain].max()
    im = np.minimum(np.maximum(im,mi), ma) # clip
    im = (im-mi)/(ma-mi)*255.0   # scale 
    
    return im


def epsg_from_utm_zone(utm_zone, datum='WGS84'):
    
    '''
    returns the epsg code given the string of a utm zone
    '''
    
    from pyproj import CRS
    args = [utm_zone[:2], '+south' if utm_zone[-1] == 'S' else '+north', datum]
    crs = CRS.from_proj4('+proj=utm +zone={} {} +datum={}'.format(*args))
    return crs.to_epsg()
    
    
def load_s2p_configs_from_image_filenames(im_fnames, s2p_configs_dir):
    
    '''
    returns all config.json fnames in the s2p_configs_dir
    where both images are part of the im_fnames list
    '''
    
    fnames = [os.path.basename(fn) for fn in im_fnames]
    config_fnames = glob.glob(os.path.join(s2p_configs_dir, '**/config.json'), recursive=True)
              
    selected_config_fnames = []
    for s2p_config_fn in config_fnames:
        d = load_dict_from_json(s2p_config_fn)
        
        if os.path.basename(d['images'][0]['img']) in fnames and os.path.basename(d['images'][1]['img']) in fnames:
            selected_config_fnames.append(s2p_config_fn)
                
    return selected_config_fnames


def load_s2p_dsm_fnames_from_dir(s2p_dir):

    '''
    returns the filenames of all dsms within a directory containing s2p outputs
    dsm.tif in tiles directories are ignored
    '''
    
    all_dsm_fnames = glob.glob(s2p_dir+'/**/dsm.tif', recursive=True)
    dsm_fnames = [fn for fn in all_dsm_fnames if '/tiles/' not in fn]
    
    return dsm_fnames


def read_geotiff_metadata(geotiff_fname):

    '''
    read geotiff metadata
    '''
    
    import rasterio
       
    # reconstructed dsms have to present the following parameters
    with rasterio.open(geotiff_fname) as src:
        #dsm_data = src.read()[0,:,:]
        dsm_metadata = src
    xmin = dsm_metadata.bounds.left
    ymin = dsm_metadata.bounds.bottom
    xmax = dsm_metadata.bounds.right
    ymax = dsm_metadata.bounds.top
    epsg = dsm_metadata.crs
    resolution = dsm_metadata.res
    h, w = dsm_data.shape
    utm_bbx = {'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax}
    
    return utm_bbx, epsg, resolution, h, w

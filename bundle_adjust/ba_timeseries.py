import numpy as np
import matplotlib.pyplot as plt

import json
import glob
import rpcm
import os
import rasterio
import pickle
import subprocess
import srtm4
from PIL import Image

from IS18 import vistools
from IS18 import utils

from bundle_adjust import ba_core
from bundle_adjust import ba_utils
from bundle_adjust.ba_pipeline import BundleAdjustmentPipeline

from contextlib import contextmanager
import sys

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout


def custom_equalization(im, mask=None, clip=True, percentiles=5):
    ''' im is a numpy array
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


def mask_from_shapely_polygons(polygons, im_size):
    import cv2
    """Convert a polygon or multipolygon list back to
       an image mask ndarray"""
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

def get_binary_mask_from_dsm_geotiff(geotiff_fname, aoi_lonlat):
    
    with rasterio.open(geotiff_fname) as src:
        geotiff_data = src.read()[0,:,:]
        geotiff_metadata = src
    
    h, w = geotiff_data.shape
    lonlat_coords = np.array(aoi_lonlat['coordinates'][0])
    lats, lons = lonlat_coords[:,1], lonlat_coords[:,0]
    east, north = utils.utm_from_latlon(lats, lons)
    poly_verts_colrow = np.fliplr(np.array([geotiff_metadata.index(e, n) for e, n in zip(east, north + 1e7)]))
    
    from shapely.geometry import shape
    shapely_poly = shape({'type': 'Polygon', 'coordinates': [poly_verts_colrow.tolist()]})
    mask = mask_from_shapely_polygons([shapely_poly], (h,w))
    
    return mask

def get_binary_mask_from_image_geotiff(geotiff_fname, geotiff_rpc, aoi_lonlat):
    
    with rasterio.open(geotiff_fname) as src:
        geotiff_data = src.read()[0,:,:]
        
    h, w = geotiff_data.shape
    lonlat_coords = np.array(aoi_lonlat['coordinates'][0])
    lats, lons = lonlat_coords[:,1], lonlat_coords[:,0]
    poly_verts_colrow = np.array([geotiff_rpc.projection(lon, lat, 0.) for lon, lat in zip(lons, lats)])
    
    from shapely.geometry import shape
    shapely_poly = shape({'type': 'Polygon', 'coordinates': [poly_verts_colrow.tolist()]})
    mask = mask_from_shapely_polygons([shapely_poly], (h,w))
    
    return mask

def json2dict(input_json_fname):
    with open(input_json_fname) as f:
        output_dict = json.load(f)
    return output_dict

def get_dict_from_rpcm(rpc):
    d = {}
    d['row_den'] = rpc.row_den
    d['row_scale'] = rpc.row_scale
    d['lat_scale'] = rpc.lat_scale
    d['lat_offset'] = rpc.lat_offset
    d['row_offset'] = rpc.row_offset
    d['col_offset'] = rpc.col_offset
    d['alt_offset'] = rpc.alt_offset
    d['alt_scale'] = rpc.alt_scale
    d['row_num'] = rpc.row_num
    d['col_scale'] = rpc.col_scale
    d['col_num'] = rpc.col_num
    d['lon_offset'] = rpc.lon_offset
    d['lon_scale'] = rpc.lon_scale
    d['col_den'] = rpc.col_den
    return d

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
    
def get_acquisition_date(geotiff_path):
    import datetime
    with utils.rio_open(geotiff_path) as src:
        date_string = src.tags()['TIFFTAG_DATETIME']
        return datetime.datetime.strptime(date_string, "%Y:%m:%d %H:%M:%S")
    
def load_image_crops(image_fnames_list, get_aoi_mask=False, rpcs=None, aoi=None, use_mask_for_equalization=False):
    
    crops = []
    compute_masks =  (get_aoi_mask and rpcs is not None and aoi is not None)
    
    for im_idx, fname in enumerate(image_fnames_list):
        im = np.array(Image.open(fname)).astype(np.float32)
        crop_already_computed = False
        if compute_masks:
            crop_mask = get_binary_mask_from_image_geotiff(fname, rpcs[im_idx], aoi)
            if use_mask_for_equalization:
                crop = custom_equalization(im, mask=crop_mask)
                crop_already_computed = True
        if not crop_already_computed:
            crop = custom_equalization(im)
        crops.append({ 'crop': crop, 'col0': 0.0, 'row0': 0.0})
        if compute_masks:
            crops[-1]['mask'] = crop_mask
        print('\rLoading {} image crops / {}'.format(im_idx+1, len(image_fnames_list)), end='\r')
    print('\nDone!\n')
    return crops

def load_adjusted_rpcs(image_fnames_list, rpc_dir):
    rpcs = []
    for im_idx, fname in enumerate(image_fnames_list):
        image_id = os.path.splitext(os.path.basename(fname))[0]
        path_to_rpc = os.path.join(rpc_dir, image_id + '_RPC_adj.txt')
        rpcs.append(rpcm.rpc_from_rpc_file(path_to_rpc))
        print('\rLoading {} image rpcs / {}'.format(im_idx+1, len(image_fnames_list)), end='\r')
    print('\nDone!\n')
    return rpcs
   
def load_adjusted_matrices(image_fnames_list, P_dir):
    proj_matrices = []
    for im_idx, fname in enumerate(image_fnames_list):
        image_id = os.path.splitext(os.path.basename(fname))[0]
        path_to_P = os.path.join(P_dir, image_id + '_pinhole_adj.json')
        with open(path_to_P,'r') as f:
            P_img = np.array(json.load(f)['P'])
        proj_matrices.append(P_img/P_img[2,3])
        print('\rLoading {} image projection matrices / {}'.format(im_idx+1, len(image_fnames_list)), end='\r')
    print('\nDone!\n')
    return proj_matrices








class Scene:
    def __init__(self, input_dir, output_dir, name, scene_type,
                 compute_aoi_masks=False, use_aoi_masks_to_equalize_crops=False):
        
        self.scene_type = scene_type
        self.src_dir = os.path.join(input_dir, name)
        if not os.path.isdir(self.src_dir):
            print('\nERROR ! Source directory does not exist')
            return False
        self.name = name
        self.dst_dir = os.path.join(output_dir, name)
        os.makedirs(self.dst_dir, exist_ok=True)
        
        # to be filled when scene is loaded
        self.aoi_lonlat = None
        self.all_images_fnames = []
        self.all_images_rpcs = []
        self.all_images_datetimes = []
        self.timeline = []
        
        # to be filled when scene is characterized
        self.epsg = None
        self.res = None
        self.h = None
        self.w = None
        self.bounds = {'left':None, 'bottom': None, 'right':None, 'top':None}
        self.mask = None
        
        # needed to run bundle adjustment
        self.compute_aoi_masks = compute_aoi_masks
        self.use_aoi_masks_to_equalize_crops = use_aoi_masks_to_equalize_crops
        self.projmats_model='Perspective'
        self.init_ba_input_data()
        self.tracks_config = None
        
        print('#############################################################')
        print('Loading scene {}...\n'.format(self.name))
        
        if self.scene_type == 'v1':
            self.load_scene_v1()
        elif self.scene_type == 'v2':
            self.load_scene_v2()
        else:
            print('ERROR! scene type is not recognized !')
            
        n_images_all = len(self.all_images_fnames)
        print('Found {} images\n'.format(n_images_all))

        if n_images_all > 0:
            self.timeline = self.group_files_by_date(self.all_images_fnames, self.all_images_datetimes)
            print('Found {} different dates in the scene timeline\n'.format(len(self.timeline)))

            print('Successfully loaded scene {}'.format(self.name))
            print('#############################################################\n\n')

     
        
    def load_scene_v1(self):   

        # get all image fnames used by s2p and their rpcs
        dir_src_images = '{}/{}_images'.format(self.src_dir, self.name)
        dir_src_configs = '{}/{}_s2p'.format(self.src_dir, self.name)
        config_fnames = glob.glob(dir_src_configs +  '/**/config.json', recursive=True)

        seen_images, config_aois = [], []
        for fname in config_fnames:
            current_config_dict = json2dict(fname)
         
            current_aoi = current_config_dict['roi_geojson']
            current_aoi['center'] = np.mean(current_aoi['coordinates'][0][:4], axis=0).tolist()
            config_aois.append(current_aoi)
            
            for view in current_config_dict['images']:
                img_basename = os.path.basename(view['img'])
                if img_basename not in seen_images:
                    seen_images.append(img_basename)

                    img_geotiff_path = glob.glob('{}/**/{}'.format(dir_src_images, img_basename), recursive=True)[0]
                    rpc = rpcm.RPCModel(view['rpc'],  dict_format='rpcm')
                    # import rpcm_model
                    # rpc = rpc_model.RPCModel(rpc_rpcm_to_geotiff_format(view['rpc']))
                    self.all_images_fnames.append(img_geotiff_path)
                    self.all_images_rpcs.append(rpc)
                    self.all_images_datetimes.append(get_acquisition_date(img_geotiff_path))

        # sort images according to the acquisition date
        sorted_indices_all = np.argsort(self.all_images_datetimes)
        self.all_images_datetimes = np.array(self.all_images_datetimes)[sorted_indices_all].tolist()
        self.all_images_fnames = np.array(self.all_images_fnames)[sorted_indices_all].tolist()
        self.all_images_rpcs = np.array(self.all_images_rpcs)[sorted_indices_all].tolist()
        self.aoi_lonlat = ba_utils.combine_aoi_borders(config_aois)
                               
    
    def load_scene_v2(self):
        
        geotiff_paths = glob.glob(self.src_dir + '/*.tif')
        for fname in geotiff_paths:
            f_id = os.path.splitext(os.path.basename(fname))[0]
            self.all_images_fnames.append(fname)
            rpc_fname = os.path.join(self.src_dir, f_id + '_RPC.TXT')
            self.all_images_rpcs.append(rpcm.rpc_from_rpc_file(rpc_fname))
            self.all_images_datetimes.append(get_acquisition_date(fname))
            self.aoi_lonlat = None                  
        
    
    def group_files_by_date(self, image_fnames, datetimes):
        d = {}
        for im_idx, fname in enumerate(image_fnames):
            im_dir = os.path.dirname(fname)
            if im_dir in d:
                d[im_dir].append(im_idx)
            else:
                d[im_dir] = [im_idx]
        timeline = []
        for k in d.keys():
            current_datetime = datetimes[d[k][0]]
            timeline.append({'datetime': current_datetime, 'id':k.split('/')[-1], \
                             'image_indices':d[k], 'n_images': len(d[k]), \
                             'adjusted': False, 'n_tracks': 'to compute'})
        return timeline
    
    def get_timeline_attributes(self, timeline_indices, attributes):
        max_lens = np.zeros(len(attributes)).tolist()
        for idx in timeline_indices:
            to_display = ''
            for a_idx, a in enumerate(attributes):
                string_len = len('{}'.format(self.timeline[idx][a]))
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
                a_value = '{}'.format(self.timeline[idx][a])
                margin = len(header_values[a_idx+1]) - len(a_value)
                to_display.append(a_value + ' '*margin if margin > 0 else a_value)
            print('  |  '.join(to_display))
       
        if 'n_images' in attributes: # add total number of images
            print('_'*len(header_row)+'\n')
            to_display = [' '*len(header_values[0])]
            for a_idx, a in enumerate(attributes):
                if a == 'n_images':
                    a_value = '{} total'.format(sum([self.timeline[idx]['n_images'] for idx in timeline_indices]))
                    margin = len(header_values[a_idx+1]) - len(a_value)
                    to_display.append(a_value + ' '*margin if margin > 0 else a_value)
                else:
                    to_display.append(' '*len(header_values[a_idx+1]))
            print('     '.join(to_display))
        print('\n')
                  
    def display_aoi(self):
        ba_utils.display_rois_over_map([self.aoi_lonlat], zoom_factor=14)
    
    def display_crops(self):
        mycrops = self.mycrops_adj + self.mycrops_new
        if len(mycrops) > 0:
            vistools.display_gallery([f['crop'] for f in mycrops])
        else:
            print('No crops have been loaded. Use load_data_from_date() to load them.')
    
    def display_image_masks(self):
        if not self.compute_aoi_masks:
            print('compute_aoi_masks is False')
        else:
            mycrops = self.mycrops_adj + self.mycrops_new
            if len(mycrops) > 0:
                vistools.display_gallery([255*f['mask'] for f in mycrops])
            else:
                print('No crops have been loaded. Use load_data_from_date() to load them.')
        
    
    def characterize_from_example_dsm(self, example_dsm_fname):
       
        # reconstructed dsms have to present the following parameters
        with rasterio.open(example_dsm_fname) as src:
            dsm_data = src.read()[0,:,:]
            dsm_metadata = src
        self.xmin = dsm_metadata.bounds.left
        self.ymin = dsm_metadata.bounds.bottom
        self.xmax = dsm_metadata.bounds.right
        self.ymax = dsm_metadata.bounds.top
        self.epsg = dsm_metadata.crs
        self.res = dsm_metadata.res
        self.h, self.w = dsm_data.shape
        
        # get binary mask
        self.mask = get_binary_mask_from_dsm_geotiff(example_dsm_fname, self.aoi_lonlat)
        Image.fromarray(self.mask.astype(np.float32)).save(os.path.join(self.dst_dir,'mask.tif'))

        print('Scene {} characterized using the following example dsm:'.format(self.name))
        print('{}\n'.format(example_dsm_fname))
    
    
    def display_dsm_mask(self):
        if self.mask is None:
            print('The mask of this scene is not defined')
        else:
            fig = plt.figure(figsize=(10,10))
            plt.imshow(self.mask);
            
            
         
    def check_adjusted_dates(self):
        
        dir_adj_rpc = os.path.join(self.dst_dir, 'RPC_adj')
        dir_src_images = '{}/{}_images'.format(self.src_dir, self.name) if self.scene_type == 'v1' else self.src_dir
        if os.path.exists(self.dst_dir + '/filenames.pickle') and os.path.isdir(dir_adj_rpc):

            # read tiff images 
            adj_fnames = pickle.load(open(self.dst_dir+'/filenames.pickle','rb'))
            print('Found {} previously adjusted images in scene {}\n'.format(len(adj_fnames), self.name))
            
            datetimes_adj = [get_acquisition_date(img_geotiff_path) for img_geotiff_path in adj_fnames]
            timeline_adj = self.group_files_by_date(adj_fnames, datetimes_adj)
            for d in timeline_adj:
                adj_id = d['id']
                adj_date = d['datetime']
                for idx in range(len(self.timeline)):
                    if self.timeline[idx]['id'] == adj_id:
                        self.timeline[idx]['adjusted'] = True
                        
            prev_adj_data_found=True      
        else:
            print('No previously adjusted data was found in scene {}\n'.format(self.name))
            prev_adj_data_found=False
            
        return prev_adj_data_found
            

    def load_data_from_dates(self, timeline_indices, adjusted=False):
        
        t_indices_files = []
        for t_idx in timeline_indices:
            t_indices_files.extend(self.timeline[t_idx]['image_indices'])
        im_fnames = np.array(self.all_images_fnames)[t_indices_files].tolist()
        print(len(im_fnames), '{} images for bundle adjustment ! \n'.format('adjusted' if adjusted else 'new'))
        
        if len(im_fnames) > 0:
            # get rpcs
            if adjusted:
                im_rpcs = load_adjusted_rpcs(im_fnames, os.path.join(self.dst_dir, 'RPC_adj'))
                # load previously adjusted projection matrices
                #self.myprojmats_adj = load_adjusted_matrices(im_fnames, os.path.join(self.dst_dir, 'P_adj'))
            else:
                im_rpcs = np.array(self.all_images_rpcs)[t_indices_files].tolist() 
                print('Loading {} image rpcs / {}'.format(len(im_rpcs), len(im_rpcs)))
                print('Done!\n')

            # get image crops
            im_crops = load_image_crops(im_fnames, get_aoi_mask = self.compute_aoi_masks, \
                                        rpcs = im_rpcs, aoi = self.aoi_lonlat, \
                                        use_mask_for_equalization = self.use_aoi_masks_to_equalize_crops)
            
        if adjusted:
            self.n_adj += len(im_fnames)
            self.myimages_adj.extend(im_fnames.copy())
            self.myrpcs_adj.extend(im_rpcs.copy())
            self.mycrops_adj.extend(im_crops.copy())
        else:
            self.n_new += len(im_fnames)
            self.myimages_new.extend(im_fnames.copy())
            self.myrpcs_new.extend(im_rpcs.copy())
            self.mycrops_new.extend(im_crops.copy())
    
    
    def load_previously_adjusted_dates(self, t_idx, n_previous_dates=1):
        
        # t_idx = timeline index of the new date to adjust
        
        found_adj_dates = self.check_adjusted_dates()
        if found_adj_dates:
        
            # get closest date in time
            prev_adj_timeline_indices = [idx for idx, d in enumerate(self.timeline) if d['adjusted']==True]
            closest_adj_timeline_indices = sorted(prev_adj_timeline_indices, key=lambda x:abs(x-t_idx))

            self.load_data_from_dates(closest_adj_timeline_indices[:n_previous_dates], adjusted=True)
        
    
    def init_ba_input_data(self):
        self.n_adj = 0
        self.myimages_adj = []
        self.mycrops_adj = [] 
        self.myrpcs_adj = [] 
        self.n_new = 0
        self.myimages_new = []
        self.mycrops_new = []
        self.myrpcs_new = [] 
    
    def set_ba_input_data(self, t_indices, tracks_config=None, n_previous_dates=0):
        
        # init
        self.init_ba_input_data()
        # load previously adjusted data (if existent) relevant for the current date
        if n_previous_dates > 0:
            self.load_previously_adjusted_dates(min(t_indices), n_previous_dates=n_previous_dates)
        # load new data to adjust
        self.load_data_from_dates(t_indices)
        
        self.ba_input_data = {}
        self.ba_input_data['input_dir'] = self.dst_dir
        self.ba_input_data['n_new'] = self.n_new
        self.ba_input_data['n_adj'] = self.n_adj
        self.ba_input_data['image_fnames'] = self.myimages_adj + self.myimages_new
        self.ba_input_data['crops'] = self.mycrops_adj + self.mycrops_new
        self.ba_input_data['rpcs'] = self.myrpcs_adj + self.myrpcs_new
        self.ba_input_data['cam_model'] = self.projmats_model
        self.ba_input_data['aoi'] = self.aoi_lonlat
        
        if self.compute_aoi_masks:
            self.ba_input_data['masks'] = [f['mask'] for f in self.mycrops_adj] + [f['mask'] for f in self.mycrops_new]
        else:
            self.ba_input_data['masks'] = None
        
        print('\nBundle Adjustment input data is ready !\n')
            
    
    def bundle_adjust(self, time_indices, n_previous_dates=0, ba_input_data=None, tracks_config=None, verbose=True):

        import timeit
        start = timeit.default_timer()
        
        # run bundle adjustment
        if verbose:
            if ba_input_data is None:
                self.set_ba_input_data(time_indices, n_previous_dates=n_previous_dates)
            else:
                self.ba_input_data = ba_input_data
            self.tracks_config = tracks_config
            self.ba_pipeline = BundleAdjustmentPipeline(self.ba_input_data, tracks_config=self.tracks_config)
            self.ba_pipeline.run()

        else:
            with suppress_stdout():
                if ba_input_data is None:
                    self.set_ba_input_data(time_indices, n_previous_dates=n_previous_dates)
                else:
                    self.ba_input_data = ba_input_data
                self.tracks_config = tracks_config
                self.ba_pipeline = BundleAdjustmentPipeline(self.ba_input_data, tracks_config=self.tracks_config)
                self.ba_pipeline.run()
        
        n_cams = int(self.ba_pipeline.C.shape[0]/2)
        n_tracks_employed = self.ba_pipeline.get_n_tracks_within_group_of_views(np.arange(n_cams))
        
        elapsed_time = int(timeit.default_timer() - start)
        
        return elapsed_time, n_tracks_employed
    
    def reset_ba_params(self):
        os.system('rm -r {}'.format(self.dst_dir))
        os.makedirs(self.dst_dir, exist_ok=True)
        for t_idx in range(len(self.timeline)):
            self.timeline[t_idx]['adjusted'] = False
    
    def print_ba_headline(self, timeline_indices):
        print('Chosen {} dates of the timeline to bundle adjust:'.format(len(timeline_indices)))
        for idx, t_idx in enumerate(timeline_indices):
            print('({}) {} --> {} views'.format(idx+1, self.timeline[t_idx]['datetime'], self.timeline[t_idx]['n_images']))
        print('\n')
    
    def print_running_time(self, in_seconds):
        hours, rem = divmod(in_seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        print('\nTOTAL TIME: {:0>2}:{:0>2}:{:05.2f}\n\n\n'.format(int(hours),int(minutes),seconds))  
        
    
    def run_sequential_bundle_adjustment(self, timeline_indices, n_previous=0, reset=False, verbose=True):

        if reset:
            self.reset_ba_params()
        self.print_ba_headline(timeline_indices)
        
        print('\nRunning bundle ajustment sequentially, each date aligned with {} previous date(s) !'.format(n_previous))
        time_per_date = []
        for idx, t_idx in enumerate(timeline_indices):
            if verbose:
                print('Bundle adjusting date {}...'.format(self.timeline[t_idx]['datetime']))
            running_time, n_tracks = self.bundle_adjust([t_idx], n_previous_dates=n_previous, 
                                                        tracks_config=self.tracks_config,
                                                        verbose=verbose)
            time_per_date.append(running_time)
            print_args = [idx+1, self.timeline[t_idx]['datetime'], running_time, n_tracks]
            print('({}) {} adjusted in {} seconds, {} tracks employed'.format(*print_args))
        print('\n')
        
        self.print_running_time(np.sum(time_per_date))
            
    
    def run_global_bundle_adjustment(self, timeline_indices, reset=False, verbose=True):
    
        if reset:
            self.reset_ba_params()
        self.print_ba_headline(timeline_indices)
        
        print('\nRunning bundle ajustment all at once !')
        running_time, n_tracks = self.bundle_adjust(timeline_indices, n_previous_dates=0, 
                                                    tracks_config=self.tracks_config,
                                                    verbose=verbose)
        print('All dates adjusted in {} seconds, {} tracks employed'.format(running_time, n_tracks))
        
        self.print_running_time(running_time)
    
    
    def extract_ba_data_indices_according_to_date(self, t_idx):
        
        # returns the indices of the files in ba_input_data['myimages'] that belong to a common timeline index
        all_img_indices_t_id = self.timeline[t_idx]['image_indices']
        img_fnames_t_id = np.array(self.all_images_fnames)[all_img_indices_t_id].tolist()
        ba_data_indices_t_id = [self.ba_input_data['image_fnames'].index(fname) for fname in img_fnames_t_id]
        return ba_data_indices_t_id
    
    
    def compare_timeline_instances(self, timeline_indices_g1, timeline_indices_g2, ba_pipeline):
               
        # (3) Draw the graph and save it as a .pgf image
        import networkx as nx
        
        # get dates
        node_labels = {}
        nodes_img_indices_in_ba_data = []
        for node_idx, timeline_indices_g in enumerate([timeline_indices_g1, timeline_indices_g2]):
            g_dates = sorted([self.timeline[t_idx]['datetime'] for t_idx in timeline_indices_g])
            g_n_img = sorted([self.timeline[t_idx]['n_images'] for t_idx in timeline_indices_g])
            label = ''
            n_views = []
            for dt, n_img in zip(g_dates, g_n_img):
                n_views.append(n_img)
                label += '{}-{}-{} {}h\n'.format(dt.day, dt.month, str(dt.year)[-2:], dt.hour)
            label += '{} {}'.format(sum(n_views), n_views)
            node_labels[node_idx] = label
            tmp = [self.extract_ba_data_indices_according_to_date(t_idx) for t_idx in timeline_indices_g]
            g_img_indices = [item for sublist in tmp for item in sublist]
            nodes_img_indices_in_ba_data.append(g_img_indices)
        
        
        #mandatory edges
        pairs_to_draw = [[0, 1]]
        
        # init graph
        G=nx.Graph()
        
        # add edges
        edge_labels = {}
        for edge_idx, node_indices in enumerate(pairs_to_draw):
            
            # add current edge
            i, j = node_indices[0], node_indices[1]
            G.add_edge(i, j)
    
            # get number of matches
            g1_img_indices = nodes_img_indices_in_ba_data[i]
            g2_img_indices = nodes_img_indices_in_ba_data[j]
            n_matches, n_matches_aoi = ba_pipeline.get_number_of_matches_between_groups_of_views(g1_img_indices, \
                                                                                                 g2_img_indices)

            current_edge = (i,j)
            if n_matches_aoi is not None:
                edge_labels[current_edge] = '{} matches\n({} aoi)'.format(n_matches, n_matches_aoi)
            else:
                edge_labels[current_edge] = '{} matches'.format(n_matches)
            
            
        # init node positions from layout
        G_pos = nx.circular_layout(G)
        
        # draw nodes
        nx.draw_networkx_nodes(G, G_pos, node_shape='s', node_size=3000, node_color='#FFFFFF', edgecolors='#000000')
        # draw edges and labels
        nx.draw_networkx_edges(G, G_pos)
        nx.draw_networkx_labels(G, G_pos, node_labels, font_size=8, font_family='sans-serif', with_labels=True)
        nx.draw_networkx_edge_labels(G, G_pos, edge_labels, font_size=8, font_family='sans-serif', with_labels=True)
        plt.axis('off')
        plt.show()
    
    def get_datetime_diff_in_days_hours(self, dt_i, dt_j):
        import time
        di_ts = time.mktime(dt_i.timetuple())
        dj_ts = time.mktime(dt_j.timetuple())
        diff_di_dj_in_mins = int(abs(di_ts - dj_ts) / 60)
        days = diff_di_dj_in_mins / 1440     
        leftover_minutes = diff_di_dj_in_mins % 1440
        hours = leftover_minutes / 60
        total_diff_minutes = diff_di_dj_in_mins
        return days, hours, total_diff_minutes
        
    def timeline_instances_diagram(self, timeline_indices, ba_pipeline, neighbors=1):
        
        # (3) Draw the graph and save it as a .pgf image
        import networkx as nx
        
        # get dates
        n_nodes = len(timeline_indices)
        nodes_dates = sorted([self.timeline[t_idx]['datetime'] for t_idx in timeline_indices])
        nodes_n_img = sorted([self.timeline[t_idx]['n_images'] for t_idx in timeline_indices])
        node_labels = {}
        for node_idx, dt in enumerate(nodes_dates):
            node_labels[node_idx] = '{}-{}\n{}\n{}h {}v'.format(dt.day, dt.month, dt.year, dt.hour, nodes_n_img[node_idx])
        nodes_img_indices_in_ba_data = [self.extract_ba_data_indices_according_to_date(t_idx) for t_idx in timeline_indices] 
        
        
        #mandatory edges
        pairs_to_draw = [[i, i+1] for i in range(n_nodes-1)]
        n_mandatory_edges = len(pairs_to_draw)
        # edges of interest
        pairs_to_draw.extend([[i, i+neighbors] for i in range(n_nodes-neighbors)])
        
        # init graph
        G=nx.Graph()
        
        # add edges
        edge_labels = {}
        edges_of_interest = []
        for edge_idx, node_indices in enumerate(pairs_to_draw):
            
            # add current edge
            i, j = node_indices[0], node_indices[1]
            G.add_edge(i, j)
            
            if edge_idx >= n_mandatory_edges:
                # get number of matches
                g1_img_indices = nodes_img_indices_in_ba_data[i]
                g2_img_indices = nodes_img_indices_in_ba_data[j]
                n_matches, _ = ba_pipeline.get_number_of_matches_between_groups_of_views(g1_img_indices, g2_img_indices)

                # get temporal difference between nodes
                days, hours, _ = self.get_datetime_diff_in_days_hours(nodes_dates[i], nodes_dates[j])
                
                current_edge = (i,j)
                edges_of_interest.append(current_edge)
                edge_labels[current_edge] = '{}m\n{}d {}h'.format(n_matches, int(days), int(hours))
            
            
        # init node positions from layout
        G_pos = nx.circular_layout(G)
        
        # draw nodes
        nx.draw_networkx_nodes(G, G_pos, node_shape='s', node_size=1000, node_color='#FFFFFF', edgecolors='#000000')
        # draw edges and labels
        nx.draw_networkx_edges(G, G_pos, edge_color='w')
        nx.draw_networkx_edges(G, G_pos, edgelist=edges_of_interest)
        nx.draw_networkx_labels(G, G_pos, node_labels, font_size=8, font_family='sans-serif', with_labels=True)
        nx.draw_networkx_edge_labels(G, G_pos, edge_labels, font_size=8, font_family='sans-serif', with_labels=True)
        plt.axis('off')
        plt.show()
        
    
    def reconstruct_date(self, timeline_index, use_corrected_rpcs=True):
        
        t_id, t_date =  self.timeline[timeline_index]['id'], self.timeline[timeline_index]['datetime']
        print('\n###################################################################################')
        print('Reconstructing scene {} at time {}'.format(self.name, t_date))
        print('Timeline id: {}'.format(t_id))
        print('###################################################################################\n')
        
        # load configs
        dir_src_images = '{}/{}_images'.format(self.src_dir, self.name)
        dir_src_configs = '{}/{}_s2p'.format(self.src_dir, self.name)
        src_config_fnames, dst_config_fnames = [], []
        for fname in glob.glob(dir_src_configs +  '/{}/**/config.json'.format(t_id), recursive=True):
            src_config_fnames.append(fname)
            tmp = fname.split('/')
            new_config = os.path.join('{}/{}/{}'.format(os.getcwd(),self.dst_dir,'s2p'), '/'.join(tmp[tmp.index(t_id):]))
            dst_config_fnames.append(new_config)
        n_dsms = len(src_config_fnames)
              
        # run s2p
        print('Running s2p...\n')
        err_indices = []
        adj_rpc_dir = os.path.join(self.dst_dir, 'RPC_adj')
        for dsm_idx, src_config_fname, dst_config_fname in zip(np.arange(n_dsms), src_config_fnames, dst_config_fnames):

            config_s2p = json2dict(src_config_fname)    
            s2p_out_dir = os.path.dirname(dst_config_fname)
            os.makedirs(s2p_out_dir, exist_ok=True)

            # set s2p config
            config_s2p['out_dir'] = s2p_out_dir 
            config_s2p['temporary_dir'] = os.path.join(s2p_out_dir, 'tmp')
            for i in [0,1]:
                img_basename = os.path.basename(config_s2p['images'][i]['img'])
                file_id = os.path.splitext(img_basename)[0]
                img_geotiff_path = glob.glob('{}/**/{}'.format(dir_src_images, img_basename), recursive=True)[0]
                config_s2p['images'][i]['img'] = img_geotiff_path
                if use_corrected_rpcs:
                    img_rpc_path = os.path.join(adj_rpc_dir, file_id + '_RPC_adj.txt')
                    config_s2p['images'][i]['rpc'] = get_dict_from_rpcm(rpcm.rpc_from_rpc_file(img_rpc_path))

            with open(dst_config_fname, 'w') as f:
                json.dump(config_s2p, f)

            log_file = os.path.join(os.path.dirname(dst_config_fname), 'log.txt')
            with open(log_file, 'w') as outfile:
                subprocess.run(['s2p', dst_config_fname], stdout=outfile)

            # gdalwarp
            s2p_warp_dir = s2p_out_dir.replace('s2p', 'warp')
            os.makedirs(s2p_warp_dir, exist_ok=True)
            src_fname, t_fname = os.path.join(s2p_out_dir, 'dsm.tif'), os.path.join(s2p_warp_dir, 'dsm.tif')
            if not os.path.exists(src_fname):
                err_indices.append(dsm_idx)
            tr = str(self.res)[1:-1].replace(',','')
            args = [self.epsg, self.xmin, self.ymin, self.xmax, self.ymax, tr, src_fname, t_fname]
            os.system('gdalwarp -s_srs {0} -t_srs {0} -te {1} {2} {3} {4} -tr {5} {6}  {7} -overwrite'.format(*args))                     
            print('\rComputed {} dsms / {} ({} err)'.format(dsm_idx+1, n_dsms, len(err_indices)),end='\r')
        
        log_file = os.path.join(self.dst_dir, '{}_s2p_crashes.txt'.format(self.name))
        with open(log_file, 'a') as outfile:
            for idx in err_indices:
                outfile.write('{}\n\n'.format(dst_config_fnames[idx]))
              
        # merge dsms
        print('\n\nMerging dsms...\n')
        import warnings
        warnings.filterwarnings('ignore')
        dsms_out_dir = '{}/{}/{}'.format(os.getcwd(), self.dst_dir, 'dsms')
        os.makedirs(dsms_out_dir, exist_ok=True)
        individual_dsm_fnames = glob.glob(self.dst_dir + '/warp/{}/**/*.tif'.format(t_id), recursive=True)
        final_dsm = np.nanmedian(np.dstack([np.array(Image.open(fname)) for fname in individual_dsm_fnames]), axis=2)
        final_dsm_fname = os.path.join(dsms_out_dir, '{}.tif'.format(t_id))
        Image.fromarray(final_dsm).save(final_dsm_fname)
        utils.copy_geotiff_metadata(individual_dsm_fnames[0], final_dsm_fname)

        # apply mask if available
        if os.path.exists(os.path.join(self.dst_dir, 'mask.tif')):
            masked_dsms_dir = os.path.join(dsms_out_dir, 'masked_dsms')
            os.makedirs(masked_dsms_dir, exist_ok=True)
            final_masked_dsm_fname = os.path.join(masked_dsms_dir, '{}.tif'.format(t_id))
            aoi_mask = np.array(Image.open(os.path.join(self.dst_dir, 'mask.tif')))
            final_dsm[~aoi_mask.astype(bool)] = np.nan
            Image.fromarray(final_dsm).save(final_masked_dsm_fname)
            utils.copy_geotiff_metadata(individual_dsm_fnames[0], final_masked_dsm_fname)

        # copy original dsm
        original_dsms_dir = os.path.join(dsms_out_dir, 'original_dsms')
        os.makedirs(original_dsms_dir, exist_ok=True)
        original_dsm_fname = '{}/{}_dsms/{}.tiff'.format(self.src_dir, self.name, t_id)
        os.system('cp {} {}'.format(original_dsm_fname, original_dsms_dir))

        print('Done!\n\n')
        
    
    def load_reconstructed_DSMs(self, timeline_indices, use_mask=False):
        dsm_timeseries = []
        for t_idx in timeline_indices:
            masked_dsm_fname = os.path.join(self.dst_dir, 'dsms/masked_dsms/{}.tif'.format(self.timeline[t_idx]['id']))
            full_dsm_fname = os.path.join(self.dst_dir, 'dsms/{}.tif'.format(self.timeline[t_idx]['id']))
            if os.path.exists(masked_dsm_fname) and use_mask:
                 dsm_timeseries.append(np.array(Image.open(masked_dsm_fname)))
            else:
                dsm_timeseries.append(np.array(Image.open(full_dsm_fname)))       
        return np.dstack(dsm_timeseries)
    
    
    def compute_3D_statistics_over_time(self, timeline_indices):
        
        print('\nComputing 4D statistics of the timeseries! Chosen dates:')
        for t_idx in timeline_indices:
            print('{}'.format(self.timeline[t_idx]['datetime']))
        
        output_dir = os.path.join(self.dst_dir, '4Dstats')
        os.makedirs(output_dir, exist_ok=True)
        
        # get timeseries
        dsm_timeseries_ndarray = self.load_reconstructed_DSMs(timeline_indices)    
        # extract mean
        mean_dsm = np.nanmean(dsm_timeseries_ndarray, axis=2)
        Image.fromarray(mean_dsm).save(output_dir + '/mean.tif')
        # extract std
        std_dsm = np.nanstd(dsm_timeseries_ndarray, axis=2)
        Image.fromarray(std_dsm).save(output_dir + '/std.tif')
        # save log of the dates employed to compute the statistics
        with open(os.path.join(output_dir, 'dates.txt'), 'w') as f:
            for t_idx in timeline_indices:
                f.write('{}\n'.format(self.timeline[t_idx]['datetime']))
        
        print('\nDone! Results were saved at {}'.format(output_dir))
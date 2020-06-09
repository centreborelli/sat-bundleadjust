import numpy as np
import matplotlib.pyplot as plt
import vistools
import json
import rpc_utils
import glob
import rpcm
import os
import utils
import rasterio
import pickle
from PIL import Image
import ba_utils
import subprocess
import srtm4

def custom_equalization(im, mask=None, clip=True, percentiles=5):
    ''' im is a numpy array
        returns a numpy array
    '''
    import numpy as np
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

def combine_aoi_borders(list_aois):
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
        crops.append({ 'crop': crop, 'x0': 0.0, 'y0': 0.0})
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

def get_perspective_cam_from_rpc(rpc, crop):
        # approximate current rpc as a perspective 3x4 matrix
    x, y, w, h = 0, 0, crop['crop'].shape[1], crop['crop'].shape[0]
    P_img = rpc_utils.approx_rpc_as_proj_matrix(rpc, [x,x+w,10], [y,y+h,10], \
                                                [rpc.alt_offset - 100, rpc.alt_offset + 100, 10])
    #express P in terms of crop coord by applying the translation x0, y0 (i.e.top-left corner of the crop)
    T_crop = np.array([[1., 0., -crop['x0']], [0., 1., -crop['y0']], [0., 0., 1.]])
    current_P = T_crop @ P_img
    return current_P/current_P[2,3]

def get_affine_cam_from_rpc(rpc, crop, lon, lat, alt):
    p_x, p_y, p_z = ba_utils.latlon_to_ecef_custom(lat, lon, alt)
    p_geocentric = [p_x, p_y, p_z]
    P_img = ba_utils.rpc_affine_approx_for_bundle_adjustment(rpc, p_geocentric)
    T_crop = np.array([[1., 0., -crop['x0']], [0., 1., -crop['y0']], [0., 0., 1.]])
    current_P = T_crop @ P_img
    return current_P/current_P[2,3]

def load_new_matrices(myrpcs_new, mycrops_new, aoi, cam_model='Perspective'):
    
    n_ims, myprojmats_new, err_indices = len(myrpcs_new), [], []

    if cam_model =='Perspective':
        for im_idx, rpc, crop in zip(np.arange(n_ims), myrpcs_new, mycrops_new):

            try:
                myprojmats_new.append(get_perspective_cam_from_rpc(rpc, crop))
            except:
                myprojmats_new.append(np.nan)
                err_indices.append(im_idx)
            print('\rLoading {} projection matrices / {} ({} err)'.format(im_idx+1,n_ims, len(err_indices)),end='\r') 

    else:
        lon, lat = aoi['center'][0], aoi['center'][1]
        alt = srtm4.srtm4(lon, lat)
        for im_idx, rpc, crop in zip(np.arange(n_ims), myrpcs_new, mycrops_new):
            
            try:
                myprojmats_new.append(get_affine_cam_from_rpc(rpc, crop, lon, lat, alt))
            except:
                myprojmats_new.append(np.nan)
                err_indices.append(im_idx)
            print('\rLoading {} projection matrices / {} ({} err)'.format(im_idx+1,n_ims, len(err_indices)),end='\r')
    
    if len(err_indices) > 0:
        err_msg = 'Max localization iterations (100) exceeded'
        err_str1 = 'Error in localization_iterative from rpcm/rpc_model.py: {} !!!'.format(err_msg)
        err_str2 = 'Failed at {} cameras'.format(len(err_indices))
        print('{}\n{}\n'.format(err_str1, err_str2))
    else:
        print('\nDone!\n')
    return myprojmats_new

def load_image_footprints(crops, rpcs):
    n_ims, footprints, err_indices = len(crops), [], []
    for crop, rpc, im_idx in zip(crops, rpcs, np.arange(n_ims)):
        
        try:
            footprints.append(ba_utils.get_image_footprints([rpc], [crop['crop']])[0])
        except:
            footprints.append(np.nan)
            err_indices.append(im_idx)
            
        print('\rLoading {} image footprints / {} ({} err)'.format(im_idx+1, n_ims, len(err_indices)), end='\r')
    if len(err_indices) > 0:
        err_msg = 'Max localization iterations (100) exceeded'
        err_str1 = 'Error in localization_iterative from rpcm/rpc_model.py: {} !!!'.format(err_msg)
        err_str2 = 'Failed at {} cameras'.format(len(err_indices))
        print('{}\n{}\n'.format(err_str1, err_str2))
    else:
        print('\nDone!\n')
    return footprints









class Scene:
    def __init__(self, input_dir, output_dir, name):
        

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
        self.compute_aoi_masks = True
        self.use_masks_for_equalization = True
        self.n_adj = 0
        self.myimages_adj = []
        self.mycrops_adj = [] 
        self.myprojmats_adj = [] 
        self.myrpcs_adj = [] 
        self.myfootprints_adj = []
        self.n_new = 0
        self.myimages_new = []
        self.mycrops_new = []
        self.myprojmats_new = [] 
        self.myrpcs_new = [] 
        self.myfootprints_new = []
        self.projmats_model='Perspective'
        
        #to be set by  set_ba_input_data
        self.ba_input_data = {}
        
        self.load_scene()
        

        
        
    def load_scene(self):   
        
        print('#############################################################')
        print('Loading scene {}...\n'.format(self.name))

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

        n_images_all = len(self.all_images_fnames)
        print('Found {} skysat images\n'.format(n_images_all))

        self.timeline = self.group_files_by_date(self.all_images_fnames, self.all_images_datetimes)
        print('Found {} different dates in the scene timeline\n'.format(len(self.timeline)))
        
        self.aoi_lonlat = combine_aoi_borders(config_aois)
        
        print('Successfully loaded scene {}'.format(self.name))
        print('#############################################################\n\n')
    
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
                             'image_indices':d[k], 'n_images': len(d[k])})
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
        display_rois_over_map([self.aoi_lonlat], zoom_factor=14)
    
    def display_crops(self):
        import vistools
        mycrops = self.mycrops_adj + self.mycrops_new
        vistools.display_gallery([f['crop'] for f in mycrops])
    
    def display_image_masks(self):
        import vistools
        if not self.compute_aoi_masks:
            print('aoi masks have not been computed for this scene')
        else:
            mycrops = self.mycrops_adj + self.mycrops_new
            vistools.display_gallery([255*f['mask'] for f in mycrops])
        
    
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
            print('The mask of this scene is not defined yet')
        else:
            fig = plt.figure(figsize=(10,10))
            plt.imshow(self.mask);
            
            
         
    def load_previously_bundle_adjusted_data(self, use_masks_to_equalize=False):
        
        dir_src_images = '{}/{}_images'.format(self.src_dir, self.name)
        dir_adj_rpc = os.path.join(self.dst_dir, 'RPC_adj')
        dir_adj_P = os.path.join(self.dst_dir, 'P_adj')
        if os.path.exists(self.dst_dir + '/myimages.pickle') and os.path.isdir(dir_adj_rpc) and os.path.isdir(dir_adj_P):

            # read tiff images 
            for img_basename in pickle.load(open(self.dst_dir+'/myimages.pickle','rb')):
                img_geotiff_path = glob.glob('{}/**/{}'.format(dir_src_images, img_basename), recursive=True)[0]
                self.myimages_adj.append(img_geotiff_path)
            self.n_adj += len(self.myimages_adj)
            print('Found {} previously adjusted images in scene {}\n'.format(self.n_adj, self.name))

            # load previously adjusted rpcs
            self.myrpcs_adj = load_adjusted_rpcs(self.myimages_adj, dir_adj_rpc)
            
            # get image crops
            self.mycrops_adj = load_image_crops(self.myimages_adj, \
                                                get_aoi_mask = self.compute_aoi_masks, \
                                                rpcs = self.myrpcs_adj, \
                                                aoi = self.aoi_lonlat, \
                                                use_mask_for_equalization = use_masks_to_equalize)

            # load previously adjusted projection matrices
            self.myprojmats_adj = load_adjusted_matrices(self.myimages_adj, dir_adj_P)

            # load image footprints
            self.myfootprints_adj = load_image_footprints(self.mycrops_adj, self.myrpcs_adj)
        else:
            print('No previously adjusted data was found in scene {}\n'.format(self.name))

     
    def load_new_dates_to_adjust(self, timeline_indices, cam_model=None, use_masks_to_equalize=False):
        
        if cam_model is not None:
            self.projmats_model = cam_model
        
        t_indices_files, t_ids = [], []
        for t_idx in timeline_indices:
            t_indices_files.extend(self.timeline[t_idx]['image_indices'])
            t_ids.append(self.timeline[t_idx]['id'])
        print('Chosen dates of the timeline to bundle adjust:')
        for t_idx in timeline_indices:
            print('{}'.format(self.timeline[t_idx]['datetime']))
        print('\n')
        im_fname = np.array(self.all_images_fnames)[t_indices_files].tolist()
        im_rpc = np.array(self.all_images_rpcs)[t_indices_files].tolist()
        print(len(im_fname), 'new images to be bundle adjusted ! \n')
        
        self.myimages_new = im_fname
        self.n_new = len(self.myimages_new)
        
        # get new rpcs
        self.myrpcs_new = im_rpc
        print('Loading {} image rpcs / {}'.format(len(self.myrpcs_new), len(self.myrpcs_new)))
        print('Done!\n')

        # get new image crops
        self.mycrops_new = load_image_crops(self.myimages_new, \
                                            get_aoi_mask = self.compute_aoi_masks, \
                                            rpcs = self.myrpcs_new, \
                                            aoi = self.aoi_lonlat, \
                                            use_mask_for_equalization = use_masks_to_equalize)
        
        # get new projection matrices
        self.myprojmats_new = load_new_matrices(self.myrpcs_new, self.mycrops_new, \
                                                self.aoi_lonlat, cam_model=self.projmats_model)

        # get new footprints
        self.myfootprints_new = load_image_footprints(self.mycrops_new, self.myrpcs_new)
        
        
    
    def set_ba_input_data(self):
        self.ba_input_data['input_dir'] = self.dst_dir
        self.ba_input_data['n_new'] = self.n_new
        self.ba_input_data['n_adj'] = self.n_adj
        self.ba_input_data['myimages'] = self.myimages_adj + self.myimages_new
        self.ba_input_data['input_crops'] = self.mycrops_adj + self.mycrops_new
        self.ba_input_data['input_rpcs'] = self.myrpcs_adj + self.myrpcs_new
        self.ba_input_data['input_P'] = self.myprojmats_adj + self.myprojmats_new
        self.ba_input_data['input_footprints'] = self.myfootprints_adj + self.myfootprints_new
        self.ba_input_data['cam_model'] = self.projmats_model
        
        if self.compute_aoi_masks:
            self.ba_input_data['input_masks'] = [f['mask'] for f in self.mycrops_adj] + [f['mask'] for f in self.mycrops_new]
        else:
            self.ba_input_data['input_masks'] = None
        
        print('\nBundle Adjustment input data is ready !\n')
        
    
    def extract_ba_data_indices_according_to_date(self, t_idx):
        
        # returns the indices of the files in ba_input_data['myimages'] that belong to a common timeline index
        all_img_indices_t_id = self.timeline[t_idx]['image_indices']
        img_fnames_t_id = np.array(self.all_images_fnames)[all_img_indices_t_id].tolist()
        ba_data_indices_t_id = [self.ba_input_data['myimages'].index(fname) for fname in img_fnames_t_id]
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
    
    def timeline_instances_diagram(self, timeline_indices, ba_pipeline, neighbors=1):
        
        def get_datetime_diff_in_days_hours(dt_i, dt_j):
            import time
            di_ts = time.mktime(dt_i.timetuple())
            dj_ts = time.mktime(dt_j.timetuple())
            diff_di_dj_in_mins = int(abs(di_ts - dj_ts) / 60)
            days = diff_di_dj_in_mins / 1440     
            leftover_minutes = diff_di_dj_in_mins % 1440
            hours = leftover_minutes / 60
            return days, hours
        
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
                days, hours = get_datetime_diff_in_days_hours(nodes_dates[i], nodes_dates[j])
                
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
        
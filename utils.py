"""
* Geotiff Read and Write
* extract metadata: datetime, RPC, 
* miscelaneous function for crop
* wrappers for gdaltransform and gdalwarp

Copyright (C) 2018, Gabriele Facciolo <facciolo@cmla.ens-cachan.fr>
Copyright (C) 2018, Carlo de Franchis <carlo.de-franchis@ens-cachan.fr>
"""


import os
import datetime
import requests
import subprocess
import numpy as np

import bs4

import rpc_model



### READ AND WRITE IMAGES WITH GDAL via rasterio
    

import rasterio
import warnings

def rio_open(*args,**kwargs):
    """
    Open an image with rasterio.

    Args:
        p: path to the image file

    Returns:
        rasterio dataset
    """
    with warnings.catch_warnings():  # noisy warning may occur here
        warnings.filterwarnings("ignore",
                                category=UserWarning)
        return rasterio.open(*args,**kwargs)    
    

    
def readGTIFF(fname):
    '''
    Reads an image file into a numpy array, 
    returns the numpy array with dimensios (height, width, channels) 
    The returned numpy array is always of type numpy.float 
    '''
    import rasterio
    import numpy as np
    # read the image into a np.array 
    with  rio_open(fname) as s:   # rio_open suppress verbose warnings
        # print('reading image of size: %s'%str(im.shape))
        im = s.read()
    return im.transpose([1,2,0]).astype(np.float)

  
def readGTIFFmeta(fname):
    '''
    Reads the image GeoTIFF metadata using rasterio and returns it,
    along with the bounding box, in a tuple: (meta, bounds)
    if the file format doesn't support metadata the returned metadata is invalid
    This is the metadata rasterio was capable to interpret, 
    but the ultimate command for reading metadata is *gdalinfo* 
    '''
    import rasterio
    import numpy as np
    # read the image into a np.array 
    with  rio_open(fname) as s:
        ## interesting information
        # print(s.crs,s.meta,s.bounds)
        return (s.meta,s.bounds)


def get_driver_from_extension(filename):
    import os.path
    ext = os.path.splitext(filename)[1].upper()
    if ext in ('.TIF', '.TIFF'):
        return 'GTiff'
    elif ext in ('.JPG', '.JPEG'):
        return 'JPEG'
    elif ext == '.PNG':
        return 'PNG'
    return None


def writeGTIFF(im, fname, copy_metadata_from=None):
    '''
    Writes a numpy array to a GeoTIFF, PNG, or JPEG image depending on fname extension.
    For GeoTIFF files the metadata can be copied from another file.
    Note that if  im  and  copy_metadata_from have different size, 
    the copied geolocation properties are not adapted. 
    '''
    import rasterio
    import numpy as np

    # set default metadata profile
    p = {'width': 0, 'height': 0, 'count': 1, 'dtype': 'uint8', 'driver': 'PNG',
             'affine': rasterio.Affine (0,1,0,0,1,0),
             'crs': rasterio.crs.CRS({'init': 'epsg:32610'}), 
             'tiled': False,  'nodata': None}
    
    # read and update input metadata if available 
    if copy_metadata_from:   
        x = rio_open(copy_metadata_from)
        p.update( x.profile )    
 
    # format input 
    if  len(im.shape) == 2: 
        im = im[:,:,np.newaxis]
        
    # override driver and shape
    indriver = get_driver_from_extension(fname)
    if indriver and (indriver != p['driver']):
        #print('writeGTIFF: driver override from %s to %s'%( p['driver'], indriver))
        p['driver'] = indriver or p['driver']
        p['dtype'] = 'float32'
    
    #if indriver == 'GTiff' and (p['height'] != im.shape[0]  or  p['width'] != im.shape[1]):
    #    # this is a problem only for GTiff
    #    print('writeGTIFF: changing the size of the GeoTIFF')
    #else:     
    #    # remove useless properties 
    #    p.pop('tiled')
    
    p['height'] = im.shape[0]
    p['width']  = im.shape[1]
    p['count']  = im.shape[2]

    with rio_open(fname, 'w', **p) as d:
            d.write((im.transpose([2,0,1]).astype(d.profile['dtype'])))

                



#### wrappers for gdaltransform and gdalwarp


def is_absolute(url):
    return bool(requests.utils.urlparse(url).netloc)


def listFD(url, ext=''):
    page = requests.get(url).text
    soup = bs4.BeautifulSoup(page, 'html.parser')
    files = [node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]

    parsed_url = requests.utils.urlparse(url)
    base = '%s://%s' % (parsed_url.scheme, parsed_url.netloc)
    return [x if is_absolute(x) else base+x for x in files]


def acquisition_date(geotiff_path):
    """
    """
    with rio_open(geotiff_path) as src:
        date_string = src.tags()['NITF_STDIDC_ACQUISITION_DATE']
        return datetime.datetime.strptime(date_string, "%Y%m%d%H%M%S")

    
    
def gdal_get_longlat_of_pixel(fname, x, y, verbose=True):
    '''
    returns the longitude latitude and altitude (wrt the WGS84 reference 
    ellipsoid) for the points at pixel coordinates (x, y) of the image fname. 
    The CRS of the input GeoTIFF is determined from the metadata in the file.

    '''
    import subprocess

    # add vsicurl prefix if needed
    env = os.environ.copy()
    if fname.startswith(('http://', 'https://')):
        env['CPL_VSIL_CURL_ALLOWED_EXTENSIONS'] = fname[-3:]
        fname = '/vsicurl/{}'.format(fname)

    # form the query string for gdaltransform
    q = b''
    for (xi,yi) in zip(x,y):
        q = q + b'%d %d\n'%(xi,yi)
    # call gdaltransform, "+proj=longlat" uses the WGS84 ellipsoid 
    #    echo '0 0' | gdaltransform -t_srs "+proj=longlat" inputimage.tif 
    cmdlist = ['gdaltransform', '-t_srs', "+proj=longlat", fname]
    if verbose:
        print ('RUN: ' +  ' '.join(cmdlist) + ' [x y from stdin]')
    p = subprocess.Popen(cmdlist,
                         stdin=subprocess.PIPE,stdout=subprocess.PIPE)
    out = (p.communicate(q)[0]).decode()
    listeout =  [ list(map(float, x.split())) for x in out.splitlines()]
    return listeout


    
    
def get_image_longlat_polygon(fname):
    """
    Return a GeoJSON polygon with the four corners of a GeoTIFF image.
    """
    # get the image size
    A = readGTIFFmeta(fname)
    H = int(A[0]['height'])
    W = int(A[0]['width'])

    cols, rows = np.meshgrid([0,W], [0,H])
    cols = [0,W,W,0]
    rows = [0,0,H,H]

    coords = gdal_get_longlat_of_pixel(fname, cols, rows, verbose=False)

    # remove the altitude (which is 0)
    coords = [[x[0], x[1]] for x in coords]

    return {'type': 'Polygon', 'coordinates': [coords]}



       
def gdal_resample_image_to_longlat(fname, outfname, verbose=True):
    '''
    resample a geotiff image file in longlat coordinates (EPSG: 4326 with WGS84 datum)
    and saves the result in outfname
    '''
    import os
    
    driver = get_driver_from_extension(outfname)
    cmd = 'gdalwarp -overwrite  -of %s -t_srs "+proj=longlat +datum=WGS84" %s %s'%(driver, fname, outfname)
    if verbose:
        print('RUN: ' + cmd)
    return os.system(cmd)
    
    

def rpc_from_geotiff(geotiff_path, outrpcfile='.rpc'):
    """
    Reads the RPC from a geotiff file
    returns the RPC in a rpc_model object
    """
    env = os.environ.copy()
    if geotiff_path.startswith(('http://', 'https://')):
        env['CPL_VSIL_CURL_ALLOWED_EXTENSIONS'] = geotiff_path[-3:]
        path = '/vsicurl/{}'.format(geotiff_path)
    else:
        path = geotiff_path

    f = open(outrpcfile, 'wb')
    x = subprocess.Popen(["gdalinfo", path], stdout=subprocess.PIPE).communicate()[0]
    x = x.splitlines()
    for l in x:

        if(1):
            if (b'SAMP_' not in l) and (b'LINE_' not in l) and (b'HEIGHT_' not in l) and (b'LAT_' not in l) and (b'LONG_' not in l) and (b'MAX_' not in l) and (b'MIN_' not in l):
                  continue
            y = l.strip().replace(b'=',b': ')
            if b'COEFF' in y:
                  z = y.split(b' ')
                  t=1
                  for j in z[1:]:
                          f.write(b'%s_%d: %s\n'%(z[0][:-1],t,j))
                          t+=1
            else:
                  f.write((y+b'\n'))
    f.close()
    return rpc_model.RPCModel(outrpcfile)


def bounding_box2D(pts):
    """
    bounding box for the points pts
    """
    dim = len(pts[0])  # should be 2
    bb_min = [min([t[i] for t in pts]) for i in range(dim)]
    bb_max = [max([t[i] for t in pts]) for i in range(dim)]
    return bb_min[0], bb_min[1], bb_max[0] - bb_min[0], bb_max[1] - bb_min[1]


def image_crop_gdal(inpath, x, y, w, h, outpath):
    """
    Image crop defined in pixel coordinates using gdal_translate.

    Args:
        inpath: path to an image file
        x, y, w, h: four integers defining the rectangular crop pixel coordinates.
            (x, y) is the top-left corner, and (w, h) are the dimensions of the
            rectangle.
        outpath: path to the output crop
    """
    if int(x) != x or int(y) != y:
        print('WARNING: image_crop_gdal will round the coordinates of your crop')

    env = os.environ.copy()
    if inpath.startswith(('http://', 'https://')):
        env['CPL_VSIL_CURL_ALLOWED_EXTENSIONS'] = inpath[-3:]
        path = '/vsicurl/{}'.format(inpath)
    else:
        path = inpath

    cmd = ['gdal_translate', path, outpath,
           '-srcwin', str(x), str(y), str(w), str(h),
           '-ot', 'Float32',
           '-co', 'TILED=YES',
           '-co', 'BIGTIFF=IF_NEEDED']

    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT, env=env)
    except subprocess.CalledProcessError as e:
        if inpath.startswith(('http://', 'https://')):
            if not requests.head(inpath).ok:
                print('{} is not available'.format(inpath))
                return
        print('ERROR: this command failed')
        print(' '.join(cmd))
        print(e.output)


def points_apply_homography(H, pts):
    """
    Applies an homography to a list of 2D points.

    Args:
        H: numpy array containing the 3x3 homography matrix
        pts: numpy array containing the list of 2D points, one per line

    Returns:
        a numpy array containing the list of transformed points, one per line
    """
    pts = np.asarray(pts)

    # convert the input points to homogeneous coordinates
    if len(pts[0]) < 2:
        print("""points_apply_homography: ERROR the input must be a numpy array
          of 2D points, one point per line""")
        return
    pts = np.hstack((pts[:, 0:2], pts[:, 0:1]*0+1))

    # apply the transformation
    Hpts = (np.dot(H, pts.T)).T

    # normalize the homogeneous result and trim the extra dimension
    Hpts = Hpts * (1.0 / np.tile( Hpts[:, 2], (3, 1)) ).T
    return Hpts[:, 0:2]


def bounding_box_of_projected_aoi(rpc, aoi, z=0, homography=None):
    """
    Return the x, y, w, h pixel bounding box of a projected AOI.

    Args:
        rpc (rpc_model.RPCModel): RPC camera model
        aoi (geojson.Polygon): GeoJSON polygon representing the AOI
        z (float): altitude of the AOI with respect to the WGS84 ellipsoid
        homography (2D array, optional): matrix of shape (3, 3) representing an
            homography to be applied to the projected points before computing
            their bounding box.

    Return:
        x, y (ints): pixel coordinates of the top-left corner of the bounding box
        w, h (ints): pixel dimensions of the bounding box
    """
    lons, lats = np.array(aoi['coordinates'][0]).T
    x, y = rpc.projection(lons, lats, z)
    pts = list(zip(x, y))
    if homography is not None:
        pts = points_apply_homography(homography, pts)
    return np.round(bounding_box2D(pts)).astype(int)


def crop_aoi(geotiff, aoi, z=0):
    """
    Crop a geographic AOI in a georeferenced image using its RPC functions.

    Args:
        geotiff (string): path or url to the input GeoTIFF image file
        aoi (geojson.Polygon): GeoJSON polygon representing the AOI
        z (float, optional): base altitude with respect to WGS84 ellipsoid (0
            by default)

    Return:
        crop (array): numpy array containing the cropped image
        x, y, w, h (ints): image coordinates of the crop. x, y are the
            coordinates of the top-left corner, while w, h are the dimensions
            of the crop.
    """
    x, y, w, h = bounding_box_of_projected_aoi(rpc_from_geotiff(geotiff), aoi, z)
    with rio_open(geotiff, 'r') as src:
        crop = src.read(window=((y, y + h), (x, x + w))).squeeze()
    return crop, x, y

# fast function to convert lonlat to utm
def utm_from_lonlat(lons, lats):
    return utm_from_latlon(lats, lons)

# fast function to convert latlon to utm
def utm_from_latlon(lats, lons):
    import utm
    import pyproj
    n = utm.latlon_to_zone_number(lats[0], lons[0])
    l = utm.latitude_to_zone_letter(lats[0])
    proj_src = pyproj.Proj('+proj=latlong')
    proj_dst = pyproj.Proj('+proj=utm +zone={}{}'.format(n, l))
    return pyproj.transform(proj_src, proj_dst, lons, lats)

def utm_from_lonlatzs(lon, lat, zonestring):
    import utm
    import pyproj
    proj_src = pyproj.Proj('+proj=latlong')
    proj_dst = pyproj.Proj('+proj=utm +zone=%s' % zonestring)
    return pyproj.transform(proj_src, proj_dst, lat, lon)

def zonestring_from_lonlat(lon, lat):
    import utm
    n = utm.latlon_to_zone_number(lat, lon)
    l = utm.latitude_to_zone_letter(lat)
    s = "%d%s" % (n, l)
    return s


# fast function to convert  to utm
def lonlat_from_utm(easts, norths, zonestring):
    import utm
    import pyproj
    #n = utm.latlon_to_zone_number(lats[0], lons[0])
    #l = utm.latitude_to_zone_letter(lats[0])
    proj_src = pyproj.Proj("+proj=utm +zone=%s" % zonestring)
    proj_dst = pyproj.Proj('+proj=latlong')
    return pyproj.transform(proj_src, proj_dst, easts, norths)




def utm_bounding_box_from_lonlat_aoi(aoi):
    """
    Computes the UTM bounding box (min_easting, min_northing, max_easting,
    max_northing)  of a projected AOI.

    Args:
        aoi (geojson.Polygon): GeoJSON polygon representing the AOI expressed in (long, lat)

    Return:
        min_easting, min_northing, max_easting, max_northing: the coordinates
        of the top-left corner and lower-right corners of the aoi in UTM coords
    """
    lons, lats  = np.array(aoi['coordinates'][0]).T
    east, north = utm_from_lonlat(lons, lats)
    pts = list(zip(east, north))
    emin, nmin, deltae, deltan = bounding_box2D(pts)
    return emin, emin+deltae, nmin, nmin+deltan




def simple_equalization_8bit(im, percentiles=5):
    ''' im is a numpy array
        returns a numpy array
    '''
    import numpy as np
    mi, ma = np.percentile(im[np.isfinite(im)], (percentiles,100-percentiles))
    im = np.minimum(np.maximum(im,mi), ma) # clip
    im = (im-mi)/(ma-mi)*255.0   # scale
    im=im.astype(np.uint8)
    return im

import numpy as np

import srtm4
import geojson

from IS18 import vistools
from IS18 import utils


def lonlat_geojson_from_geotiff_crop(rpc, crop_offset):
    
    z = srtm4.srtm4(rpc.lon_offset, rpc.lat_offset)
    col0, row0, w, h = crop_offset['col0'], crop_offset['row0'], crop_offset['width'], crop_offset['height']
    lons, lats = rpc.localization([col0, col0, col0 + w, col0 + w, col0],
                                  [row0, row0 + h, row0 + h, row0, row0],
                                  [z, z, z, z, z])
    lonlat_geojson = geojson.Feature(geometry=geojson.Polygon([list(zip(lons, lats))]))['geometry']
    lonlat_geojson['center'] = np.mean(lonlat_geojson['coordinates'][0][:4], axis=0).tolist()

    return lonlat_geojson

    
def measure_squared_km_from_lonlat_geojson(lonlat_geojson):
    
    from shapely.geometry import shape
    
    lons = np.array(lonlat_geojson['coordinates'][0])[:,0]
    lats = np.array(lonlat_geojson['coordinates'][0])[:,1]
    easts, norths = utils.utm_from_lonlat(lons, lats)
    aoi_utm = shape({'type': 'Polygon', 'coordinates': [(np.vstack((easts, norths)).T).tolist()]})
    
    area_squared_m = aoi_utm.area
    area_squared_km = area_squared_m * (1e-6/1.)
    
    return area_squared_km


def lonlat_geojson_from_utm_geojson(utm_geojson, utm_zone):
    
    easts = np.array(utm_geojson["coordinates"][0])[:,0]
    norths = np.array(utm_geojson["coordinates"][0])[:,1]  
    lons, lats = utils.lonlat_from_utm(easts, norths, utm_zone)
    lonlat_coords = np.vstack((lons, lats)).T
    lonlat_geojson = {'coordinates': [lonlat_coords.tolist()], 'type': 'Polygon'}
    lonlat_geojson['center'] = np.mean(lonlat_geojson['coordinates'][0][:4], axis=0).tolist()
    
    return lonlat_geojson


def utm_geojson_from_lonlat_geojson(lonlat_geojson):

    lons = np.array(lonlat_geojson["coordinates"][0])[:,0]
    lats = np.array(lonlat_geojson["coordinates"][0])[:,1]
    easts, norths = utils.utm_from_lonlat(lons, lats)
    utm_coords = np.vstack((easts, norths)).T
    utm_geojson = {'coordinates': [utm_coords.tolist()], 'type': 'Polygon'}
    utm_geojson['center'] = np.mean(utm_geojson['coordinates'][0][:4], axis=0).tolist()
    
    return utm_geojson
      
    
def utm_zonestring_from_lonlat_geojson(lonlat_geojson):
    
    c_lon = lonlat_geojson['center'][0]
    c_lat = lonlat_geojson['center'][1]
    utm_zone = utils.zonestring_from_lonlat(c_lon, c_lat)
    
    return utm_zone
    

def get_aoi_where_at_least_two_lonlat_geojson_overlap(lonlat_geojson_list):
    
    # computes the union of all pair intersections
    
    from shapely.ops import cascaded_union
    from shapely.geometry import shape
    from itertools import combinations
    
    utm_zone = utm_zonestring_from_lonlat_geojson(lonlat_geojson_list[0])
    utm_geojson_list = [utm_geojson_from_lonlat_geojson(x) for x in lonlat_geojson_list]  
    
    geoms = [shape(g) for g in utm_geojson_list]
    geoms = [a.intersection(b) for a, b in combinations(geoms, 2)]
    combined_borders_shapely = cascaded_union([ geom if geom.is_valid else geom.buffer(0) for geom in geoms])
    vertices = (np.array(combined_borders_shapely.boundary.coords.xy).T)[:-1,:]
    utm_geojson = {'coordinates': [vertices.tolist()], 'type': 'Polygon'}
    utm_geojson['center'] = np.mean(utm_geojson['coordinates'][0][:4], axis=0).tolist()
    
    return lonlat_geojson_from_utm_geojson(utm_geojson, utm_zone)


def combine_lonlat_geojson_borders(lonlat_geojson_list):
    
    # computes the union of all geojson borders
    
    from shapely.ops import cascaded_union
    from shapely.geometry import shape
    
    utm_zone = utm_zonestring_from_lonlat_geojson(lonlat_geojson_list[0])
    utm_geojson_list = [utm_geojson_from_lonlat_geojson(x) for x in lonlat_geojson_list]  
    
    geoms = [shape(g) for g in utm_geojson_list] # convert aois to shapely polygons
    combined_borders_shapely = cascaded_union([ geom if geom.is_valid else geom.buffer(0) for geom in geoms])
    vertices = (np.array(combined_borders_shapely.boundary.coords.xy).T)[:-1,:]
    utm_geojson = {'coordinates': [vertices.tolist()], 'type': 'Polygon'}
    utm_geojson['center'] = np.mean(utm_geojson['coordinates'][0][:4], axis=0).tolist()
    
    return lonlat_geojson_from_utm_geojson(utm_geojson, utm_zone)


def display_lonlat_geojson_list_over_map(lonlat_geojson_list, zoom_factor=14):
    mymap = vistools.clickablemap(zoom=zoom_factor)
    for aoi in lonlat_geojson_list:   
        mymap.add_GeoJSON(aoi) 
    mymap.center = lonlat_geojson_list[int(len(lonlat_geojson_list)/2)]['center'][::-1]
    display(mymap)


def reestimate_lonlat_geojson_after_rpc_correction(initial_rpc, corrected_rpc, lonlat_geojson):

    aoi_lons_init = np.array(lonlat_geojson['coordinates'][0])[:,0]
    aoi_lats_init = np.array(lonlat_geojson['coordinates'][0])[:,1]
    alt = srtm4.srtm4(np.mean(aoi_lons_init), np.mean(aoi_lats_init))
    aoi_cols_init, aoi_rows_init = initial_rpc.projection(aoi_lons_init, aoi_lats_init,
                                                          [alt]*aoi_lons_init.shape[0])
    aoi_lons_ba, aoi_lats_ba = corrected_rpc.localization(aoi_cols_init, aoi_rows_init,
                                                          [alt]*aoi_lons_init.shape[0])
    lonlat_coords = np.vstack((aoi_lons_ba, aoi_lats_ba)).T
    lonlat_geojson = {'coordinates': [lonlat_coords.tolist()], 'type': 'Polygon'}
    lonlat_geojson['center'] = np.mean(lonlat_geojson['coordinates'][0][:4], axis=0).tolist()

    return lonlat_geojson


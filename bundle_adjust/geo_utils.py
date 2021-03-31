"""
A Generic Bundle Adjustment Methodology for Indirect RPC Model Refinement of Satellite Imagery
code for Image Processing On Line https://www.ipol.im/

author: Roger Mari <roger.mari@ens-paris-saclay.fr>
year: 2021

This script consists of a series of functions dedicated to deal with different geographic coordinate systems
and the GeoJSON format, which is used to delimit geographic areas
"""

import numpy as np


def utm_from_lonlat(lons, lats):
    """
    convert lon-lat to utm
    """
    return utm_from_latlon(lats, lons)


def utm_from_latlon(lats, lons):
    """
    convert lat-lon to utm
    """
    import pyproj
    import utm

    n = utm.latlon_to_zone_number(lats[0], lons[0])
    l = utm.latitude_to_zone_letter(lats[0])
    proj_src = pyproj.Proj("+proj=latlong")
    proj_dst = pyproj.Proj("+proj=utm +zone={}{}".format(n, l))
    return pyproj.transform(proj_src, proj_dst, lons, lats)


def zonestring_from_lonlat(lon, lat):
    """
    return utm zone string from lon-lat point
    """
    import utm

    n = utm.latlon_to_zone_number(lat, lon)
    l = utm.latitude_to_zone_letter(lat)
    s = "%d%s" % (n, l)
    return s


def lonlat_from_utm(easts, norths, zonestring):
    """
    convert utm to lon-lat
    """
    import pyproj

    proj_src = pyproj.Proj("+proj=utm +zone=%s" % zonestring)
    proj_dst = pyproj.Proj("+proj=latlong")
    return pyproj.transform(proj_src, proj_dst, easts, norths)


def utm_bbox_from_aoi_lonlat(lonlat_geojson):
    """
    compute the utm bounding box where a certain lon-lat geojson is inscribed
    """
    lons, lats = np.array(lonlat_geojson["coordinates"][0]).T
    easts, norths = utm_from_latlon(lats, lons)
    norths[norths < 0] = norths[norths < 0] + 10000000
    utm_bbx = {"xmin": easts.min(), "xmax": easts.max(), "ymin": norths.min(), "ymax": norths.max()}
    return utm_bbx


def lonlat_geojson_from_geotiff_crop(rpc, crop_offset, z=None):
    """
    compute the lonlat_geojson given a rpc and some crop bounding box coordinates
    """
    if z is None:
        import srtm4

        z = srtm4.srtm4(rpc.lon_offset, rpc.lat_offset)
    col0, row0, w, h = crop_offset["col0"], crop_offset["row0"], crop_offset["width"], crop_offset["height"]
    cols = [col0, col0, col0 + w, col0 + w, col0]
    rows = [row0, row0 + h, row0 + h, row0, row0]
    alts = [z, z, z, z, z]
    lons, lats = rpc.localization(cols, rows, alts)
    lonlat_coords = np.vstack((lons, lats)).T
    return geojson_polygon(lonlat_coords)


def measure_squared_km_from_lonlat_geojson(lonlat_geojson):
    """
    measure the area in squared km covered by a lonlat_geojson
    """
    from shapely.geometry import shape

    lons, lats = np.array(lonlat_geojson["coordinates"][0]).T
    easts, norths = utm_from_lonlat(lons, lats)
    utm_coords = np.vstack((easts, norths)).T
    area_squared_m = shape(geojson_polygon(utm_coords)).area
    area_squared_km = area_squared_m * (1e-6 / 1.0)
    return area_squared_km


def geojson_polygon(coords_array):
    """
    define a geojson polygon from a Nx2 numpy array with N 2d coordinates delimiting a boundary
    """
    geojson_dict = {"coordinates": [coords_array.tolist()], "type": "Polygon"}
    x, y = np.array(geojson_dict["coordinates"][0]).T
    x_c = x.min() + ((x.max() - x.min()) / 2)
    y_c = y.min() + ((y.max() - y.min()) / 2)
    geojson_dict["center"] = [x_c, y_c]
    return geojson_dict


def lonlat_geojson_from_utm_geojson(utm_geojson, utm_zone):
    """
    to convert a utm_geojson to a lonlat_geojson
    """
    easts, norths = np.array(utm_geojson["coordinates"][0]).T
    lons, lats = lonlat_from_utm(easts, norths, utm_zone)
    lonlat_coords = np.vstack((lons, lats)).T
    return geojson_polygon(lonlat_coords)


def utm_geojson_from_lonlat_geojson(lonlat_geojson):
    """
    to convert a lonlat_geojson to a utm_geojson
    """
    lons, lats = np.array(lonlat_geojson["coordinates"][0]).T
    easts, norths = utm_from_lonlat(lons, lats)
    utm_coords = np.vstack((easts, norths)).T
    return geojson_polygon(utm_coords)


def utm_zonestring_from_lonlat_geojson(lonlat_geojson):
    """
    get the utm zone string of a lonlat_geojson
    """
    return zonestring_from_lonlat(*lonlat_geojson["center"])


def combine_utm_geojson_borders(utm_geojson_list):
    """
    compute the union of a list of utm_geojson
    """
    from shapely.geometry import shape
    from shapely.ops import cascaded_union

    geoms = [shape(g) for g in utm_geojson_list]  # convert aois to shapely polygons
    combined_borders_shapely = cascaded_union([geom if geom.is_valid else geom.buffer(0) for geom in geoms])
    vertices = np.array(combined_borders_shapely.exterior.xy).T[:-1, :]
    return geojson_polygon(vertices)


def combine_lonlat_geojson_borders(lonlat_geojson_list):
    """
    compute the union of a list of lonlat_geojson
    """
    utm_zone = utm_zonestring_from_lonlat_geojson(lonlat_geojson_list[0])
    utm_geojson_list = [utm_geojson_from_lonlat_geojson(x) for x in lonlat_geojson_list]
    utm_geojson = combine_utm_geojson_borders(utm_geojson_list)
    return lonlat_geojson_from_utm_geojson(utm_geojson, utm_zone)


def latlon_to_ecef_custom(lat, lon, alt):
    """
    convert from geodetic (lat, lon, alt) to geocentric coordinates (x, y, z)
    """
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
    """
    convert from geocentric coordinates (x, y, z) to geodetic (lat, lon, alt)
    """
    a = 6378137.0
    e = 8.1819190842622e-2
    asq = a ** 2
    esq = e ** 2
    b = np.sqrt(asq * (1 - esq))
    bsq = b ** 2
    ep = np.sqrt((asq - bsq) / bsq)
    p = np.sqrt((x ** 2) + (y ** 2))
    th = np.arctan2(a * z, b * p)
    lon = np.arctan2(y, x)
    lat = np.arctan2((z + (ep ** 2) * b * (np.sin(th) ** 3)), (p - esq * a * (np.cos(th) ** 3)))
    N = a / (np.sqrt(1 - esq * (np.sin(lat) ** 2)))
    alt = p / np.cos(lat) - N
    lon = lon * 180 / np.pi
    lat = lat * 180 / np.pi
    return lat, lon, alt


def ecef_to_latlon_custom_ad(x, y, z):
    """
    convert from geocentric coordinates (x, y, z) to geodetic (lat, lon, alt) for automatic differentation

    IMPORTANT: the 'ad' package is unable to differentiate numpy trigonometry functions (sin, tan, etc.)
               also, 'ad.admath' can't handle lists/arrays, so x, y, z are expected to be floats here
    """
    from ad import admath as math

    a = 6378137.0
    e = 8.1819190842622e-2
    asq = a ** 2
    esq = e ** 2
    b = math.sqrt(asq * (1 - esq))
    bsq = b ** 2
    ep = math.sqrt((asq - bsq) / bsq)
    p = math.sqrt((x ** 2) + (y ** 2))
    th = math.atan2(a * z, b * p)
    lon = math.atan2(y, x)
    lat = math.atan2((z + (ep ** 2) * b * (math.sin(th) ** 3)), (p - esq * a * (math.cos(th) ** 3)))
    N = a / (math.sqrt(1 - esq * (math.sin(lat) ** 2)))
    alt = p / math.cos(lat) - N
    lon = lon * 180 / math.pi
    lat = lat * 180 / math.pi
    return lat, lon, alt

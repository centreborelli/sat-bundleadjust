"""
A Generic Bundle Adjustment Methodology for Indirect RPC Model Refinement of Satellite Imagery
author: Roger Mari <roger.mari@ens-paris-saclay.fr>
year: 2021

This script consists of a series of functions dedicated to deal with different geographic coordinate systems
and the GeoJSON format, which is used to delimit geographic areas
"""

import numpy as np
import pyproj
import utm


def utm_from_lonlat(lons, lats):
    """
    convert lon-lat to utm
    """
    return utm_from_latlon(lats, lons)


def utm_from_latlon(lats, lons):
    """
    convert lat-lon to utm
    """
    n = utm.latlon_to_zone_number(lats[0], lons[0])
    proj_src = pyproj.Proj("+proj=latlong")
    proj_dst = pyproj.Proj("+proj=utm +zone={}".format(n))
    easts, norths = pyproj.transform(proj_src, proj_dst, lons, lats)
    return easts, norths


def zonestring_from_lonlat(lon, lat):
    """
    return utm zone string from lon-lat point
    """
    n = utm.latlon_to_zone_number(lat, lon)
    l = utm.latitude_to_zone_letter(lat)
    s = "%d%s" % (n, l)
    return int(n)


def epsg_code_from_utm_zone(utm_zonestring):
    """
    Compute the EPSG code of a given utm zone
    """
    zone_number = int(utm_zonestring[:-1])
    hemisphere = utm_zonestring[-1]

    # EPSG = CONST + ZONE where CONST is
    # - 32600 for positive latitudes
    # - 32700 for negative latitudes
    const = 32600 if hemisphere == "N" else 32700
    return const + zone_number


def lonlat_from_utm(easts, norths, zonestring):
    """
    convert utm to lon-lat
    """
    proj_src = pyproj.Proj("+proj=utm +zone=%s" % zonestring)
    proj_dst = pyproj.Proj("+proj=latlong")
    return pyproj.transform(proj_src, proj_dst, easts, norths)


def utm_bbox_from_aoi_lonlat(lonlat_geojson):
    """
    compute the utm bounding box where a certain lon-lat geojson is inscribed
    """
    lons, lats = np.array(lonlat_geojson["coordinates"][0]).T
    easts, norths = utm_from_latlon(lats, lons)
    norths[norths < 0] += 10e6
    utm_bbx = {"xmin": easts.min(), "xmax": easts.max(), "ymin": norths.min(), "ymax": norths.max()}
    return utm_bbx


def utm_bbox_shape(utm_bbx, resolution):
    """
    compute height and width in rows-columns of a utm boundig box discretized at a certain resolution
    """
    height = int((utm_bbx["ymax"] - utm_bbx["ymin"]) // resolution + 1)
    width = int((utm_bbx["xmax"] - utm_bbx["xmin"]) // resolution + 1)
    return height, width


def compute_relative_utm_coords_inside_utm_bbx(pts2d_utm, utm_bbx, resolution):
    """
    given a Nx2 array of (east, north) utm coordinates, i.e. pts2d_utm,
    return a Nx2 array of (col, row) coordinates representing the pixel position
    of the utm coordinates into a utm_bbx discretized with a certain resolution
    """
    easts, norths = pts2d_utm.T
    norths[norths < 0] += 10e6
    height, width = utm_bbox_shape(utm_bbx, resolution)
    cols = (easts - utm_bbx["xmin"]) // resolution
    rows = height - (norths - utm_bbx["ymin"]) // resolution
    return np.vstack([cols, rows]).T


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


def geojson_polygon(coords_array):
    """
    define a geojson polygon from a Nx2 numpy array with N 2d coordinates delimiting a boundary
    """
    from shapely.geometry import Polygon

    # first attempt to construct the polygon, assuming the input coords_array are ordered
    # the centroid is computed using shapely.geometry.Polygon.centroid
    # taking the mean is easier but does not handle different densities of points in the edges
    pp = coords_array.tolist()
    poly = Polygon(pp)
    x_c, y_c = np.array(poly.centroid.xy).ravel()

    # check that the polygon is valid, i.e. that non of its segments intersect
    # if the polygon is not valid, then coords_array was not ordered and we have to do it
    # a possible fix is to sort points by polar angle using the centroid (anti-clockwise order)
    if not poly.is_valid:
        pp.sort(key=lambda p: np.arctan2(p[0] - x_c, p[1] - y_c))

    # construct the geojson
    geojson_polygon = {"coordinates": [pp], "type": "Polygon"}
    geojson_polygon["center"] = [x_c, y_c]
    return geojson_polygon


def geojson_to_shapely_polygon(geojson_polygon):
    """
    convert a polygon from geojson format to shapely format
    """
    from shapely.geometry import shape

    return shape(geojson_polygon)


def geojson_from_shapely_polygon(shapely_polygon):
    """
    convert a shapely polygon to geojson format
    """
    vertices = np.array(shapely_polygon.exterior.xy).T[:-1, :]
    return geojson_polygon(vertices)


def geojson_polygon_convex_hull(coords_array):
    """
    define a geojson polygon from the convex hull of a Nx2 numpy array with N 2d coordinates
    """
    from shapely.geometry import MultiPoint

    shapely_convex_hull = MultiPoint([p for p in coords_array]).convex_hull
    return geojson_from_shapely_polygon(shapely_convex_hull)


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
    from shapely.ops import cascaded_union

    geoms = [geojson_to_shapely_polygon(g) for g in utm_geojson_list]  # convert aois to shapely polygons
    union_shapely = cascaded_union([geom if geom.is_valid else geom.buffer(0) for geom in geoms])
    union_shapely = union_shapely.convex_hull if union_shapely.geom_type == "MultiPolygon" else union_shapely
    return geojson_from_shapely_polygon(union_shapely)


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


def measure_squared_km_from_lonlat_geojson(lonlat_geojson):
    """
    measure the area in squared km covered by a lonlat_geojson
    """
    utm_geojson = utm_geojson_from_lonlat_geojson(lonlat_geojson)
    area_squared_m = geojson_to_shapely_polygon(utm_geojson).area
    area_squared_km = area_squared_m * (1e-6 / 1.0)
    return area_squared_km

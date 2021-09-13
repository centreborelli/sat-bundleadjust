# Copyright (C) 2015, Carlo de Franchis <carlo.de-franchis@cmla.ens-cachan.fr>
# Copyright (C) 2015, Gabriele Facciolo <facciolo@cmla.ens-cachan.fr>
# Copyright (C) 2015, Enric Meinhardt <enric.meinhardt@cmla.ens-cachan.fr>


import warnings
import rasterio
import numpy as np

import rpcm
import srtm4

from . import geographiclib

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


def find_corresponding_point(model_a, model_b, x, y, z):
    """
    Finds corresponding points in the second image, given the heights.

    Arguments:
        model_a, model_b: two instances of the rpcm.RPCModel class, or of

            the projective_model.ProjModel class
        x, y, z: three 1D numpy arrrays, of the same length. x, y are the
        coordinates of pixels in the image, and z contains the altitudes of the
        corresponding 3D point.

    Returns:
        xp, yp, z: three 1D numpy arrrays, of the same length as the input. xp,
            yp contains the coordinates of the projection of the 3D point in image
            b.
    """
    t1, t2 = model_a.localization(x, y, z)
    xp, yp = model_b.projection(t1, t2, z)
    return (xp, yp, z)


def compute_height(model_a, model_b, x1, y1, x2, y2):
    """
    Computes the height of a point given its location inside two images.

    Arguments:
        model_a, model_b: two instances of the rpcm.RPCModel class, or of
            the projective_model.ProjModel class
        x1, y1: two 1D numpy arrrays, of the same length, containing the
            coordinates of points in the first image.
        x2, y2: two 2D numpy arrrays, of the same length, containing the
            coordinates of points in the second image.

    Returns:
        a 1D numpy array containing the list of computed heights.
    """
    n = len(x1)
    h0 = np.zeros(n)
    h0_inc = h0
    p2 = np.vstack([x2, y2]).T
    HSTEP = 1
    err = np.zeros(n)

    for i in range(100):
        tx, ty, tz = find_corresponding_point(model_a, model_b, x1, y1, h0)
        r0 = np.vstack([tx,ty]).T
        tx, ty, tz = find_corresponding_point(model_a, model_b, x1, y1, h0+HSTEP)
        r1 = np.vstack([tx,ty]).T
        a = r1 - r0
        b = p2 - r0
        # implements: h0_inc = dot(a,b) / dot(a,a)
        # For some reason, the formulation below causes massive memory leaks on
        # some systems.
        # h0_inc = np.divide(np.diag(np.dot(a, b.T)), np.diag(np.dot(a, a.T)))
        # Replacing with the equivalent:
        diagabdot = np.multiply(a[:, 0], b[:, 0]) + np.multiply(a[:, 1], b[:, 1])
        diagaadot = np.multiply(a[:, 0], a[:, 0]) + np.multiply(a[:, 1], a[:, 1])
        h0_inc = np.divide(diagabdot, diagaadot)
#        if np.any(np.isnan(h0_inc)):
#            print(x1, y1, x2, y2)
#            print(a)
#            return h0, h0*0
        # implements:   q = r0 + h0_inc * a
        q = r0 + np.dot(np.diag(h0_inc), a)
        # implements: err = sqrt(dot(q-p2, q-p2))
        tmp = q-p2
        err =  np.sqrt(np.multiply(tmp[:, 0], tmp[:, 0]) + np.multiply(tmp[:, 1], tmp[:, 1]))
#       print(np.arctan2(tmp[:, 1], tmp[:, 0])) # for debug
#       print(err) # for debug
        h0 = np.add(h0, h0_inc*HSTEP)
        # implements: if fabs(h0_inc) < 0.0001:
        if np.max(np.fabs(h0_inc)) < 0.001:
            break

    return h0, err


def geodesic_bounding_box(rpc, x, y, w, h):
    """
    Computes a bounding box on the WGS84 ellipsoid associated to a Pleiades
    image region of interest, through its rpc function.

    Args:
        rpc: instance of the rpcm.RPCModel class
        x, y, w, h: four integers defining a rectangular region of interest
            (ROI) in the image. (x, y) is the top-left corner, and (w, h) are
            the dimensions of the rectangle.

    Returns:
        4 geodesic coordinates: the min and max longitudes, and the min and
        max latitudes.
    """
    # compute altitude coarse extrema from rpc data
    m = rpc.alt_offset - rpc.alt_scale
    M = rpc.alt_offset + rpc.alt_scale

    # build an array with vertices of the 3D ROI, obtained as {2D ROI} x [m, M]
    x = np.array([x, x,   x,   x, x+w, x+w, x+w, x+w])
    y = np.array([y, y, y+h, y+h,   y,   y, y+h, y+h])
    a = np.array([m, M,   m,   M,   m,   M,   m,   M])

    # compute geodetic coordinates of corresponding world points
    lon, lat = rpc.localization(x, y, a)

    # extract extrema
    # TODO: handle the case where longitudes pass over -180 degrees
    # for latitudes it doesn't matter since for latitudes out of [-60, 60]
    # there is no SRTM data
    return np.min(lon), np.max(lon), np.min(lat), np.max(lat)


def altitude_range_coarse(rpc, scale_factor=1):
    """
    Computes a coarse altitude range using the RPC informations only.

    Args:
        rpc: instance of the rpcm.RPCModel class
        scale_factor: factor by which the scale offset is multiplied

    Returns:
        the altitude validity range of the RPC.
    """
    m = rpc.alt_offset - scale_factor * rpc.alt_scale
    M = rpc.alt_offset + scale_factor * rpc.alt_scale
    return m, M


def utm_zone(rpc, x, y, w, h):
    """
    Compute the UTM zone where the ROI probably falls (or close to its border).

    Args:
        rpc: instance of the rpcm.RPCModel class, or path to a GeoTIFF file
        x, y, w, h: four integers defining a rectangular region of interest
            (ROI) in the image. (x, y) is the top-left corner, and (w, h)
            are the dimensions of the rectangle.

    Returns:
        a string of the form '18N' or '18S' where 18 is the utm zone
        identificator.
    """
    # read rpc file
    if not isinstance(rpc, rpcm.RPCModel):
        rpc = rpcm.rpc_from_geotiff(rpc)

    # determine lat lon of the center of the roi, assuming median altitude
    lon, lat = rpc.localization(x + .5*w, y + .5*h, rpc.alt_offset)[:2]

    return geographiclib.compute_utm_zone(lon, lat)


def generate_point_mesh(col_range, row_range, alt_range):
    """
    Generates image coordinates (col, row, alt) of 3D points located on the grid
    defined by col_range and row_range, at uniformly sampled altitudes defined
    by alt_range.
    Args:
        col_range: triplet (col_min, col_max, n_col), where n_col is the
            desired number of samples
        row_range: triplet (row_min, row_max, n_row)
        alt_range: triplet (alt_min, alt_max, n_alt)

    Returns:
        3 lists, containing the col, row and alt coordinates.
    """
    # input points in col, row, alt space
    cols, rows, alts = [np.linspace(v[0], v[1], v[2]) for v in
            [col_range, row_range, alt_range]]

    # make it a kind of meshgrid (but with three components)
    # if cols, rows and alts are lists of length 5, then after this operation
    # they will be lists of length 5x5x5
    cols, rows, alts =\
            (  cols+0*rows[:,np.newaxis]+0*alts[:,np.newaxis,np.newaxis]).reshape(-1),\
            (0*cols+  rows[:,np.newaxis]+0*alts[:,np.newaxis,np.newaxis]).reshape(-1),\
            (0*cols+0*rows[:,np.newaxis]+  alts[:,np.newaxis,np.newaxis]).reshape(-1)

    return cols, rows, alts


def ground_control_points(rpc, x, y, w, h, m, M, n):
    """
    Computes a set of ground control points (GCP), corresponding to RPC data.

    Args:
        rpc: instance of the rpcm.RPCModel class
        x, y, w, h: four integers defining a rectangular region of interest
            (ROI) in the image. (x, y) is the top-left corner, and (w, h) are
            the dimensions of the rectangle.
        m, M: minimal and maximal altitudes of the ground control points
        n: cube root of the desired number of ground control points.

    Returns:
        a list of world points, given by their geodetic (lon, lat, alt)
        coordinates.
    """
    # points will be sampled in [x, x+w] and [y, y+h]. To avoid always sampling
    # the same four corners with each value of n, we make these intervals a
    # little bit smaller, with a dependence on n.
    col_range = [x+(1.0/(2*n))*w, x+((2*n-1.0)/(2*n))*w, n]
    row_range = [y+(1.0/(2*n))*h, y+((2*n-1.0)/(2*n))*h, n]
    alt_range = [m, M, n]
    col, row, alt = generate_point_mesh(col_range, row_range, alt_range)
    lon, lat = rpc.localization(col, row, alt)
    return lon, lat, alt


def matches_from_rpc(rpc1, rpc2, x, y, w, h, n):
    """
    Uses RPC functions to generate matches between two Pleiades images.

    Args:
        rpc1, rpc2: two instances of the rpcm.RPCModel class
        x, y, w, h: four integers defining a rectangular region of interest
            (ROI) in the first view. (x, y) is the top-left corner, and (w, h)
            are the dimensions of the rectangle. In the first view, the matches
            will be located in that ROI.
        n: cube root of the desired number of matches.

    Returns:
        an array of matches, one per line, expressed as x1, y1, x2, y2.
    """
    m, M = altitude_range_coarse(rpc1)
    lon, lat, alt = ground_control_points(rpc1, x, y, w, h, m, M, n)
    x1, y1 = rpc1.projection(lon, lat, alt)
    x2, y2 = rpc2.projection(lon, lat, alt)

    return np.vstack([x1, y1, x2, y2]).T


def gsd_from_rpc(rpc, z=0):
    """
    Compute an image ground sampling distance from its RPC camera model.

    Args:
        rpc (rpcm.RPCModel): camera model
        z (float, optional): ground elevation

    Returns:
        float (meters per pixel)
    """
    # we assume that RPC col/row offset match the coordinates of image center
    c = rpc.col_offset
    r = rpc.row_offset

    a = geographiclib.lonlat_to_geocentric(*rpc.localization(c+0, r, z), alt=z)
    b = geographiclib.lonlat_to_geocentric(*rpc.localization(c+1, r, z), alt=z)
    return np.linalg.norm(np.asarray(b) - np.asarray(a))

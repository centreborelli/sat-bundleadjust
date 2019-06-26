"""
* affine triangulation
* display point clouds
* DEM projection

Copyright (C) 2018, Enric Meinhardt-Llopis <enric.meinhardt@cmla.ens-cachan.fr>
Copyright (C) 2018, Carlo de Franchis <carlo.de-franchis@ens-cachan.fr>
Copyright (C) 2018, Gabriele Facciolo <facciolo@cmla.ens-cachan.fr>
"""

import numpy as np


def triangulation_affine(PA, PB, x1, y1, x2, y2):
    """
    Triangulate a (list of) match(es) between two images of affine cameras.

    Arguments:
        PA, PB : affine (projection) camera matrices of the two images
        x1, y1 : pixel coordinates in the domain of the first image
        x2, y2 : pixel coordinates in the domain of the second image

    Return value: a 4-tuple (lon, lat, h, e)
        lon, lat, h, e : coordinates of the 3D point(s), reprojection error
    """

    # build projection and localization matrices as 4x4 homogeneous maps
    # (x,y,h,1) <-> (lon,lat,h,1)
    PA = np.vstack([ PA[0:2], [0,0,1,0], [0,0,0,1]])  # pick only first two rows
    PB = np.vstack([ PB[0:2], [0,0,1,0], [0,0,0,1]])  # pick only first two rows
    LA = np.linalg.inv(PA)  # inverse of a 4x4 matrix

    # affine epipolar correspondence
    E = PB @ LA

    # Now, the linear system E * (x1, y1, h, 1) = (x2, y2, h, 1)
    # has two different equations and one unknown h.  We solve it by
    # least squares (a simple projection, in this case).

    # give names to the 8 non-trivial coefficients of E
    a, b, p, r = E[0]
    c, d, q, s = E[1]

    # coefficients of the affine triangulation
    f = [-p*a - q*c, -p*b - q*d, p, q, -p*r - q*s ] / (p*p + q*q)

    # apply the triangulation (first use of the input points)
    h = f[0]*x1 + f[1]*y1 + f[2]*x2 + f[3]*y2 + f[4]

    # finish the computation and return the 4 required numbers (or vectors)
    lon = LA[0,0] * x1 + LA[0,1] * y1 + LA[0,2] * h + LA[0,3]
    lat = LA[1,0] * x1 + LA[1,1] * y1 + LA[1,2] * h + LA[1,3]
    ex = E[0,0] * x1 + E[0,1] * y1 + E[0,2] * h + E[0,3] - x2
    ey = E[1,0] * x1 + E[1,1] * y1 + E[1,2] * h + E[1,3] - y2
    e = ex * ex + ey * ey
    return lon, lat, h, e


def triangulate_disparities(dmap, rpc1, rpc2, S1, S2, PA, PB,):
    """
    Triangulate a disparity map

    Arguments:
        dmap : a disparity map between two rectified images
        rpc1, rpc2 : calibration data of each image
        S1, S2 : rectifying affine maps (from the domain of the fullsize images)
        PA, PB : the affine approximations of rpc1 and rpc2 (not always used)

    Return:
        xyz : a matrix of size Nx3 (where N is the number of finite disparites
              in dmap) this matrix contains the coordinates of the 3d points in
              "lon,lat,h" or "e,n,h"
    """
    from utils import utm_from_lonlat

    # 1. unroll all the valid (finite) disparities of dmap into a vector
    m = np.isfinite(dmap.flatten())
    x = np.argwhere(np.isfinite(dmap))[:,1]  # attention to order of the indices
    y = np.argwhere(np.isfinite(dmap))[:,0]
    d = dmap.flatten()[m]

    # 2. for each disparity

    # 2.1. produce a pair in the original domain by composing with S1 and S2
    p = np.linalg.inv(S1) @ np.vstack( (x+0, y, np.ones(len(d))) )
    q = np.linalg.inv(S2) @ np.vstack( (x+d, y, np.ones(len(d))) )

    # 2.2. triangulate the pair of image points to find a 3D point (in UTM)
    lon, lat, h, e = triangulation_affine(PA,PB, p[0,:],p[1,:], q[0,:],q[1,:])
    #east, north, _, _ = np.vectorize(utm.from_latlon)(lat, lon)
    east, north = utm_from_lonlat(lon, lat)

    # 2.3. add this point to the output list
    xyz = np.vstack((east, north, h)).T

    return xyz


# code for projection into UTM coordinates
from numba import jit

# utility function for computing the average at each cell
@jit
def reduceavg(w, h,  ix, iy, z):
    D_sum = np.zeros((h,w))  #-np.ones((h,w))*np.inf
    D_cnt = np.zeros((h,w))
    for t in range(len(ix)):
        if iy[t]<0 or ix[t]<0 or iy[t]>=h or ix[t]>=w:
            continue
        ty = iy[t]
        tx = ix[t]
        D_sum[ty,tx] += z[t]
        D_cnt[ty,tx] += 1

    D_sum /= D_cnt*(D_cnt>0) + 1*(D_cnt==0) # needed for computing average
    
    # important not to consider unknown values in the later steps
    idx_nan_values = (D_cnt==0)
    D_sum_nan = D_sum.copy()
    D_sum_nan[idx_nan_values] = np.nan
    
    return D_sum, D_sum_nan

# utility function for computing the std deviation at each cell
@jit
def getcloudstd(w, h,  ix, iy, z, dem):
    D_sum = np.zeros((h,w))  #-np.ones((h,w))*np.inf
    D_cnt = np.zeros((h,w))
    
    for t in range(len(ix)):
        if iy[t]<0 or ix[t]<0 or iy[t]>=h or ix[t]>=w:
            continue
        ty = iy[t]
        tx = ix[t]
        D_sum[ty,tx] += (z[t] - dem[ty,tx]) ** 2
        D_cnt[ty,tx] += 1
    
    D_sum /= D_cnt*(D_cnt>0) + 1*(D_cnt==0) # needed for computing average
    
    return np.sqrt(D_sum)

# projects them into a grid defined by the bunding box (emin, emax, nmin, nmax)
def project_cloud_into_utm_grid(xyz, emin, emax, nmin, nmax, resolution=1):
    """
    Project a point cloud into an utm grid to produce a DEM
    The algorithm averages all the points that fall into each square of the grid

    Arguments:
        xyz : a Nx3 matrix representing a point cloud in (lon,lat,h) coordinates
        emin,emax,nmin,nmax : a bounding box in UTM coordinates
        resolution : the target resolution in meters (by default, 1 meter)

    Return:
        dem : a 2D array of heights in meters
    """

    # width and height of the image domain
    w = int(np.ceil((emax - emin)/resolution))
    h = int(np.ceil((nmax - nmin)/resolution))

    # extract and quantize columns
    x = xyz[:,0]
    y = xyz[:,1]
    z = xyz[:,2]

    ix = np.asarray((x - emin)/resolution, dtype="int")
    iy = np.asarray((nmax - y)/resolution, dtype="int")

    # per-cell average
    dem, dem_nan = reduceavg (w,h,  ix, iy, z)
    
    # per-cell std
    dem_std = getcloudstd (w,h,  ix, iy, z, dem)
    
    return dem, dem_nan, dem_std
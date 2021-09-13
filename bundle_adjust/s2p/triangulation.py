# Copyright (C) 2015, Carlo de Franchis <carlo.de-franchis@cmla.ens-cachan.fr>
# Copyright (C) 2015, Gabriele Facciolo <facciolo@cmla.ens-cachan.fr>
# Copyright (C) 2015, Enric Meinhardt <enric.meinhardt@cmla.ens-cachan.fr>

import os
import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
from ctypes import c_int, c_float, c_double, byref, POINTER

from . import geographiclib

here = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(os.path.dirname(here), 'lib', 'disp_to_h.so')
lib = ctypes.CDLL(lib_path)


class RPCStruct(ctypes.Structure):
    """
    ctypes version of the RPC C struct defined in rpc.h.
    """
    _fields_ = [("numx", c_double * 20),
                ("denx", c_double * 20),
                ("numy", c_double * 20),
                ("deny", c_double * 20),
                ("scale", c_double * 3),
                ("offset", c_double * 3),
                ("inumx", c_double * 20),
                ("idenx", c_double * 20),
                ("inumy", c_double * 20),
                ("ideny", c_double * 20),
                ("iscale", c_double * 3),
                ("ioffset", c_double * 3),
                ("dmval", c_double * 4),
                ("imval", c_double * 4),
                ("delta", c_double)]

    def __init__(self, rpc, delta=1.0):
        """
        Args:
            rpc (rpcm.RPCModel): rpc model
        """
        self.offset[0] = rpc.col_offset
        self.offset[1] = rpc.row_offset
        self.offset[2] = rpc.alt_offset
        self.ioffset[0] = rpc.lon_offset
        self.ioffset[1] = rpc.lat_offset
        self.ioffset[2] = rpc.alt_offset

        self.scale[0] = rpc.col_scale
        self.scale[1] = rpc.row_scale
        self.scale[2] = rpc.alt_scale
        self.iscale[0] = rpc.lon_scale
        self.iscale[1] = rpc.lat_scale
        self.iscale[2] = rpc.alt_scale

        for i in range(20):
            self.inumx[i] = rpc.col_num[i]
            self.idenx[i] = rpc.col_den[i]
            self.inumy[i] = rpc.row_num[i]
            self.ideny[i] = rpc.row_den[i]

        if hasattr(rpc, 'lat_num'):
            for i in range(20):
                self.numx[i] = rpc.lon_num[i]
                self.denx[i] = rpc.lon_den[i]
                self.numy[i] = rpc.lat_num[i]
                self.deny[i] = rpc.lat_den[i]
        else:
            for i in range(20):
                self.numx[i] = np.nan
                self.denx[i] = np.nan
                self.numy[i] = np.nan
                self.deny[i] = np.nan

        # initialization factor for iterative localization
        self.delta = delta


def stereo_corresp_to_xyz(rpc1, rpc2, pts1, pts2, out_crs=None):
    """
    Compute a point cloud from stereo correspondences between two images using RPC camera models.
    No need to go through the disparity map

    Args:
        rpc1, rpc2 (rpcm.RPCModel): camera models
        pts1, pts2 (arrays): 2D arrays of shape (N, 2) containing the image coordinates of
            N 2d keypoints matched beween im1 and im2,
            i.e. cooridnates in the same row of these arrays back-project to the same 3D point
        out_crs (pyproj.crs.CRS): object defining the desired coordinate reference system for the
            output xyz map
    Returns:
        xyz: array of shape (h, w, 3) where each pixel contains the 3D
            coordinates of the triangulated point in the coordinate system
            defined by `out_crs`
        err: array of shape (h, w) where each pixel contains the triangulation
            error
    """
    # copy rpc coefficients to an RPCStruct object
    rpc1_c_struct = RPCStruct(rpc1, delta=0.1)
    rpc2_c_struct = RPCStruct(rpc2, delta=0.1)

    # get number of points to triangulate
    n = pts1.shape[0]

    # define the argument types of the stereo_corresp_to_lonlatalt function from disp_to_h.so
    lib.stereo_corresp_to_lonlatalt.argtypes = (ndpointer(dtype=c_double, shape=(n, 3)),
                                                ndpointer(dtype=c_float, shape=(n, 1)),
                                                ndpointer(dtype=c_float, shape=(n, 2)),
                                                ndpointer(dtype=c_float, shape=(n, 2)),
                                                c_int, POINTER(RPCStruct), POINTER(RPCStruct))

    # call the stereo_corresp_to_lonlatalt function from disp_to_h.so
    lonlatalt =  np.zeros((n, 3), dtype='float64')
    err =  np.zeros((n, 1), dtype='float32')
    lib.stereo_corresp_to_lonlatalt(lonlatalt, err,
                                    pts1.astype('float32'), pts2.astype('float32'),
                                    n, byref(rpc1_c_struct), byref(rpc2_c_struct))

    # output CRS conversion
    in_crs = geographiclib.pyproj_crs("epsg:4979")

    if out_crs and out_crs != in_crs:

        x, y, z = geographiclib.pyproj_transform(lonlatalt[:, 0], lonlatalt[:, 1],
                                                 in_crs, out_crs, lonlatalt[:, 2])

        xyz_array = np.column_stack((x, y, z)).astype(np.float64)
    else:
        xyz_array = lonlatalt

    return xyz_array, err


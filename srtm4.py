#!/usr/bin/env python
# vim: set fileencoding=utf-8
# pylint: disable=C0103

"""
Module to download terrain digital elevation models from the SRTM 90m DEM.

Copyright (C) 2016, Carlo de Franchis <carlo.de-franchis@ens-cachan.fr>
"""

from __future__ import print_function
import subprocess
import zipfile
import os

import numpy as np
import requests
import filelock


BIN = '/usr/local/bin'
SRTM_DIR = os.path.join(os.path.expanduser('~'), '.srtm')
SRTM_URL = 'http://data_public:GDdci@data.cgiar-csi.org/srtm/tiles/GeoTIFF'


def download(to_file, from_url):
    """
    Download a file from the internet.

    Args:
        to_file: path where to store the downloaded file
        from_url: url of the file to download
    """
    r = requests.get(from_url, stream=True)
    file_size = int(r.headers['content-length'])
    print("Downloading: {} Bytes: {}".format(to_file, file_size))

    with open(to_file, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def get_srtm_tile(srtm_tile, out_dir):
    """
    Download and unzip an srtm tile from the internet.

    Args:
        srtm_tile: string following the pattern 'srtm_%02d_%02d', identifying
            the desired strm tile
        out_dir: directory where to store and extract the srtm tiles
    """
    # check if the tile is already there
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    srtm_zip_download_lock = os.path.join(out_dir, 'srtm_zip.lock')
    srtm_tif_write_lock = os.path.join(out_dir, 'srtm_tif.lock')

    if os.path.exists(os.path.join(out_dir, '{}.tif'.format(srtm_tile))):
        # the tif file is either being written or finished writing
        # locking will ensure it is not being written.
        # Also by construction we won't write on something complete.
        lock_tif = filelock.FileLock(srtm_tif_write_lock)
        lock_tif.acquire()
        lock_tif.release()
        return

    # download the zip file
    srtm_tile_url = '{}/{}.zip'.format(SRTM_URL, srtm_tile)
    zip_path = os.path.join(out_dir, '{}.zip'.format(srtm_tile))

    lock_zip = filelock.FileLock(srtm_zip_download_lock)
    lock_zip.acquire()

    if os.path.exists(os.path.join(out_dir, '{}.tif'.format(srtm_tile))):
        # since the zip lock is returned after the tif lock
        # if we end up here, it means another process downloaded the zip
        # and extracted it.
        # No need to wait on lock_tif
        lock_zip.release()
        return

    if os.path.exists(zip_path):
        print ('zip already exists')
        # Only possibility here is that the previous process was cut short

    download(zip_path, srtm_tile_url)

    lock_tif = filelock.FileLock(srtm_tif_write_lock)
    lock_tif.acquire()

    # extract the tif file
    if zipfile.is_zipfile(zip_path):
        z = zipfile.ZipFile(zip_path, 'r')
        z.extract('{}.tif'.format(srtm_tile), out_dir)
    else:
        print('{} not available'.format(srtm_tile))

    # remove the zip file
    os.remove(zip_path)

    # release locks
    lock_tif.release()
    lock_zip.release()


def srtm4(lon, lat):
    """
    Gives the SRTM height of a (list of) point(s).

    Args:
        lon, lat: lists of longitudes and latitudes (same length), or single
            longitude and latitude

    Returns:
        height(s) in meters above the WGS84 ellipsoid (not the EGM96 geoid)
    """
    # determine the needed srtm tiles by running the srtm4_which_tile binary
    here = os.path.dirname(__file__)
    p = subprocess.Popen(['%s/srtm4/srtm4_which_tile'%here],
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         env={'PATH': BIN, 'SRTM4_CACHE': SRTM_DIR})

    # feed it from stdin
    try:
        lon_lats = '\n'.join('{} {}'.format(a, b) for a, b in zip(lon, lat))
    except TypeError:
        lon_lats = '{} {}'.format(lon, lat)
    outs, errs = p.communicate(input=lon_lats.encode())

    # read the list of needed tiles
    srtm_tiles = outs.decode().split()

    # download the tiles if not already there
    for srtm_tile in set(srtm_tiles):
        get_srtm_tile(srtm_tile, SRTM_DIR)

    # run the srtm4 binary and feed it from stdin
    p = subprocess.Popen(['%s/srtm4/srtm4'%here],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         env={'PATH': BIN, 'SRTM4_CACHE': SRTM_DIR})
    outs, errs = p.communicate(input=lon_lats.encode())

    # return the altitudes
    alts = list(map(float, outs.decode().split()))
    return alts if isinstance(lon, (list, np.ndarray)) else alts[0]

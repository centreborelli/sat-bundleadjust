import numpy as np
import matplotlib.pyplot as plt
import os

from PIL import Image
from bundle_adjust import data_loader as loader


def reproject_pts3d_and_compute_errors(cam_before, cam_after, cam_model, obs2d, pts3d_before, pts3d_after,
                                        image_fname=None, verbose=False):

    if image_fname is not None and not os.path.exists(image_fname):
        image_fname = None

    from bundle_adjust.camera_utils import project_pts3d
    # open image if available
    image = np.array(Image.open(image_fname)) if (image_fname is not None) else None
    # reprojections before bundle adjustment
    pts2d_before = project_pts3d(cam_before, cam_model, pts3d_before)
    # reprojections after bundle adjustment
    pts2d_after = project_pts3d(cam_after, cam_model, pts3d_after)
    # compute average residuals and reprojection errors
    avg_residuals = np.mean(abs(pts2d_after - obs2d), axis=1)/2.0
    err_before = np.linalg.norm(pts2d_before - obs2d, axis=1)
    err_after = np.linalg.norm(pts2d_after - obs2d, axis=1)

    if verbose:

        print('path to image: {}'.format(image_fname))
        args = [np.mean(err_before), np.median(err_before)]
        print('Reprojection error before BA (mean / median): {:.2f} / {:.2f}'.format(*args))
        args = [np.mean(err_after), np.median(err_after)]
        print('Reprojection error after  BA (mean / median): {:.2f} / {:.2f}\n'.format(*args))
        # reprojection error histograms for the selected image
        fig = plt.figure(figsize=(10,3))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.title.set_text('Reprojection error before BA')
        ax2.title.set_text('Reprojection error after  BA')
        ax1.hist(err_before, bins=40)
        ax2.hist(err_after, bins=40)
        #ax2.hist(err_after, bins=40, range=(err_before.min(), err_before.max()))
        plt.show()      

        plot = True
        if image is not None and plot:
            # warning: this is slow...
            # green crosses represent the observations from feature tracks seen in the image,
            # red vectors are the distance to the reprojected point locations.
                fig = plt.figure(figsize=(20,6))
                ax1 = fig.add_subplot(121)
                ax2 = fig.add_subplot(122)
                ax1.title.set_text('Before BA')
                ax2.title.set_text('After  BA')
                ax1.imshow(loader.custom_equalization(image), cmap="gray")
                ax2.imshow(loader.custom_equalization(image), cmap="gray")
                for k in range(min(3000, obs2d.shape[0])):
                    # before bundle adjustment
                    ax1.plot([obs2d[k, 0], pts2d_before[k, 0]], [obs2d[k, 1], pts2d_before[k, 1]], 'r-', lw=3)
                    ax1.plot(*obs2d[k], 'yx')
                    # after bundle adjustment
                    ax2.plot([obs2d[k, 0], pts2d_after[k, 0]], [obs2d[k, 1], pts2d_after[k, 1]], 'r-', lw=3)
                    ax2.plot(*obs2d[k], 'yx')
                plt.show()
    
    return pts2d_before, pts2d_after, err_before, err_after, avg_residuals


def warp_stereo_dsms(complete_dsm_fname, stereo_dsms_fnames):
    
    n_dsms = len(stereo_dsms_fnames)
    
    # warping dsms
    for dsm_idx, src_fname in enumerate(stereo_dsms_fnames):

        dst_fname = loader.add_suffix_to_fname(src_fname, 'warp')
        os.makedirs(os.path.dirname(dst_fname), exist_ok=True)

        args = [src_fname, dst_fname, complete_dsm_fname]

        if os.path.isfile(dst_fname):
            continue
        os.system('rio clip {} {} --like {} --with-complement --overwrite'.format(*args))

        if not os.path.isfile(dst_fname):
            print(' ERROR ! gdalwarp failed !') 
            print(dst_fname)

        print('\rClipping DSMs... {}/{}'.format(dsm_idx+1, n_dsms), end='\r')
    print('\n')



def compute_stat_for_specific_date_from_tiles(complete_dsm_fname, stereo_dsms_fnames,
                                              tile_size=500, output_dir=None, stat='std',
                                              clean_tmp_warps=True, clean_tmp_tiles=True, mask=None):

    import rasterio
    import warnings
    warnings.filterwarnings("ignore")
    
    warp_stereo_dsms(complete_dsm_fname, stereo_dsms_fnames)
    warp_fnames = [loader.add_suffix_to_fname(fn, 'warp') for fn in stereo_dsms_fnames]
    
    h, w = loader.read_image_size(complete_dsm_fname)

    tiles_dir = os.path.join(output_dir, 'tmp_tiles_{}_{}'.format(loader.get_id(complete_dsm_fname), stat))
    os.makedirs(tiles_dir, exist_ok=True)

    m = tile_size

    y_lims = np.arange(0, h, m).astype(int)
    x_lims = np.arange(0, w, m).astype(int)

    n_tiles = len(y_lims)*len(x_lims)
    tile_idx = 0
    for row in y_lims:
        for col in x_lims:
            crops = []
            tile_idx += 1

            limit_row = row + int(m if row+m < h else h - row)
            limit_col = col + int(m if col+m < w else w - col)

            tile_fn = os.path.join(tiles_dir, 'row{}_col{}.tif'.format(row, col))
            if os.path.isfile(tile_fn):
                continue
            
            for fn in warp_fnames:
                with rasterio.open(fn) as src:
                    crops.append(src.read(window=((row, limit_row), (col, limit_col))).squeeze())
            dsm_ndarray = np.dstack(crops)
            
            if stat == 'std':
                counts_per_coord = np.sum(1*~np.isnan(dsm_ndarray), axis=2)
                overlapping_coords_mask = counts_per_coord >= 2
                tile_stat = np.nanstd(dsm_ndarray, axis=2)
                tile_stat[~overlapping_coords_mask] = np.nan
            else:
                tile_stat = np.nanmean(dsm_ndarray, axis=2)

            Image.fromarray(tile_stat).save(tile_fn)

            print('\rComputing tiles... {}/{}'.format(tile_idx, n_tiles), end='\r')
    print('\n')

    stat_per_date = np.zeros((h,w))
    stat_per_date[:] = np.nan
    for row in y_lims:
        for col in x_lims:
            tile_fn = os.path.join(tiles_dir, 'row{}_col{}.tif'.format(row, col))
            tile_im = np.array(Image.open(tile_fn))
            tile_h, tile_w = tile_im.shape
            stat_per_date[row:row + tile_h, col:col + tile_w] = tile_im

    #from scipy import ndimage
    #im_filt = ndimage.median_filter(stat_per_date, size=3)
    #im_filt[np.isnan(stat_per_date)] = np.nan
    #stat_per_date = im_filt

    #clean temporary files
    if clean_tmp_tiles:
        os.system('rm -r {}'.format(tiles_dir))
    if clean_tmp_warps:
        for fn in warp_fnames:
            os.system('rm {}'.format(fn))

    # write geotiff
    import rasterio
    output_fn = complete_dsm_fname.replace(os.path.dirname(complete_dsm_fname), output_dir)
    raster = stat_per_date.astype(rasterio.float32)
    with rasterio.open(complete_dsm_fname) as src_data:
        kwds = src_data.profile
        with rasterio.open(output_fn, 'w', **kwds) as dst_data:
            if mask is not None:
                raster = loader.apply_mask_to_raster(raster, mask)
            dst_data.write(raster, 1)

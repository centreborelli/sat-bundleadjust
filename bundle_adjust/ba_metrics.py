import numpy as np
import matplotlib.pyplot as plt

import os

from PIL import Image
from bundle_adjust import data_loader as loader

def reprojection_error_from_C(P_before, P_after, pts_gt, pts_3d_before, pts_3d_after, C, image_fname=None, verbose=False):

    # open image
    if image_fname is not None:
        image = np.array(Image.open(image_fname))
    else:
        image = None
    
    # reprojections before bundle adjustment
    proj = P_before @ np.hstack((pts_3d_before, np.ones((pts_3d_before.shape[0],1)))).T
    pts_reproj_before = (proj[:2,:]/proj[-1,:]).T

    # reprojections after bundle adjustment
    proj = P_after @ np.hstack((pts_3d_after, np.ones((pts_3d_after.shape[0],1)))).T
    pts_reproj_after = (proj[:2,:]/proj[-1,:]).T

    avg_residuals = np.mean(abs(pts_reproj_after - pts_gt), axis=1)/2.0
    
    err_before = np.linalg.norm(pts_reproj_before - pts_gt, axis=1)
    err_after = np.linalg.norm(pts_reproj_after - pts_gt, axis=1)
    
    
    if image is not None and verbose:
        
        print('{}, mean abs reproj error before BA: {:.4f}'.format(image_fname, np.mean(err_before)))
        print('{}, mean abs reproj error after  BA: {:.4f}'.format(image_fname, np.mean(err_after)))

        # reprojection error histograms for the selected image
        fig = plt.figure(figsize=(10,3))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.title.set_text('Reprojection error before BA')
        ax2.title.set_text('Reprojection error after  BA')
        ax1.hist(err_before, bins=40); 
        ax2.hist(err_after, bins=40);
        plt.show()      

        plot = True
        if plot:
        # warning: this is slow...
        # Green crosses represent the observations from feature tracks seen in the image, 
        # red vectors are the distance to the reprojected point locations. 
            fig = plt.figure(figsize=(20,6))
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            ax1.title.set_text('Before BA')
            ax2.title.set_text('After  BA')
            ax1.imshow(image, cmap="gray")
            ax2.imshow(image, cmap="gray")
            for k in range(min(3000,pts_gt.shape[0])):
                # before bundle adjustment
                ax1.plot([pts_gt[k,0], pts_reproj_before[k,0] ], [pts_gt[k,1], pts_reproj_before[k,1] ], 'r-', lw=3)
                ax1.plot(*pts_gt[k], 'yx')
                # after bundle adjustment
                ax2.plot([pts_gt[k,0], pts_reproj_after[k,0] ], [pts_gt[k,1], pts_reproj_after[k,1]], 'r-', lw=3)
                ax2.plot(*pts_gt[k], 'yx')
            plt.show()
    
    return avg_residuals, err_before, err_after, pts_gt, pts_reproj_before, pts_reproj_after




def warp_stereo_dsms(complete_dsm_fname, stereo_dsms_fnames):
    
    n_dsms = len(stereo_dsms_fnames)
    
    # warping dsms
    print('\nClipping dsms...') 
    for dsm_idx, src_fname in zip(np.arange(n_dsms), stereo_dsms_fnames):

        dst_fname = loader.add_suffix_to_fname(src_fname, 'warp')
        os.makedirs(os.path.dirname(dst_fname), exist_ok=True)

        args = [src_fname, dst_fname, complete_dsm_fname]

        if os.path.isfile(dst_fname):
            continue
        os.system('rio clip {} {} --like {} --with-complement --overwrite'.format(*args))

        if not os.path.isfile(dst_fname):
            print(' ERROR ! gdalwarp failed !') 
            print(dst_fname)

        print('\r{} dsms / {}'.format(dsm_idx+1, n_dsms),end='\r')

    print('\nDone!\n')



def compute_std_for_specific_date_from_tiles(complete_dsm_fname, stereo_dsms_fnames, tile_size=500, output_dir=None):
    
    print('\n###################################################################################')
    print('Computing std for specific date...')
    print('  - complete_dsm_fname: {}'.format(complete_dsm_fname))
    print('  - tile_size: {}'.format(tile_size))
    print('###################################################################################\n')
    
    import warnings
    warnings.filterwarnings("ignore")
    
    from IS18.utils import rio_open
    
    warp_stereo_dsms(complete_dsm_fname, stereo_dsms_fnames)
    warp_fnames = [loader.add_suffix_to_fname(fn, 'warp') for fn in stereo_dsms_fnames]
    
    h, w = loader.read_image_size(complete_dsm_fname)

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

            tile_fn = loader.add_suffix_to_fname(complete_dsm_fname, 'row{}_col{}_tmp_std_tile'.format(row, col))
            if os.path.isfile(tile_fn):
                continue
            
            for fn in warp_fnames:
                with rio_open(fn, 'r') as src:
                    crops.append(src.read(window=((row, limit_row), (col, limit_col))).squeeze())

            dsm_ndarray = np.dstack(crops)
            counts_per_coord = np.sum(1*~np.isnan(dsm_ndarray), axis=2)
            overlapping_coords_mask = counts_per_coord >= 2
            std_along_time = np.nanstd(dsm_ndarray, axis=2)
            std_along_time[~overlapping_coords_mask] = np.nan

            
            os.makedirs(os.path.dirname(tile_fn), exist_ok=True)
            Image.fromarray(std_along_time).save(tile_fn)

            print('\r{} tiles / {}'.format(tile_idx, n_tiles), end='\r')

    std_per_date = np.zeros((h,w))
    std_per_date[:] = np.nan
    for row in y_lims:
        for col in x_lims:
            tile_fn = loader.add_suffix_to_fname(complete_dsm_fname, 'row{}_col{}_tmp_std_tile'.format(row, col))
            tile_im = np.array(Image.open(tile_fn))
            tile_h, tile_w = tile_im.shape
            std_per_date[row:row + tile_h, col:col + tile_w] = tile_im
    
    #clean temporary files
    os.system('rm -r {}/{}_*_tmp_std_tile.tif'.format(os.path.dirname(complete_dsm_fname), 
                                                      loader.get_id(complete_dsm_fname)))
    for fn in warp_fnames:
        os.system('rm {}'.format(fn))
    
    dsm_per_date_fn = loader.add_suffix_to_fname(complete_dsm_fname, 'std')
    Image.fromarray(std_per_date).save(dsm_per_date_fn)
    
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        os.system('mv {} {}'.format(dsm_per_date_fn, output_dir))
    
    print('\nDone!\n')
              
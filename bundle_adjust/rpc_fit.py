import numpy as np
import matplotlib.pyplot as plt

from IS18 import vistools
from bundle_adjust import ba_core
from bundle_adjust import ba_utils
from bundle_adjust import ba_rotate
from bundle_adjust import camera_utils
import rpcm

def poly_vect(x, y, z):
    """
    Returns evaluated polynomial vector without the first constant term equal to 1,
    using the order convention defined in rpc_model.apply_poly
    """
    return np.array([y, x, z, 
                     y*x, y*z, x*z, y*y, x*x, z*z, 
                     x*y*z, y*y*y, y*x*x, y*z*z, y*y*x, x*x*x, x*z*z, y*y*z, x*x*z, z*z*z])

def normalize_target(rpc, target):
    """
    Normalize in image space
    """
    target_norm = np.vstack(((target[:, 0] - rpc.col_offset) / rpc.col_scale,
                             (target[:, 1] - rpc.row_offset) / rpc.row_scale)).T
    return target_norm

def normalize_input_locs(rpc, input_locs):
    """
    Normalize in world space
    """
    input_locs_norm = np.vstack(((input_locs[:, 0] - rpc.lon_offset) / rpc.lon_scale,
                                 (input_locs[:, 1] - rpc.lat_offset) / rpc.lat_scale,
                                 (input_locs[:, 2] - rpc.alt_offset) / rpc.alt_scale)).T
    return input_locs_norm    

def update_rpc(rpc, x):
    """
    Update rpc coefficients
    """
    rpc.row_num, rpc.row_den = x[:20], x[20:40]
    rpc.col_num, rpc.col_den = x[40:60], x[60:]
    return rpc

def calculate_RMSE_row_col(rpc, input_locs, target):
    """
    Calculate MSE & RMSE in image domain
    """
    col_pred, row_pred = rpc.projection(lon=input_locs[:,0], lat=input_locs[:,1], alt=input_locs[:,2])
    MSE_col, MSE_row = np.mean((np.hstack([col_pred.reshape(-1, 1), row_pred.reshape(-1, 1)]) - target) ** 2, axis=0)
    MSE_row_col = np.mean([MSE_col, MSE_row]) # the number of data is equal in MSE_col and MSE_row
    RMSE_row_col = np.sqrt(MSE_row_col)
    return RMSE_row_col

def weighted_lsq(rpc_to_calibrate, target, input_locs, h=1e-3, tol=1e-2, max_iter=20):
    """
    Regularized iterative weighted least squares for calibrating rpc.
    
    Args: 
        max_iter : maximum number of iterations
        h : regularization parameter
        tol : tolerance criterion on improvment of RMSE over iterations
        
    Warning: this code is to be employed with the rpc_model defined in s2p
    """
    reg_matrix = (h ** 2) * np.eye(39)  # regularization matrix
    target_norm = normalize_target(rpc_to_calibrate, target)  # col, row
    input_locs_norm = normalize_input_locs(rpc_to_calibrate, input_locs)  # lon, lat, alt

    # define C, R and M
    C, R = target_norm[:, 0][:, np.newaxis], target_norm[:, 1][:, np.newaxis]
    lon, lat, alt = input_locs_norm[:,0], input_locs_norm[:,1], input_locs_norm[:,2]
    col, row = target_norm[:,0][:, np.newaxis], target_norm[:,1][:, np.newaxis]
    MC = np.hstack([np.ones((lon.shape[0], 1)), poly_vect(x=lat, y=lon, z=alt).T, -C * poly_vect(x=lat, y=lon, z=alt).T])
    MR = np.hstack([np.ones((lon.shape[0], 1)), poly_vect(x=lat, y=lon, z=alt).T, -R * poly_vect(x=lat, y=lon, z=alt).T])
       
    # calculate direct solution
    JR = np.linalg.inv(MR.T @ MR) @ (MR.T @ R)
    JC = np.linalg.inv(MC.T @ MC) @ (MC.T @ C)

    # update rpc and get error
    coefs = np.vstack([JR[:20], 1, JR[20:], JC[:20], 1, JC[20:]]).reshape(-1)
    rpc_to_calibrate = update_rpc(rpc_to_calibrate, coefs)
    RMSE_row_col = calculate_RMSE_row_col(rpc_to_calibrate, input_locs, target)
    
    for n_iter in range(1, max_iter+1):
        WR2 = np.diagflat(1 / ((MR[:, :20] @ coefs[20:40]) ** 2))  # diagonal matrix with 1 / denom ** 2
        JR_iter = np.linalg.inv((MR.T @ WR2 @ MR) + reg_matrix) @ (MR.T @ WR2 @ R)
        WC2 = np.diagflat(1 / ((MC[:, :20] @ coefs[60:80]) ** 2))  # diagonal matrix with 1 / denom ** 2
        JC_iter = np.linalg.inv((MC.T @ WC2 @ MC) + reg_matrix) @ (MC.T @ WC2 @ C)

        # update rpc and get error
        coefs = np.vstack([JR_iter[:20], 1, JR_iter[20:], JC_iter[:20], 1, JC_iter[20:]]).reshape(-1)
        rpc_to_calibrate = update_rpc(rpc_to_calibrate, coefs)
        RMSE_row_col_prev = RMSE_row_col
        RMSE_row_col = calculate_RMSE_row_col(rpc_to_calibrate, input_locs, target)

        # check convergence
        if np.abs(RMSE_row_col_prev - RMSE_row_col) < tol:
            break
    
    return rpc_to_calibrate


def define_grid3d_from_cloud(input_ecef, n_samples=10, margin=500, verbose=False):

    # define 3D grid to be fitted
    x, y, z = input_ecef[:,0], input_ecef[:,1], input_ecef[:,2]
    x_grid_coords = np.linspace(np.percentile(x, 5)-margin, np.percentile(x, 95)+margin, n_samples)
    y_grid_coords = np.linspace(np.percentile(y, 5)-margin, np.percentile(y, 95)+margin, n_samples)
    z_grid_coords = np.linspace(np.percentile(z, 5)-margin, np.percentile(z, 95)+margin, n_samples)
    x_grid, y_grid, z_grid = np.meshgrid(x_grid_coords, y_grid_coords, z_grid_coords)
    samples = np.vstack((x_grid.ravel(), y_grid.ravel(), z_grid.ravel())).T
    lat, lon, alt = ba_utils.ecef_to_latlon_custom(samples[:, 0], samples[:, 1], samples[:, 2])
    input_locs = np.vstack((lon, lat, alt)).T # lon, lat, alt

    if verbose:
        print('- {} 3D points to be used. '.format(input_locs.shape[0]))
        print('- Limits of the 3D space to fit:')
        print('         min lat: {:.4f}, max lat: {:.4f}'.format(min(lat), max(lat)))
        print('         min lon: {:.4f}, max lon: {:.4f}'.format(min(lon), max(lon)))
        print('         min alt: {:.4f}, max alt: {:.4f}\n'.format(min(alt), max(alt)))

        mymap = vistools.clickablemap(zoom=12)
        ## set the coordinates of the area of interest as a GeoJSON polygon
        aoi = {'coordinates': [[[min(lon), min(lat)], [min(lon), max(lat)],
                                [max(lon), max(lat)], [max(lon), min(lat)],
                                [min(lon), min(lat)]]], 'type': 'Polygon'}
        # set the center of the aoi
        aoi['center'] = np.mean(aoi['coordinates'][0][:4], axis=0).tolist()
        # display a polygon covering the aoi and center the map
        mymap.add_GeoJSON(aoi)
        mymap.center = aoi['center'][::-1]
        mymap.zoom = 14
        display(mymap)

    return input_locs


def scaling_params(vect):
    '''
    returns scale, offset based
    on vect min and max values
    '''
    min_vect = min(vect)
    max_vect = max(vect)
    scale = (max_vect - min_vect)/2
    offset = min_vect + scale
    return scale, offset


def initialize_rpc(target, input_locs):
    '''
    Creates an empty rpc instance
    '''
    d = {}
    listkeys = ['LINE_OFF', 'SAMP_OFF', 'LAT_OFF', 'LONG_OFF', 'HEIGHT_OFF',
                'LINE_SCALE', 'SAMP_SCALE', 'LAT_SCALE', 'LONG_SCALE', 'HEIGHT_SCALE',
                'LINE_NUM_COEFF', 'LINE_DEN_COEFF', 'SAMP_NUM_COEFF', 'SAMP_DEN_COEFF']
    for key in listkeys:
        d[key] = '0'

    rpc_init = rpcm.RPCModel(d)
    rpc_init.row_scale, rpc_init.row_offset = scaling_params(target[:, 1])
    rpc_init.col_scale, rpc_init.col_offset = scaling_params(target[:, 0])
    rpc_init.lat_scale, rpc_init.lat_offset = scaling_params(input_locs[:, 1])
    rpc_init.lon_scale, rpc_init.lon_offset = scaling_params(input_locs[:, 0])
    rpc_init.alt_scale, rpc_init.alt_offset = scaling_params(input_locs[:, 2])

    return rpc_init

def fit_rpc_from_projection_matrix(P, input_ecef, verbose=False):
    '''
    Fit an rpc from a set of 2d-3d correspondences given by a projection matrix
    
    Args:
        P : projection matrix that will be emulated with the calibrated rpc
        locs_3d : a set of points within the 3d world space that the rpc will fit - in ECEF coordinates
        verbose : displays map with the area covered by the 3d space to fit + shows the lat-lon-alt limits of such space
    '''

    input_locs = define_grid3d_from_cloud(input_ecef)
    x, y, z = ba_utils.latlon_to_ecef_custom(input_locs[:, 1], input_locs[:, 0], input_locs[:, 2])
    target = camera_utils.apply_projection_matrix(P, np.vstack([x, y, z]).T)
    rpc_init = initialize_rpc(target, input_locs)
    
    rpc_calib = weighted_lsq(rpc_init, target, input_locs)
    rmse_err = calculate_RMSE_row_col(rpc_calib, input_locs, target)

    # check the histogram of errors
    if verbose:
        check_errors(rpc_calib, input_locs, target)

    return rpc_calib, rmse_err


def fit_Rcorrected_rpc(euler_vec, original_rpc, input_ecef, verbose=False):
    '''
    Fit an rpc from a set of 2d-3d correspondences given by a projection matrix

    Args:
        P : projection matrix that will be emulated with the calibrated rpc
        locs_3d : a set of points within the 3d world space that the rpc will fit - in ECEF coordinates
        verbose : displays map with the area covered by the 3d space to fit + shows the lat-lon-alt limits of such space
    '''

    input_locs = define_grid3d_from_cloud(input_ecef)
    x, y, z = ba_utils.latlon_to_ecef_custom(input_locs[:, 1], input_locs[:, 0], input_locs[:, 2])
    pts_3d_adj = ba_core.rotate_euler(np.vstack([x, y, z]).T, euler_vec)
    target = camera_utils.apply_rpc_projection(original_rpc, pts_3d_adj)
    rpc_init = initialize_rpc(target, input_locs)

    rpc_calib = weighted_lsq(rpc_init, target, input_locs)
    rmse_err = calculate_RMSE_row_col(rpc_calib, input_locs, target)

    # check the histogram of errors
    if verbose:
        check_errors(rpc_calib, input_locs, target)

    return rpc_calib, rmse_err


def check_errors(rpc_calib, input_locs, target):
    lat, lon, alt = input_locs[:,1], input_locs[:,0], input_locs[:,2]
    col_pred, row_pred = rpc_calib.projection(lon, lat, alt)
    err = np.linalg.norm(np.hstack([col_pred.reshape(-1, 1), row_pred.reshape(-1, 1)]) - target, axis=1)
    plt.figure()
    plt.hist(err, bins=30)
    plt.show()
    return

def sample_direct(x_min, x_max, y_min, y_max, z_mean, grid_size = (15, 15, 15)):
    """
    Sample regularly spaced points over pixels of the image
    """
    x_grid_coords = np.linspace(x_min, x_max, grid_size[0])
    y_grid_coords = np.linspace(y_min, y_max, grid_size[1])
    z_grid_coords = np.linspace(0, 1.5*z_mean, grid_size[2])
    x_grid, y_grid, z_grid = np.meshgrid(x_grid_coords, y_grid_coords, z_grid_coords)
    samples = np.zeros((x_grid.size, 3), dtype = np.float32)
    samples[:, 0] = x_grid.ravel()
    samples[:, 1] = y_grid.ravel()
    samples[:, 2] = z_grid.ravel()
    return samples

def localize_target_to_input(rpc, samples):
    """
    Applies localization function over samples
    Returns : (lon, lat, alt)
    """
    input_locations = np.zeros_like(samples)
    input_locations[:, 2] = samples[:, 2]  # copy altitude
    for i in range(samples.shape[0]):
        #print(samples[i])
        input_locations[i, 0:2] = rpc.localization(*tuple(samples[i]))  # col, row, alt
    return input_locations  # lon, lat, alt

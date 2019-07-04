import numpy as np
import vistools
import ba_utils

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
    target_norm = np.vstack(((target[:, 0] - rpc.colOff) / rpc.colScale,
                             (target[:, 1] - rpc.linOff) / rpc.linScale)).T
    return target_norm

def normalize_input_locs(rpc, input_locs):
    """
    Normalize in world space
    """
    input_locs_norm = np.vstack(((input_locs[:, 0] - rpc.lonOff) / rpc.lonScale,
                                 (input_locs[:, 1] - rpc.latOff) / rpc.latScale,
                                 (input_locs[:, 2] - rpc.altOff) / rpc.altScale)).T
    return input_locs_norm    

def update_rpc(rpc, x):
    """
    Update rpc coefficients
    """
    rpc.inverseLinNum, rpc.inverseLinDen = x[:20], x[20:40]
    rpc.inverseColNum, rpc.inverseColDen = x[40:60], x[60:]
    rpc.directLatNum, rpc.directLatDen = x[:20], x[20:40]
    rpc.directLonNum, rpc.directLonDen = x[40:60], x[60:]
    return rpc

def calculate_RMSE_row_col(rpc, input_locs, target):
    """
    Calculate MSE & RMSE in image domain
    """
    col_pred, row_pred, _ = rpc.inverse_estimate(lon=input_locs[:,0], lat=input_locs[:,1], alt=input_locs[:,2])
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

def fit_rpc_from_projection_matrix(rpc_init, input_P, input_im, input_ecef, n_samples=10, margin=500, verbose=False):
    '''
    Fit an rpc from a set of 2d-3d correspondences given by a projection matrix
    
    Args:
        rpc_init : inital values of the rpc to be fitted (set nan if unknown)
        input_P : projection matrix that will be emulated with the calibrated rpc
        locs_3d : a set of points within the 3d world space that the rpc will fit - in ECEF coordinates 
        n_samples : the 3D space to be fit will be a grid of n_samples x n_samples x n_samples
        margin : extra meters added to the limits of locs_3d_ecef to ensure that the 3d space to fit covers the whole aoi
        verbose : displays map with the area covered by the 3d space to fit + shows the lat-lon-alt limits of such space
    '''
    
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
   
    rows, cols = input_im.shape
    rpc_init.linOff = float(rows)/2
    rpc_init.colOff = float(cols)/2
    rpc_init.latOff = min(lat) + (max(lat) - min(lat))/2
    rpc_init.lonOff = min(lon) + (max(lon) - min(lon))/2
    rpc_init.altOff = min(alt) + (max(alt) - min(alt))/2
    rpc_init.linScale = float(rows)/2
    rpc_init.colScale = float(cols)/2
    rpc_init.latScale = (max(lat) - min(lat))/2
    rpc_init.lonScale = (max(lon) - min(lon))/2
    rpc_init.altScale = (max(alt) - min(alt))/2
    
    proj = input_P @ np.hstack((samples, np.ones((samples.shape[0],1)))).T
    target = (proj[:2,:]/proj[-1,:]).T
    
    rpc_calib = weighted_lsq(rpc_init, target, input_locs)
    rmse_err = calculate_RMSE_row_col(rpc_calib, input_locs, target)
    
    return rpc_calib, rmse_err
    
    
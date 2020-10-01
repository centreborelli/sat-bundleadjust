import numpy as np
from scipy.linalg import cholesky
from scipy.sparse import lil_matrix
from ba_utils import rotate_euler, euler_angles_from_R, euler_angles_to_R


def ortographic_pose_estimation(corresp, calM):
    '''
    ORTHOGRAPHICPOSEESTIMATION Pose estimation by the Scaled-Orthographic model

    Computation of the orientation of M perspective cameras from N 
    corresponding tracks of image points and their internal paramaters.
    The method used is the SCALED ORTHOGRAPHIC model as described in the IPOL
    submission "The Orthographic Projection Model for Pose Calibration of Long 
    Focal Images" by L. F. Julia, P. Monasse, M. Pierrot-Deseilligny and it
    is based on the factorization method by C. J. Poelman and T. Kanade.

    Input arguments:
    Corresp  - 2MxN matrix containing in each column, the M projections of
         the same space point onto the M images.
    CalM     - 3Mx3 matrix containing the M calibration 3x3 matrices for 
         each camera concatenated.

    Output arguments: the two possible solutions Sol1, Sol2, in format of
            1x5-cell. Sol={Rot,Trans,Reconst, R, T} where:

    Rot      - 3Mx3 matrix containing the M 3x3 rotation matrices for each 
         camera concatenated. The first will always be the identity.
    Trans    - 3Mx1 vector containing the M translation vectors for each
         camera concatenated.
    Reconst  - 3xN matrix containing the 3D reconstruction of the
         correspondences.
    R, T     - 2Mx3 motion matrix and 2Mx1 translation matrix in terms of
         the orthographic model
    '''

    N = int(corresp.shape[1])      # number of correspondences
    M = int(corresp.shape[0]/2)    # number of views

    if N<4 or M<3:
        print('At least 4 tracks and 3 views are needed for pose estimation.\n')

    # focal lenghts and principal points
    focalL = np.array([calM[3*np.arange(M),0]]).T
    ppalP  = np.array([calM[np.setdiff1d(np.arange(M*3), 3*np.arange(1,M+1)-1),2]]).T

    # center image points subtracting ppal point and compute mean
    W = corresp - np.tile(ppalP, (1, N))
    T = np.array([np.mean(W,1)]).T

    # subtract the mean to obtain W* 
    W_star = W - np.tile(T, (1, N))

    # compute svd of W*
    U, S, Vh = np.linalg.svd(W_star, full_matrices=False)
    S = np.diag(S)

    # impose rank deficiency and compute rank factorization
    R_aux = U[:,:3] @ (S[:3,:3]**(1/2))
    S_aux = (S[:3,:3]**(1/2)) @ Vh[:3,:]

    # to find QQ' s.t. R=R_aux*Q and S=inv(Q)*S_aux we solve the homogeneous linear system M*coef(QQ')=0
    systM = np.zeros((2*M,6))
    for i in range(M):
        m = np.array([R_aux[2*i,:]])
        n = np.array([R_aux[2*i+1,:]])
        # norm constraints
        A = 2*(m.T*m - n.T*n).ravel()
        systM[2*i,:] = np.hstack(( (1./2)*A[np.array([0,4,8])], A[np.array([1,2,5])] ))
        # perpendicularity constraints
        A = (m.T*n+n.T*m).ravel()
        systM[2*i+1,:] = np.hstack(( (1./2)*A[np.array([0,4,8])], A[np.array([1,2,5])] ))

    # solution to the system
    _, _, Vh = np.linalg.svd(systM, full_matrices=False)
    #print(Vh.T)
    coefQQ = Vh[-1,:]*np.sign(Vh[-1,1]);
    #print(coefQQ)
    QQ = np.array([[coefQQ[0], coefQQ[3], coefQQ[4]],
                   [coefQQ[3], coefQQ[1], coefQQ[5]],
                   [coefQQ[4], coefQQ[5], coefQQ[2]]])
    Q = cholesky(QQ, lower=False)
    Q = Q.T
    # TODO: The Cholesky decomposition will fail if QQ is not positive semi-definite.
    #       We should check that before applying Cholesky

    # final rank decomposition
    R = R_aux @ Q
    S = np.linalg.inv(Q) @ S_aux

    #print(R_aux, S)

    # recover rotation parameters
    norms = np.sqrt(np.sum(R ** 2,1))
    Rot = np.zeros((3*M,3))
    Rot[np.setdiff1d(np.arange(M*3), 3*np.arange(1,M+1)-1),:] = R/(np.tile(norms, (3, 1)).T)
    Rot[3*np.arange(1,M+1)-1,:] = np.cross(Rot[3*np.arange(1,M+1)-3,:],Rot[3*np.arange(1,M+1)-2,:],axis=1)

    # recover translation parameters
    Trans = np.zeros((3*M,1))
    Trans[np.setdiff1d(np.arange(M*3),3*np.arange(1,M+1)-1)] = T
    Trans[3*np.arange(1,M+1)-1] = focalL
    s = np.reshape(np.tile(np.sum(np.reshape(norms, (M,2)).T,0)/2, (3,1)).T, (-1,1))
    Trans = Trans/s

    # depth ambiguity: second solution
    a = np.array([1,1,-1])
    Rot2 = np.diag(np.tile(a, (M,1)).ravel()) @ Rot @ np.diag(a)
    S2, R2 = np.diag(a) @ S, R @ np.diag(a)
    T2, Trans2 = T, Trans
    Reconstr, Reconstr2 = S, S2

    # translation and rotation to bring the center of the first camera to the world origin
    Reconstr  = Rot[:3,:] @ Reconstr + np.tile(Trans[:3],(1,N))
    Reconstr2 = Rot2[:3,:] @ Reconstr2 + np.tile(Trans2[:3], (1,N))
    R, R2 = R @ np.linalg.inv(Rot[:3,:]), R2 @ np.linalg.inv(Rot2[:3,:])
    T, T2 = T - R @ Trans[:3], T2 - R2 @ Trans2[:3]

    Rot = Rot @ np.linalg.inv(Rot[:3,:])
    Trans = Trans - Rot @ Trans[:3]
    Rot2 = Rot2 @ np.linalg.inv(Rot2[:3,:])
    Trans2 = Trans2 - Rot2 @ Trans2[:3]

    # scaling so that distance from the first camera to the second is 1
    alpha, alpha2 = 1/np.linalg.norm(Trans[3:6]), 1/np.linalg.norm(Trans2[3:6])
    Trans, Trans2 = alpha*Trans, alpha2*Trans2 
    R, R2 = (1/alpha)*R, (1/alpha2)*R2
    Reconstr, Reconstr2 = alpha*Reconstr, alpha2*Reconstr2

    sol1 = {'Rot': Rot,  'Trans': Trans,  'Reconstr': Reconstr,  'R': R,  'T': T+ppalP}
    sol2 = {'Rot': Rot2, 'Trans': Trans2, 'Reconstr': Reconstr2, 'R': R2, 'T': T2+ppalP}

    return sol1, sol2

def my_randsample(n,k):
    y = []
    for i in range(-1,k-1):
        j = np.random.randint(0,n-i-1) # only n-i integers not picked yet
        # find j-th element in 1:n \setminus y
        l = 0
        while l<=i and y[l]<=j:
            l, j = l+1, j+1
        # insert j in vector y
        y = y[:l] + [j] + y[l:]
    return np.array(y).T

def ac_ransac_orthographic(Corresp,CalM,min_imsize,NFA_th=1,max_it=100):

    N = Corresp.shape[1] # number of tracks
    n_sample = 4         # minimal number of matching points for pose estimation
    d = 2                # dimension of the error (repr. error used)
    n_out = 2            # number of possible output orientations
    a = np.pi/min_imsize # probability of having error 1 for nul hypothesis

    k_inliers = n_sample+1 # max number of inliers found
    inliers = []           # set of inliers
    ransac_th = np.inf     # ransac threshold

    it, max_old = 0, max_it
    while it<max_it:
        it = it+1
        sample = my_randsample(N,n_sample)

        # compute orientations with Orthographic model from this sample
        # (if the function fails, we start a new iteration)
        try:
            Sol1, Sol2 = ortographic_pose_estimation(Corresp[:,sample],CalM)
        except:
            if max_it<2*max_old:
                max_it +=1
            continue
        
        # compute the residual error for both solutions and choose the one with lower repr. error
        R, T, err_min, Sol_it = Sol1['R'], Sol1['T'], np.inf, Sol1

        for j in range(2):
            ind=[np.arange(4),np.array([4,5]),np.array([0,1,4,5]),np.array([2,3]),np.arange(2,6),np.array([0,1])]
            err=np.zeros((1,N))
            for k in range(3):
                # (orthographic) 3d reconstruction from one pair of views
                p3D=np.linalg.pinv(R[ind[2*k],:]) @ (Corresp[ind[2*k],:]-np.tile(T[ind[2*k]],(1,N)));
                # reprojection to the remaining view and error
                error=np.sqrt(np.sum((R[ind[2*k+1],:] @ p3D+np.tile(T[ind[2*k+1]],(1,N)) - Corresp[ind[2*k+1],:]) ** 2,0));
                # take the max of the error
                err=np.max(np.vstack((err,error)),axis=0)

            if np.sum(err) < err_min:
                vec_errors = err.copy()
                err_min = np.sum(err)
                if j==1:
                    Sol_it = Sol2

            R, T = Sol2['R'], Sol2['T']

        # points not in the sample used
        nosample = np.setdiff1d(np.arange(N),sample)

        # sort the list of errors
        ind_sorted = np.argsort(vec_errors[nosample])
        # search for minimum of NFA(model,k)
        NFA_min, k_min, err_threshold = NFA_th, 0, np.inf
        factor = n_out*np.prod(np.arange(N-n_sample,N))/np.math.factorial(n_sample)
        for k in np.arange(n_sample+1,N):
            factor=factor*( (N-(k-1))/(k-n_sample) )*a
            NFA=factor*(vec_errors[nosample[ind_sorted[k-n_sample-1]]]) ** (d*(k-n_sample))
            if NFA<=NFA_min:
                NFA_min, k_min = NFA, k
                err_threshold = vec_errors[nosample[ind_sorted[k-n_sample-1]]]

        # If the found model has more inliers or the same number with less error than the previous we keep it
        if k_min>k_inliers or (k_min==k_inliers and err_threshold<ransac_th):
            k_inliers=k_min
            inliers=np.hstack((np.reshape(sample,(1,-1)), np.reshape(nosample[ind_sorted[:(k_inliers-n_sample+1)]], (1,-1))))
            ransac_th=err_threshold
            Sol=Sol_it

    return inliers, Sol, ransac_th

########################################### until here same code from https://github.com/LauraFJulia/OrthographicPE


########################################### new code to complete the BA pipeline using the ortographic-scaled model

def define_ba_params(ortographic_sol, corresp, calM, opt_X=True, opt_R=True, opt_T=False, opt_K=False):
    n_pts = corresp.shape[1]
    n_cam = int(corresp.shape[0]/2)
    
    # define camera_params as needed in bundle adjustment
    cam_params = np.zeros((n_cam,9))
    for i in range(n_cam):
        R = ortographic_sol['Rot'][(3*i):(3*i)+3,:]
        if not np.allclose(euler_angles_to_R(euler_angles_from_R(R)),R):
            u, s, vh = np.linalg.svd(R, full_matrices=False)
            R = u @ np.diag([1.,1.,1.]) @ vh
        t = ortographic_sol['Trans'][(3*i):(3*i)+3,:]
        focalL, ppalP = calM[3*i,0], calM[(3*i):(3*i)+2,2]
        cam_params[i,:] = np.hstack((euler_angles_from_R(R).ravel(), t.ravel(),focalL, ppalP))

    # define camera_ind, points_ind, points_2d as needed in bundle adjustment
    point_ind, camera_ind, points_2d = [], [], []
    for i in range(n_pts):
        for j in range(n_cam):
            point_ind.append(i)
            camera_ind.append(j)
            points_2d.append(corresp[(2*j):(2*j)+2,i].T)
    
    pts_ind, cam_ind, pts_2d = np.array(point_ind), np.array(camera_ind), np.vstack(points_2d)
    
    # other ba parameters
    ba_params = {
    'n_cam'   : n_cam,
    'n_pts'   : n_pts,
    'n_params': 0,
    'opt_X'   : opt_X,
    'opt_R'   : opt_R,
    'opt_T'   : opt_T,
    'opt_K'   : opt_K
    }

    n_params = 0
    if ba_params['opt_R']:
        n_params += 3
        cam_params_opt = cam_params[:,:3]
        if ba_params['opt_T']:
            n_param += 3
            cam_params_opt = np.hstack((cam_params_opt, cam_params[:,3:6]))  # REVIEW !!!
            if ba_params['opt_K']:
                n_params += 3
                cam_params_opt = np.hstack((cam_params_opt, cam_params[:,6:]))   # REVIEW !!!
    ba_params['n_params'] = n_params    
    
    pts_3d = ortographic_sol['Reconstr'].T
    
    return cam_params_opt, cam_params, pts_2d, cam_ind, pts_ind, ba_params, pts_3d   
            
def project(pts, cam_params):
    """
    Convert 3D points to 2D by projecting onto images
    """
    f, cx, cy = np.array([cam_params[:, 6]]), np.array([cam_params[:, 7]]), np.array([cam_params[:, 8]])
    pts_proj = rotate_euler(pts, cam_params[:, :3])
    pts_proj += cam_params[:, 3:6]
    # up until here we have completed [ R | t ] @ x
    # now we apply K 
    pts_proj[:,0] = f * pts_proj[:,0] + cx * pts_proj[:,2]
    pts_proj[:,1] = f * pts_proj[:,1] + cy * pts_proj[:,2]
    pts_proj = pts_proj[:, :2] / pts_proj[:, 2, np.newaxis] # set scale = 1
    return pts_proj

def fun(params, cam_ind, pts_ind, pts_2d, cam_params_init, pts_3d_init, ba_params):
    """
    Compute BA residuals.
    `params` contains those parameters to be optimized (3D points and camera paramters)
    """
    n_cam, n_pts, n_params = ba_params['n_cam'], ba_params['n_pts'], ba_params['n_params']

    # get pts_3d from 'params' to optimize (or fix them to the input locations)
    if ba_params['opt_X']:
        pts_3d = params[n_cam * n_params:].reshape((n_pts, 3))
    else:
        pts_3d = pts_3d_init.copy()
     
    # get camera params to optimize and camera params to be fixed
    cam_params_opt = params[:n_cam * n_params].reshape((n_cam, n_params)) # camera paramters to be optimized
    if n_params > 0:
        camera_params = np.hstack((cam_params_opt,cam_params_init[:,n_params:]))
    else:
        camera_params = cam_params_init.copy()
    
    # project 3d points using the current camera params
    points_proj = project(pts_3d[pts_ind], camera_params[cam_ind])
    
    # compute reprojection errors
    err = (points_proj - pts_2d).ravel()
    return err

def bundle_adjustment_sparsity(cam_ind, pts_ind, ba_params):
    n_cam, n_pts, n_params = ba_params['n_cam'], ba_params['n_pts'], ba_params['n_params']
    m = cam_ind.size * 2
    #n = n_cam * n_params + n_pts * 3
    
    if ba_params['opt_X']:
        n = n_cam * n_params + n_pts * 3
    else:
        n = n_cam * n_params
    
    A = lil_matrix((m, n), dtype=int)
    #print('m:', m, 'n:', n, 'n_cam:', n_cam)
    
    i = np.arange(cam_ind.size)
    for s in range(n_params):
        A[2 * i, cam_ind * n_params + s] = 1
        A[2 * i + 1, cam_ind * n_params + s] = 1
        
    if ba_params['opt_X']:
        for s in range(3):
            A[2 * i, n_cam * n_params + pts_ind * 3 + s] = 1
            A[2 * i + 1, n_cam * n_params + pts_ind * 3 + s] = 1
            
    return A

def recover_ba_output(optimized_params, ba_params, cam_params_init, pts_3d_init):
    
    n_cam, n_pts, n_params = ba_params['n_cam'], ba_params['n_pts'], ba_params['n_params']
    # remember the structure of optimized_params = np.hstack((camera_params.ravel(), points_3d.ravel()))

    # recover the bundle adjusted 3d points
    if ba_params['opt_X']:
        pts_3d_ba = optimized_params[n_cam * n_params:].reshape((n_pts, 3))
    else:
        pts_3d_ba = pts_3d_init

    # recover the bundle adjusted camera matrices
    cam_params_opt = optimized_params[:n_cam * n_params].reshape((n_cam, n_params))

    if n_params > 0:
        cam_params_ba = np.hstack((cam_params_opt,cam_params_init[:,n_params:]))
    else:
        cam_params_ba = cam_params_init

    P_crop_ba = []
    for i in range(n_cam):
        P_crop_ba.append(ba_cam_params_to_P(cam_params_ba[i,:]))
        
    return pts_3d_ba, cam_params_ba, P_crop_ba

def ba_cam_params_to_P(cam_params):
    vecR, vecT, f, cx, cy = cam_params[0:3], cam_params[3:6], cam_params[6], cam_params[7], cam_params[8]
    K = np.array([[f, 0., cx], [0., f, cy], [0., 0., 1.]])
    R = euler_angles_to_R(vecR)
    P = K @ np.hstack((R, vecT.reshape((3,1))))
    return P

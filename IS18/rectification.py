"""
Stereo rectification tools
Copyright (C) 2018, Gabriele Facciolo <facciolo@cmla.ens-cachan.fr>
Copyright (C) 2018, Carlo de Franchis <carlo.de-franchis@ens-cachan.fr>
"""

from __future__ import print_function
from scipy import ndimage
import numpy as np
import cv2
import ad
import srtm4

from IS18 import utils

def get_angle_from_cos_and_sin(c, s):
    """
    Computes x in ]-pi, pi] such that cos(x) = c and sin(x) = s.
    """
    if s >= 0:
        return np.arccos(c)
    else:
        return -np.arccos(c)


def matrix_translation(x, y):
    """
    Return the (3, 3) matrix representing a 2D shift in homogeneous coordinates.
    """
    t = np.eye(3)
    t[0, 2] = x
    t[1, 2] = y
    return t


def projection_of_3d_vertical_line(rpc, lon, lat, h_min=-200, h_max=3000, h_step=10):
    """
    Sample the projection of a 3d vertical line on the image plane.

    Args:
        rpc: instance of the rpc_model.RPCModel class
        lon, lat: geographic coordinates of the ground point through which the vertical line passes
        h_min, h_max: min, max altitude bounds of the vertical line
        h_step: step used to sample the vertical line

    Return:
        list of points in the image plane given by their pixel coordinates
    """
    return [rpc.projection(lon, lat, h) for h in np.arange(h_min, h_max, h_step)]


def epipolar_curve(rpc1, rpc2, x, y, h_min=-200, h_max=3000, h_step=10):
    """
    Sample the epipolar curve of image 2 associated to point (x, y) of image 1.

    Args:
        rpc1, rpc2: instances of the rpc_model.RPCModel class
        x, y: pixel coordinates of a point in the first image (ie associated to rpc1)
        h_min, h_max: min, max altitudes defining the bounds of the epipolar curve
        h_step: step used to sample the epipolar curve

    Return:
        list of points in the second image given by their pixel coordinates
    """
    return [rpc2.projection(*rpc1.localization(x, y, h), h) for h in
            np.arange(h_min, h_max, h_step)]


def trace_epipolar_curve(image1, image2, aoi, x0, y0):
    """auxiliary function to display in image2 the epipolar curve 
        corresponding to the point x0,y0 in the cropped image1"""
    
    import matplotlib.pyplot as plt

    # get the altitude of the center of the AOI
    lon, lat = aoi['center']
    z = srtm4.srtm4(lon, lat)

    # read the RPC coefficients of images i and j
    rpc1 = utils.rpc_from_geotiff(image1)
    rpc2 = utils.rpc_from_geotiff(image2)

    # crop the two images
    im1, x1, y1 = utils.crop_aoi(image1, aoi, z)
    im2, x2, y2 = utils.crop_aoi(image2, aoi, z)

    # translation matrices needed to compensate the crop offset
    H1 = matrix_translation(x1, y1)
    H2 = matrix_translation(x2, y2)

    # select a point in the first image
    #x0, y0 = 200, 200

    # compensate the crop offset of the first image
    x, y = np.dot(H1, [x0, y0, 1])[:2]

    # compute the epipolar curve
    epi = epipolar_curve(rpc1, rpc2, x, y)

    # compensate for the crop offset of the second image
    p = np.array([np.dot(np.linalg.inv(H2), [x, y, 1])[:2] for x, y in epi])

    # plot the epipolar curve on the second image
    f, ax = plt.subplots(1, 2, figsize=(13,10))
    ax[0].plot(x0, y0, 'r+')
    ax[1].plot(p[:, 0], p[:, 1], 'r-')
    ax[0].imshow(np.sqrt(im1.squeeze()), cmap='gray')
    ax[1].imshow(np.sqrt(im2.squeeze()), cmap='gray')


def rpc_affine_approximation(rpc, p):
    """
    Compute the first order Taylor approximation of an RPC projection function.

    Args:
        rpc: instance of the rpc_model.RPCModel class
        p: lon, lat, h coordinates

    Return:
        array of shape (3, 4) representing the affine camera matrix equal to the
        first order Taylor approximation of the RPC projection function at point p.
    """
    p = ad.adnumber(p)
    q = rpc.projection(*p)
    J = ad.jacobian(q, p)

    A = np.zeros((3, 4))
    A[:2, :3] = J
    A[:2, 3] = np.array(q) - np.dot(J, p)
    A[2, 3] = 1
    return A



def affine_fundamental_matrix(p, q):
    """
    Compute the affine fundamental matrix from two affine camera matrices.

    Args:
        p, q: arrays of shape (3, 4) representing the input camera matrices.

    Return:
        array of shape (3, 3) representing the affine fundamental matrix computed
        with the formula 17.3 (p. 412) from Hartley & Zisserman book (2nd ed.).
    """
    X0 = p[[1, 2], :]
    X1 = p[[2, 0], :]
    X2 = p[[0, 1], :]
    Y0 = q[[1, 2], :]
    Y1 = q[[2, 0], :]
    Y2 = q[[0, 1], :]

    F = np.zeros((3, 3))
    F[0, 2] = np.linalg.det(np.vstack([X2, Y0]))
    F[1, 2] = np.linalg.det(np.vstack([X2, Y1]))
    F[2, 0] = np.linalg.det(np.vstack([X0, Y2]))
    F[2, 1] = np.linalg.det(np.vstack([X1, Y2]))
    F[2, 2] = np.linalg.det(np.vstack([X2, Y2]))

    return F


def rectifying_similarities_from_affine_fundamental_matrix(F, debug=False):
    """
    Computes two similarities from an affine fundamental matrix.

    Args:
        F: 3x3 numpy array representing the input fundamental matrix
        debug (optional, default is False): boolean flag to activate verbose
            mode

    Returns:
        S, S': two similarities such that, when used to resample the two images
            related by the fundamental matrix, the resampled images are
            stereo-rectified.
    """
    # check that the input matrix is an affine fundamental matrix
    assert(np.shape(F) == (3, 3))
    assert(np.linalg.matrix_rank(F) == 2)
    np.testing.assert_allclose(F[:2, :2], np.zeros((2, 2)))

    # notations
    a = F[2, 0]
    b = F[2, 1]
    c = F[0, 2]
    d = F[1, 2]
    e = F[2, 2]

    # rotations
    r = np.sqrt(a*a + b*b)
    s = np.sqrt(c*c + d*d)
    R1 = (1.0 / r) * np.array([[b, -a], [a, b]])
    R2 = (1.0 / s) * np.array([[-d, c], [-c, -d]])

    # zoom and translation
    z = np.sqrt(r / s)
    t = 0.5 * e / np.sqrt(r * s)

    if debug:
        theta_1 = get_angle_from_cos_and_sin(b / r, a / r)
        print("reference image:")
        print("\trotation: %f deg" % np.rad2deg(theta_1))
        print("\tzoom: %f" % z)
        print("\tvertical translation: %f" % t)
        print()
        theta_2 = get_angle_from_cos_and_sin(-d / s, -c / s)
        print("secondary image:")
        print("\trotation: %f deg" % np.rad2deg(theta_2))
        print("\tzoom: %f" % (1.0 / z))
        print("\tvertical translation: %f" % -t)

    # output similarities
    S1 = np.zeros((3, 3))
    S1[0:2, 0:2] = z * R1
    S1[1, 2] = t
    S1[2, 2] = 1

    S2 = np.zeros((3, 3))
    S2[0:2, 0:2] = (1.0 / z) * R2
    S2[1, 2] = -t
    S2[2, 2] = 1

    return S1, S2


def affine_transformation(x, xx):
    """
    Estimate a 2D affine transformation from a list of point matches.

    Args:
        x:  Nx2 numpy array, containing a list of points
        xx: Nx2 numpy array, containing the list of corresponding points

    Returns:
        3x3 numpy array, representing in homogeneous coordinates an affine
        transformation that maps the points of x onto the points of xx.

    This function implements the Gold-Standard algorithm for estimating an
    affine homography, described in Hartley & Zisserman page 130 (second
    edition).
    """
    # check that there are at least 3 points
    if len(x) < 3:
        print("ERROR: affine_transformation needs at least 3 matches")
        return np.eye(3)

    # translate the input points so that the centroid is at the origin.
    t = -np.mean(x,  0)
    tt = -np.mean(xx, 0)
    x = x + t
    xx = xx + tt

    # compute the Nx4 matrix A
    A = np.hstack((x, xx))

    # two singular vectors corresponding to the two largest singular values of
    # matrix A. See Hartley and Zissermann for details.  These are the first
    # two lines of matrix V (because np.linalg.svd returns V^T)
    U, S, V = np.linalg.svd(A)
    v1 = V[0, :]
    v2 = V[1, :]

    # compute blocks B and C, then H
    tmp = np.vstack((v1, v2)).T
    assert(np.shape(tmp) == (4, 2))
    B = tmp[0:2, :]
    C = tmp[2:4, :]
    H = np.dot(C, np.linalg.inv(B))

    # return A
    A = np.eye(3)
    A[0:2, 0:2] = H
    A[0:2, 2] = np.dot(H, t) - tt
    return A


def rectifying_affine_transforms(rpc1, rpc2, aoi, z=0):
    """
    Compute two affine transforms that rectify two images over a given AOI.

    Args:
        rpc1, rpc2 (rpc_model.RPCModel): two RPC camera models
        aoi (geojson.Polygon): area of interest

    Return:
        S1, S2 (2D arrays): two numpy arrays of shapes (3, 3) representing the
            rectifying affine transforms in homogeneous coordinates
        w, h (ints): minimal width and height of the rectified image crops
            needed to cover the AOI
        P1, P2 (2D arrays): two numpy arrays of shapes (3, 3) representing the
            affine camera matrices used to approximate the rpc camera models
    """
    # center of the AOI
    lons, lats = np.asarray(aoi['coordinates'][0][:4]).T
    lon, lat = np.mean([lons, lats], axis=1)

    # affine projection matrices that approximate the rpc models around the
    # center of the AOI
    P1 = rpc_affine_approximation(rpc1, (lon, lat, z))
    P2 = rpc_affine_approximation(rpc2, (lon, lat, z))

    # affine fundamental matrix associated to our two images
    F = affine_fundamental_matrix(P1, P2)

    # compute rectifying similarities
    S1, S2 = rectifying_similarities_from_affine_fundamental_matrix(F)

    # affine correction of S2 to register the ground (horizontal plane at z)
    q1 = S1 @ P1 @ [lons, lats, [z, z, z, z], [1, 1, 1, 1]]
    q2 = S2 @ P2 @ [lons, lats, [z, z, z, z], [1, 1, 1, 1]]
    S2 = affine_transformation(q2[:2].T, q1[:2].T) @ S2

    # shift the rectified images so that their top-left corners fall on (0, 0)
    x1, y1, w1, h1 = utils.bounding_box_of_projected_aoi(rpc1, aoi, z=z,
                                                         homography=S1)
    x2, y2, w2, h2 = utils.bounding_box_of_projected_aoi(rpc2, aoi, z=z,
                                                         homography=S2)
    S1 = matrix_translation(-x1, -0.5 * (y1 + y2)) @ S1
    S2 = matrix_translation(-x2, -0.5 * (y1 + y2)) @ S2

    w = int(round(max(w1, w2)))
    h = int(round(max(h1, h2)))
    return S1, S2, w, h, P1, P2


def match_pair(a, b):
    """
    Find SIFT matching points in two images represented as numpy arrays.

    Args:
        a, b (arrays): two numpy arrays containing the input images to match

    Return:
        pts1, pts2: two lists of pairs of coordinates of matching points
    """
    a = utils.simple_equalization_8bit(a)
    b = utils.simple_equalization_8bit(b)

    # KP
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(a, None)
    kp2, des2 = sift.detectAndCompute(b, None)
#    kp = sift.detect(a, None)
#    img = cv2.drawKeypoints(a, kp, b)
#    display_image(img)
#    cv2.imwrite('sift_keypoints.jpg', img)

    # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)


    # cv2.drawMatchesKnn expects list of lists as matches.
#    img3 = cv2.drawMatchesKnn(a,kp1,b,kp2,good,a,flags=2)

    pts1 = np.asarray(pts1)
    pts2 = np.asarray(pts2)
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)

    # We select only inlier points
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]

#   display_image(img3)
    return  pts1, pts2


def sift_roi(file1, file2, aoi, z):
    """
    Args:
        file1, file2: filename of two satellite images
        aoi: area of interest
        z: base height for the aoi

    Returns:
        q1, q2: numpy arrays with the coordinates of the matching points in the
            original (full-size) image domains
    """
    # image crops
    crop1, x1, y1 = utils.crop_aoi(file1, aoi, z=z)
    crop2, x2, y2 = utils.crop_aoi(file2, aoi, z=z)

    # sift keypoint matches
    p1, p2 = match_pair(crop1, crop2)
    q1 = utils.points_apply_homography(matrix_translation(x1, y1), p1)
    q2 = utils.points_apply_homography(matrix_translation(x2, y2), p2)
    return q1, q2


def affine_crop(input_path, A, w, h):
    """
    Apply an affine transform to an image.

    Args:
        input_path (string): path or url to the input image
        A (numpy array): 3x3 array representing an affine transform in
            homogeneous coordinates
        w, h (ints): width and height of the output image

    Return:
        numpy array of shape (h, w) containing a subset of the transformed
        image. The subset is the rectangle between points 0, 0 and w, h.
    """
    # determine the rectangle that we need to read in the input image
    output_rectangle = [[0, 0], [w, 0], [w, h], [0, h]]
    x, y, w0, h0 = utils.bounding_box2D(utils.points_apply_homography(np.linalg.inv(A),
                                                                      output_rectangle))
    x, y = np.floor((x, y)).astype(int)
    w0, h0 = np.ceil((w0, h0)).astype(int)

    # crop the needed rectangle in the input image
    with utils.rio_open(input_path, 'r') as src:
        aoi = src.read(indexes=1, window=((y, y + h0), (x, x + w0)))

    # compensate the affine transform for the crop
    B = A @ matrix_translation(x, y)

    # apply the affine transform
    out = ndimage.affine_transform(aoi.T, np.linalg.inv(B), output_shape=(w, h)).T
    return out


def rectify_aoi(file1, file2, aoi, z=None):
    """
    Args:
        file1, file2: filename of two satellite images
        aoi: area of interest
        z (float, optional): base altitude with respect to WGS84 ellipsoid. If
            None, z is retrieved from srtm.

    Returns:
        rect1, rect2: numpy arrays with the images
        S1, S2: transformation matrices from the coordinate system of the original images
        disp_min, disp_max: horizontal disparity range
        P1, P2: affine rpc approximations of the two images computed during the rectification
    """
    # read the RPC coefficients
    rpc1 = utils.rpc_from_geotiff(file1)
    rpc2 = utils.rpc_from_geotiff(file2)

    # get the altitude of the center of the AOI
    if z is None:
        lon, lat = np.mean(aoi['coordinates'][0][:4], axis=0)
        z = srtm4.srtm4(lon, lat)

    # compute rectifying affine transforms
    S1, S2, w, h, P1, P2 = rectifying_affine_transforms(rpc1, rpc2, aoi, z=z)

    # compute sift keypoint matches
    q1, q2 = sift_roi(file1, file2, aoi, z)

    # transform the matches to the domain of the rectified images
    q1 = utils.points_apply_homography(S1, q1)
    q2 = utils.points_apply_homography(S2, q2)

    # pointing correction (y_shift)
    y_shift = np.median(q2 - q1, axis=0)[1]
    S2 = matrix_translation(0, -y_shift) @ S2

    # rectify the crops
    rect1 = affine_crop(file1, S1, w, h)
    rect2 = affine_crop(file2, S2, w, h)

    # disparity range bounds
    kpts_disps = (q2 - q1)[:, 0]
    disp_min = np.percentile(kpts_disps, 2)
    disp_max = np.percentile(kpts_disps, 100 - 2)

    return rect1, rect2, S1, S2, disp_min, disp_max, P1, P2

import numpy as np

def euler_to_quaternion(roll, pitch, yaw):
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return [qx, qy, qz, qw]

def quaternion_to_euler(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    X = np.arctan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = np.arcsin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z = np.arctan2(t3, t4)
    return X, Y, Z

def quaternion_to_R(q0, q1, q2, q3):
    """Convert a quaternion into rotation matrix form.
    https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    """
    matrix = np.zeros((3,3))
    matrix[0, 0] = q0**2 + q1**2 - q2**2 - q3**2
    matrix[1, 1] = q0**2 - q1**2 + q2**2 - q3**2
    matrix[2, 2] = q0**2 - q1**2 - q2**2 + q3**2
    matrix[0, 1] = 2.0 * (q1*q2 - q0*q3)
    matrix[0, 2] = 2.0 * (q0*q2 + q1*q3)
    matrix[1, 2] = 2.0 * (q2*q3 - q0*q1)
    matrix[1, 0] = 2.0 * (q1*q2 + q0*q3)
    matrix[2, 0] = 2.0 * (q1*q3 - q0*q2)
    matrix[2, 1] = 2.0 * (q0*q1 + q2*q3)  
    return matrix

def rotate_euler(pts, vecR):
    """
    Rotate points by using Euler angles
    """
    # R = R(z)R(y)R(x)
    cosx, sinx = np.cos(vecR[:,0]), np.sin(vecR[:,0]) 
    cosy, siny = np.cos(vecR[:,1]), np.sin(vecR[:,1])
    cosz, sinz = np.cos(vecR[:,2]), np.sin(vecR[:,2])
    
    # rotate along x-axis
    pts_Rx = np.vstack((pts[:,0], cosx*pts[:,1]-sinx*pts[:,2], sinx*pts[:,1] + cosx*pts[:,2])).T
    
    # rotate along y-axis
    pts_Ryx = np.vstack((cosy*pts_Rx[:,0] + siny*pts_Rx[:,2], pts_Rx[:,1], -siny*pts_Rx[:,0] + cosy*pts_Rx[:,2])).T
    
    # rotate along z-axis
    pts_Rzyx = np.vstack((cosz*pts_Ryx[:,0] - sinz*pts_Ryx[:,1], sinz*pts_Ryx[:,0] + cosz*pts_Ryx[:,1], pts_Ryx[:,2])).T
    
    return pts_Rzyx

def euler_angles_from_R(R) :
    """
    Convert a 3x3 rotation matrix R to the Euler angles representation
    Source: https://www.learnopencv.com/rotation-matrix-to-euler-angles/
    """
    sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
 
    if not singular:
        xa, ya, za = np.arctan2(R[2,1] , R[2,2]), np.arctan2(-R[2,0], sy), np.arctan2(R[1,0], R[0,0])
    else:
        xa, ya, za = np.arctan2(-R[1,2], R[1,1]), np.arctan2(-R[2,0], sy), 0
 
    return np.array([xa, ya, za])

def euler_angles_to_R(vecR):
    """
    Recover the 3x3 rotation matrix R from the Euler angles representation
    Source: https://www.learnopencv.com/rotation-matrix-to-euler-angles/
    """
    
    R_x = np.array([[1, 0,                0              ],
                    [0, np.cos(vecR[0]), -np.sin(vecR[0])],
                    [0, np.sin(vecR[0]),  np.cos(vecR[0])]])
         
    R_y = np.array([[ np.cos(vecR[1]), 0, np.sin(vecR[1])],
                    [ 0,               1, 0              ],
                    [-np.sin(vecR[1]), 0, np.cos(vecR[1])]])
                 
    R_z = np.array([[np.cos(vecR[2]), -np.sin(vecR[2]), 0],
                    [np.sin(vecR[2]),  np.cos(vecR[2]), 0],
                    [0,                0,               1]])
                     
    R = R_z @ R_y @ R_x
    
    return R

def axis_angle_from_R(R):
    """
    Convert a 3x3 rotation matrix R to the axis-angle representation
    Source: https://github.com/Wallacoloo/printipi/blob/master/util/rotation_matrix.py
    """
    # Axis
    axis = np.array([ R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]])
    # Angle
    r, t = np.hypot(axis[0], np.hypot(axis[1], axis[2])), R[0,0] + R[1,1] + R[2,2]
    theta = np.arctan2(r, t-1)
    # Normalise axis
    axis = axis / r
    return axis, theta

def axis_angle_to_R(axis, angle):
    """
    Recover the 3x3 rotation matrix R from the axis-angle representation
    Source: https://github.com/Wallacoloo/printipi/blob/master/util/rotation_matrix.py
    """
    # Trig factors
    ca, sa = np.cos(angle), np.sin(angle)
    C = 1 - ca
    # Depack the axis
    x, y, z = axis
    # Multiplications (to remove duplicate calculations)
    xs, ys, zs, xC, yC, zC = x*sa, y*sa, z*sa, x*C, y*C, z*C
    xyC, yzC, zxC = x*yC, y*zC, z*xC
    # Update the rotation matrix
    R = np.array([[x*xC + ca, xyC - zs, zxC + ys], [xyC + zs, y*yC + ca, yzC - xs], [zxC - ys, yzC + xs, z*zC + ca]])
    return R

def get_proper_R(R):
    """
    Get proper rotation matrix (i.e. det(R) = 1) from improper rotation matrix (i.e. det(R) = -1)
    Notes on proper and improper rotations: https://pdfs.semanticscholar.org/9934/061eedc830fab32edd97cd677b95f21248e1.pdf
    """
    ind_of_reflection_axis = 99   # arbitrary value to indicate that R is a proper rotation matrix
    if (np.linalg.det(R) - (-1.0)) < 1e-5:
        L, V = np.linalg.eig(R)
        ind_of_reflection_axis = np.where(np.abs(L + 1.0) < 1e-5)[0][0]
        D = np.array([1,1,1])
        D[ind_of_reflection_axis] *= -1
        D = np.diag(D)
        R = np.real(V @ D @ np.diag(L) @ np.linalg.inv(V))
    return R, ind_of_reflection_axis

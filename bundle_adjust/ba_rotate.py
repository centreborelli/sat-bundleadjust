"""
Bundle Adjustment for 3D Reconstruction from Multi-Date Satellite Images
This script implements a series of functions for the representation of rotations in the 3d space
by Roger Mari <roger.mari@ens-paris-saclay.fr>
"""

import numpy as np


def euler_to_quaternion(roll, pitch, yaw):
    '''
    Converts euler angles (roll, pitch, yaw) to quaternion (qw, qx, qy, qz)
    '''
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return qw, qx, qy, qz


def quaternion_to_euler(qw, qx, qy, qz):
    '''
    Converts a quaternion (qw, qx, qy, qz) to euler angles (roll, pitch, yaw)
    '''
    t0 = +2.0 * (qw * qx + qy * qz)
    t1 = +1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(t0, t1)
    t2 = +2.0 * (qw * qy - qz * qx)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = np.arcsin(t2)
    t3 = +2.0 * (qw * qz + qx * qy)
    t4 = +1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(t3, t4)
    return roll, pitch, yaw


def quaternion_to_R(q0, q1, q2, q3):
    '''
    Converts a quaternion (q0, q1, q2, q3) into 3x3 rotation matrix (R)
    Note that a quaternion (qw, qx, qy, qz) may be equivalently noted as (q0, q1, q2, q3)
    Source: https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    '''
    R = np.zeros((3,3), dtype=np.float32)
    R[0, 0] = q0**2 + q1**2 - q2**2 - q3**2
    R[1, 1] = q0**2 - q1**2 + q2**2 - q3**2
    R[2, 2] = q0**2 - q1**2 - q2**2 + q3**2
    R[0, 1] = 2.0 * (q1 * q2 - q0 * q3)
    R[0, 2] = 2.0 * (q0 * q2 + q1 * q3)
    R[1, 2] = 2.0 * (q2 * q3 - q0 * q1)
    R[1, 0] = 2.0 * (q1 * q2 + q0 * q3)
    R[2, 0] = 2.0 * (q1 * q3 - q0 * q2)
    R[2, 1] = 2.0 * (q0 * q1 + q2 * q3)
    return R


def R_to_quaternion(R):
    '''
    Converts a 3x3 rotation matrix (R) into quaternion (q0, q1, q2, q3)
    '''
    return euler_to_quaternion(*euler_angles_from_R(R))


def euler_angles_from_R(R):
    """
    Converts a 3x3 rotation matrix (R) to euler angles representation
    Source: https://www.learnopencv.com/rotation-matrix-to-euler-angles/
    """
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        roll, pitch, yaw = np.arctan2(R[2, 1] , R[2, 2]), np.arctan2(-R[2, 0], sy), np.arctan2(R[1, 0], R[0, 0])
    else:
        roll, pitch, yaw = np.arctan2(-R[1, 2], R[1, 1]), np.arctan2(-R[2, 0], sy), 0
    return roll, pitch, yaw


def euler_angles_to_R(roll, pitch, yaw):
    """
    Recover the 3x3 rotation matrix R from the Euler angles representation
    Source: https://www.learnopencv.com/rotation-matrix-to-euler-angles/
    """
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll),  np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw),  np.cos(yaw), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def axis_angle_from_R(R):
    """
    Convert a 3x3 rotation matrix R to the axis-angle representation
    Source: https://github.com/Wallacoloo/printipi/blob/master/util/rotation_matrix.py
    """
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    r, t = np.hypot(axis[0], np.hypot(axis[1], axis[2])), R[0,0] + R[1,1] + R[2,2]
    theta = np.arctan2(r, t-1)
    axis = axis / r
    return axis, theta


def axis_angle_to_R(axis, angle):
    """
    Recover the 3x3 rotation matrix R from the axis-angle representation
    Source: https://github.com/Wallacoloo/printipi/blob/master/util/rotation_matrix.py
    """
    ca, sa = np.cos(angle), np.sin(angle)
    C = 1 - ca
    x, y, z = axis
    xs, ys, zs, xC, yC, zC = x * sa, y * sa, z * sa, x * C, y * C, z * C
    xyC, yzC, zxC = x * yC, y * zC, z * xC
    R = np.array([[x * xC + ca, xyC - zs, zxC + ys],
                  [xyC + zs, y * yC + ca, yzC - xs],
                  [zxC - ys, yzC + xs, z * zC + ca]])
    return R

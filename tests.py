import sys
import numpy as np


class Error(Exception):
    pass


def test_camera_utils():
    P = np.array([[ 7.29623172e-02, -5.17799277e-02, -1.02734764e-02, -9.62027582e+04],
                  [-5.01011603e-02, -6.23291457e-02, -4.15721807e-02, -2.59250341e+05],
                  [2.78193760e-08,  7.15619726e-08, -1.43761111e-07,  1.00000000e+00]])
    from bundle_adjust.camera_utils import compose_perspective_camera, decompose_perspective_camera, \
                                            compose_affine_camera, decompose_affine_camera
    K, R, _, oC = decompose_perspective_camera(P)
    if not np.allclose(P, compose_perspective_camera(K, R, oC)):
        raise Error('perspective decomposition failed')
    P = np.array([[7.61064055e-01, -9.35843155e-01, -1.00554841e-01, -1.13554311e+06],
                  [6.65950776e-02, -7.40405784e-02, 1.36333044e+00, 4.07093217e+06],
                  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    if not np.allclose(P, compose_affine_camera(*decompose_affine_camera(P))):
        raise Error('affine decomposition failed')


def test_ba_rotate():
    R = np.array([[0.25538431, -0.96424759, -0.07074919],
                  [0.86330366, 0.19447877, 0.46570891],
                  [-0.43529948, -0.18001279, 0.8821053]])
    from bundle_adjust.ba_rotate import euler_angles_from_R, euler_angles_to_R, \
                                        euler_to_quaternion, quaternion_to_euler, R_to_quaternion, quaternion_to_R
    if not np.allclose(R, euler_angles_to_R(*euler_angles_from_R(R))):
        raise Error('conversion between rotation matrix and euler angles failed')
    if not np.allclose(euler_angles_from_R(R), quaternion_to_euler(*euler_to_quaternion(*euler_angles_from_R(R)))):
        raise Error('conversion between quaternion and euler angles failed')
    if not np.allclose(R, quaternion_to_R(*R_to_quaternion(R))):
        raise Error('conversion between quaternion and rotation matrix failed')
    if not np.allclose(R, quaternion_to_R(*R_to_quaternion(R))):
        raise Error('conversion between axis-angle and rotation matrix failed')

def main():

    test_camera_utils()
    test_ba_rotate()

    print('all tests ok')
    
    
if __name__ == "__main__":
    sys.exit(main()) 
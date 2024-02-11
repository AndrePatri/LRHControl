import numpy as np
import math

def incremental_rotate(
                    q_initial: np.quaternion, 
                    d_angle, 
                    axis) -> np.quaternion:

    # np.quaternion is [w,x,y,z]
    q_incremental = np.array([np.cos(d_angle / 2),
                                axis[0] * np.sin(d_angle / 2),
                                axis[1] * np.sin(d_angle / 2),
                                axis[2] * np.sin(d_angle / 2)
                                ])

    # normalize the quaternion
    q_incremental /= np.linalg.norm(q_incremental)

    # initial orientation of the base

    # final orientation of the base
    q_result = np.quaternion(*q_incremental) * np.quaternion(*q_initial)

    return q_result

def quaternion_multiply(q1: np.ndarray, q2: np.ndarray):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])

def conjugate_quaternion(q: np.ndarray):
    q_conjugate = np.copy(q)
    q_conjugate[1:] *= -1.0
    return q_conjugate

def rotate_vector(vector: np.ndarray, quaternion: np.ndarray):

    # normalize the quaternion
    quaternion = quaternion / np.linalg.norm(quaternion)

    # construct a pure quaternion
    v = np.array([0, vector[0], vector[1], vector[2]])

    # rotate the vector p = q* v q
    rotated_v = quaternion_multiply(quaternion, quaternion_multiply(v, conjugate_quaternion(quaternion)))

    # extract the rotated vector
    rotated_vector = rotated_v[1:]

    return rotated_vector

def quat_to_eul(x_quat: np.ndarray, 
                y_quat: np.ndarray, 
                z_quat: np.ndarray, 
                w_quat: np.ndarray):

    # convert quaternion to Euler angles
    roll = math.atan2(2 * (w_quat * x_quat + y_quat * z_quat), 1 - 2 * (x_quat * x_quat + y_quat * y_quat))
    pitch = math.asin(2 * (w_quat * y_quat - z_quat * x_quat))
    yaw = math.atan2(2 * (w_quat * z_quat + x_quat * y_quat), 1 - 2 * (y_quat * y_quat + z_quat * z_quat))

    roll = math.degrees(roll)
    pitch = math.degrees(pitch)
    yaw = math.degrees(yaw)

    return np.array([roll, pitch, yaw])

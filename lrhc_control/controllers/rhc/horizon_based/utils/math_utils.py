
import numpy as np

def w2hor_frame(v_w: np.ndarray,
        q_b: np.ndarray,
        v_out: np.ndarray):
    """
    Transforms a velocity vector expressed in WORLD frame to
    an "horizontal" frame (z aligned as world, x aligned as the projection
    of the x-axis of the base frame described by q_b). This is useful for specifying locomotion
    references in a "game"-like fashion.
    v_out will hold the result
    """
    # q_b = q_b / q_b.norm(dim=1, keepdim=True)
    q_w, q_i, q_j, q_k = q_b[:, 3], q_b[:, 0], q_b[:, 1], q_b[:, 2]
    
    R_11 = 1 - 2 * (q_j ** 2 + q_k ** 2)
    R_21 = 2 * (q_i * q_j + q_k * q_w)
    
    norm = np.sqrt(R_11 ** 2 + R_21 ** 2)
    x_proj_x = R_11 / norm
    x_proj_y = R_21 / norm
    
    y_proj_x = -x_proj_y
    y_proj_y = x_proj_x
        
    v_out[:, 0] = v_w[:, 0] * x_proj_x + v_w[:, 1] * x_proj_y
    v_out[:, 1] = v_w[:, 0] * y_proj_x + v_w[:, 1] * y_proj_y
    v_out[:, 2] = v_w[:, 2]  # z-component remains the same

def hor2w_frame(v_h: np.ndarray,
        q_b: np.ndarray,
        v_out: np.ndarray):
    """
    Transforms a velocity vector expressed in "horizontal" frame to WORLD
    v_out will hold the result
    """

    # Extract quaternion components
    q_w, q_i, q_j, q_k = q_b[:, 3], q_b[:, 0], q_b[:, 1], q_b[:, 2]
    
    # Compute rotation matrix elements
    R_11 = 1 - 2 * (q_j ** 2 + q_k ** 2)
    R_21 = 2 * (q_i * q_j + q_k * q_w)
    
    # Normalize to get projection components
    norm = np.sqrt(R_11 ** 2 + R_21 ** 2)
    x_proj_x = R_11 / norm
    x_proj_y = R_21 / norm
    
    # Orthogonal vector components
    y_proj_x = -x_proj_y
    y_proj_y = x_proj_x
    
    # Transform velocity vector components from horizontal to world frame
    v_out[:, 0] = v_h[:, 0] * x_proj_x + v_h[:, 1] * y_proj_x
    v_out[:, 1] = v_h[:, 0] * x_proj_y + v_h[:, 1] * y_proj_y
    v_out[:, 2] = v_h[:, 2]  # z-component remains the same

def base2world_frame(v_b: np.ndarray, q_b: np.ndarray, v_out: np.ndarray):
    """
    Transforms a velocity vector expressed in the base frame to
    the WORLD frame using the given quaternion that describes the orientation
    of the base with respect to the world frame. The result is written in v_out.
    """
    # q_b = q_b / q_b.norm(dim=1, keepdim=True)
    q_w, q_i, q_j, q_k = q_b[:, 3], q_b[:, 0], q_b[:, 1], q_b[:, 2]
    
    R_00 = 1 - 2 * (q_j ** 2 + q_k ** 2)
    R_01 = 2 * (q_i * q_j - q_k * q_w)
    R_02 = 2 * (q_i * q_k + q_j * q_w)
    
    R_10 = 2 * (q_i * q_j + q_k * q_w)
    R_11 = 1 - 2 * (q_i ** 2 + q_k ** 2)
    R_12 = 2 * (q_j * q_k - q_i * q_w)
    
    R_20 = 2 * (q_i * q_k - q_j * q_w)
    R_21 = 2 * (q_j * q_k + q_i * q_w)
    R_22 = 1 - 2 * (q_i ** 2 + q_j ** 2)
    
    # Extract the velocity components in the base frame
    v_x, v_y, v_z = v_b[:, 0], v_b[:, 1], v_b[:, 2]
    
    # Transform the velocity to the world frame
    v_out[:, 0] = v_x * R_00 + v_y * R_01 + v_z * R_02
    v_out[:, 1] = v_x * R_10 + v_y * R_11 + v_z * R_12
    v_out[:, 2] = v_x * R_20 + v_y * R_21 + v_z * R_22

def world2base_frame(v_w: np.ndarray, q_b: np.ndarray, v_out: np.ndarray):
    """
    Transforms a velocity vector expressed in the WORLD frame to
    the base frame using the given quaternion that describes the orientation
    of the base with respect to the world frame. The result is written in v_out.
    """
    # q_b = q_b / q_b.norm(dim=1, keepdim=True)
    q_w, q_i, q_j, q_k = q_b[:, 3], q_b[:, 0], q_b[:, 1], q_b[:, 2]
    
    R_00 = 1 - 2 * (q_j ** 2 + q_k ** 2)
    R_01 = 2 * (q_i * q_j - q_k * q_w)
    R_02 = 2 * (q_i * q_k + q_j * q_w)
    
    R_10 = 2 * (q_i * q_j + q_k * q_w)
    R_11 = 1 - 2 * (q_i ** 2 + q_k ** 2)
    R_12 = 2 * (q_j * q_k - q_i * q_w)
    
    R_20 = 2 * (q_i * q_k - q_j * q_w)
    R_21 = 2 * (q_j * q_k + q_i * q_w)
    R_22 = 1 - 2 * (q_i ** 2 + q_j ** 2)
    
    # Extract the velocity components in the world frame
    v_x, v_y, v_z = v_w[:, 0], v_w[:, 1], v_w[:, 2]
    
    # Transform the velocity to the base frame using the transpose of the rotation matrix
    v_out[:, 0] = v_x * R_00 + v_y * R_10 + v_z * R_20
    v_out[:, 1] = v_x * R_01 + v_y * R_11 + v_z * R_21
    v_out[:, 2] = v_x * R_02 + v_y * R_12 + v_z * R_22

if __name__ == "__main__":  

    n_envs = 5000
    v_b = np.random.rand(n_envs, 3)

    q_b = np.random.rand(n_envs, 4)
    q_b_norm = q_b / np.linalg.norm(q_b, axis=1, keepdims=True)

    v_w = np.zeros_like(v_b)  # To hold horizontal frame velocities
    v_b_recovered = np.zeros_like(v_b)  # To hold recovered world frame velocities
    base2world_frame(v_b, q_b_norm, v_w)
    world2base_frame(v_w, q_b_norm, v_b_recovered)
    assert np.allclose(v_b, v_b_recovered, atol=1e-6), "Test failed: v_w_recovered does not match v_b"
    print("Forward test passed: v_b_recovered matches v_b")
    
    v_b2 = np.zeros_like(v_b)  # To hold horizontal frame velocities
    v_w_recovered = np.zeros_like(v_b)
    world2base_frame(v_b, q_b_norm, v_b2)
    base2world_frame(v_b2, q_b_norm, v_w_recovered)
    assert np.allclose(v_b, v_w_recovered, atol=1e-6), "Test failed: v_w_recovered does not match v_b"
    print("Backward test passed: v_w_recovered matches v_w")
    
    # test transf. world-horizontal frame
    v_h = np.zeros_like(v_b)  # To hold horizontal frame velocities
    v_recovered = np.zeros_like(v_b)
    w2hor_frame(v_b, q_b_norm, v_h)
    hor2w_frame(v_h, q_b_norm, v_recovered)
    assert np.allclose(v_b, v_recovered, atol=1e-6), "Test failed: v_recovered does not match v_b"
    print("horizontal forward frame test passed:  matches ")

    v_w = np.zeros_like(v_b)  # To hold horizontal frame velocities
    v_h_recovered = np.zeros_like(v_b)
    hor2w_frame(v_b, q_b_norm, v_w)
    w2hor_frame(v_w, q_b_norm, v_h_recovered)
    assert np.allclose(v_b, v_h_recovered, atol=1e-6), "Test failed: v_h_recovered does not match v_b"
    print("horizontal backward frame test passed:  matches ")
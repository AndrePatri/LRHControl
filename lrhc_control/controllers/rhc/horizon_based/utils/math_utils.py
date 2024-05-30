
import numpy as np

def w2hor_frame(t_w: np.ndarray,
        q_b: np.ndarray,
        t_out: np.ndarray):
    """
    Transforms a twist vector expressed in WORLD frame to
    an "horizontal" frame (z aligned as world, x aligned as the projection
    of the x-axis of the base frame described by q_b). This is useful for specifying locomotion
    references in a "game"-like fashion.
    t_out will hold the result
    """
    # q_b = q_b / q_b.norm(dim=1, keepdim=True)
    q_w, q_i, q_j, q_k = q_b[3, :], q_b[0, :], q_b[1, :], q_b[2, :]
    
    R_11 = 1 - 2 * (q_j ** 2 + q_k ** 2)
    R_21 = 2 * (q_i * q_j + q_k * q_w)
    
    norm = np.sqrt(R_11 ** 2 + R_21 ** 2)
    x_proj_x = R_11 / norm
    x_proj_y = R_21 / norm
    
    y_proj_x = -x_proj_y
    y_proj_y = x_proj_x
        
    t_out[0, :] = t_w[0, :] * x_proj_x + t_w[1, :] * x_proj_y
    t_out[1, :] = t_w[0, :] * y_proj_x + t_w[1, :] * y_proj_y
    t_out[2, :] = t_w[2, :]  # z-component remains the same
    t_out[3, :] = t_w[3, :] * x_proj_x + t_w[4, :] * x_proj_y
    t_out[4, :] = t_w[3, :] * y_proj_x + t_w[4, :] * y_proj_y
    t_out[5, :] = t_w[5, :]  # z-component remains the same

def hor2w_frame(t_h: np.ndarray,
        q_b: np.ndarray,
        t_out: np.ndarray):
    """
    Transforms a velocity vector expressed in "horizontal" frame to WORLD
    v_out will hold the result
    """

    # Extract quaternion components
    q_w, q_i, q_j, q_k = q_b[3, :], q_b[0, :], q_b[1, :], q_b[2, :]
    
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
    t_out[0, :] = t_h[0, :] * x_proj_x + t_h[1, :] * y_proj_x
    t_out[1, :] = t_h[0, :] * x_proj_y + t_h[1, :] * y_proj_y
    t_out[2, :] = t_h[2, :]  # z-component remains the same
    t_out[3, :] = t_h[3, :] * x_proj_x + t_h[4, :] * y_proj_x
    t_out[4, :] = t_h[3, :] * x_proj_y + t_h[4, :] * y_proj_y
    t_out[5, :] = t_h[5, :]  # z-component remains the same

def base2world_frame(t_b: np.ndarray, 
                q_b: np.ndarray,
                t_out: np.ndarray):
    """
    Transforms a velocity vector expressed in the base frame to
    the WORLD frame using the given quaternion that describes the orientation
    of the base with respect to the world frame. The result is written in v_out.
    """
    # q_b = q_b / q_b.norm(dim=1, keepdim=True)
    q_w, q_i, q_j, q_k = q_b[3, :], q_b[0, :], q_b[1, :], q_b[2, :]
    
    R_00 = 1 - 2 * (q_j ** 2 + q_k ** 2)
    R_01 = 2 * (q_i * q_j - q_k * q_w)
    R_02 = 2 * (q_i * q_k + q_j * q_w)
    
    R_10 = 2 * (q_i * q_j + q_k * q_w)
    R_11 = 1 - 2 * (q_i ** 2 + q_k ** 2)
    R_12 = 2 * (q_j * q_k - q_i * q_w)
    
    R_20 = 2 * (q_i * q_k - q_j * q_w)
    R_21 = 2 * (q_j * q_k + q_i * q_w)
    R_22 = 1 - 2 * (q_i ** 2 + q_j ** 2)
    
    # Transform the velocity to the world frame
    t_out[0, :] = t_b[0, :] * R_00 + t_b[1, :] * R_01 + t_b[2, :] * R_02
    t_out[1, :] = t_b[0, :] * R_10 + t_b[1, :] * R_11 + t_b[2, :] * R_12
    t_out[2, :] = t_b[0, :] * R_20 + t_b[1, :] * R_21 + t_b[2, :] * R_22
    t_out[3, :] = t_b[3, :] * R_00 + t_b[4, :] * R_01 + t_b[5, :] * R_02
    t_out[4, :] = t_b[3, :] * R_10 + t_b[4, :] * R_11 + t_b[5, :] * R_12
    t_out[5, :] = t_b[3, :] * R_20 + t_b[4, :] * R_21 + t_b[5, :] * R_22

def world2base_frame(t_w: np.ndarray, 
                q_b: np.ndarray, 
                t_out: np.ndarray):
    """
    Transforms a velocity vector expressed in the WORLD frame to
    the base frame using the given quaternion that describes the orientation
    of the base with respect to the world frame. The result is written in v_out.
    """
    # q_b = q_b / q_b.norm(dim=1, keepdim=True)
    q_w, q_i, q_j, q_k = q_b[3, :], q_b[0, :], q_b[1, :], q_b[2, :]
    
    R_00 = 1 - 2 * (q_j ** 2 + q_k ** 2)
    R_01 = 2 * (q_i * q_j - q_k * q_w)
    R_02 = 2 * (q_i * q_k + q_j * q_w)
    
    R_10 = 2 * (q_i * q_j + q_k * q_w)
    R_11 = 1 - 2 * (q_i ** 2 + q_k ** 2)
    R_12 = 2 * (q_j * q_k - q_i * q_w)
    
    R_20 = 2 * (q_i * q_k - q_j * q_w)
    R_21 = 2 * (q_j * q_k + q_i * q_w)
    R_22 = 1 - 2 * (q_i ** 2 + q_j ** 2)
        
    # Transform the velocity to the base frame using the transpose of the rotation matrix
    t_out[0, :] = t_w[0, :] * R_00 + t_w[1, :] * R_10 + t_w[2, :] * R_20
    t_out[1, :] = t_w[0, :] * R_01 + t_w[1, :] * R_11 + t_w[2, :] * R_21
    t_out[2, :] = t_w[0, :] * R_02 + t_w[1, :] * R_12 + t_w[2, :] * R_22
    t_out[3, :] = t_w[3, :] * R_00 + t_w[4, :] * R_10 + t_w[5, :] * R_20
    t_out[4, :] = t_w[3, :] * R_01 + t_w[4, :] * R_11 + t_w[5, :] * R_21
    t_out[5, :] = t_w[3, :] * R_02 + t_w[4, :] * R_12 + t_w[5, :] * R_22

if __name__ == "__main__":  

    n_envs = 5000
    t_b = np.random.rand(6, n_envs)

    q_b = np.random.rand(4, n_envs)
    q_b_norm = q_b / np.linalg.norm(q_b, axis=0, keepdims=True)

    t_w = np.zeros_like(t_b)  # To hold horizontal frame velocities
    t_b_recovered = np.zeros_like(t_b)  # To hold recovered world frame velocities
    base2world_frame(t_b, q_b_norm, t_w)
    world2base_frame(t_w, q_b_norm, t_b_recovered)
    assert np.allclose(t_b, t_b_recovered, atol=1e-6), "Test failed: t_w_recovered does not match t_b"
    print("Forward test passed: t_b_recovered matches t_b")
    
    t_b2 = np.zeros_like(t_b)  # To hold horizontal frame velocities
    t_w_recovered = np.zeros_like(t_b)
    world2base_frame(t_b, q_b_norm, t_b2)
    base2world_frame(t_b2, q_b_norm, t_w_recovered)
    assert np.allclose(t_b, t_w_recovered, atol=1e-6), "Test failed: t_w_recovered does not match t_b"
    print("Backward test passed: t_w_recovered matches t_w")
    
    # test transf. world-horizontal frame
    v_h = np.zeros_like(t_b)  # To hold horizontal frame velocities
    v_recovered = np.zeros_like(t_b)
    w2hor_frame(t_b, q_b_norm, v_h)
    hor2w_frame(v_h, q_b_norm, v_recovered)
    assert np.allclose(t_b, v_recovered, atol=1e-6), "Test failed: v_recovered does not match t_b"
    print("horizontal forward frame test passed:  matches ")

    t_w = np.zeros_like(t_b)  # To hold horizontal frame velocities
    v_h_recovered = np.zeros_like(t_b)
    hor2w_frame(t_b, q_b_norm, t_w)
    w2hor_frame(t_w, q_b_norm, v_h_recovered)
    assert np.allclose(t_b, v_h_recovered, atol=1e-6), "Test failed: v_h_recovered does not match t_b"
    print("horizontal backward frame test passed:  matches ")

    
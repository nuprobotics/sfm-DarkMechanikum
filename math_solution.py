import numpy as np

def triangulation(
        camera_matrix: np.ndarray,
        camera_position1: np.ndarray,
        camera_rotation1: np.ndarray,
        camera_position2: np.ndarray,
        camera_rotation2: np.ndarray,
        image_points1: np.ndarray,
        image_points2: np.ndarray
):
    """
    Triangulate 3D points from corresponding 2D image points in two images.

    :param camera_matrix: Intrinsic matrix for both cameras, np.ndarray of shape (3, 3)
    :param camera_position1: Position of the first camera in world coordinates, np.ndarray of shape (3,)
    :param camera_rotation1: Rotation matrix from camera 1 to world coordinates, np.ndarray of shape (3, 3)
    :param camera_position2: Position of the second camera in world coordinates, np.ndarray of shape (3,)
    :param camera_rotation2: Rotation matrix from camera 2 to world coordinates, np.ndarray of shape (3, 3)
    :param image_points1: Points in the first image, np.ndarray of shape (N, 2)
    :param image_points2: Corresponding points in the second image, np.ndarray of shape (N, 2)
    :return: Triangulated 3D points, np.ndarray of shape (N, 3)
    """

    # Function to compute the projection matrix for a camera
    def compute_projection_matrix(K, R_cw, C):
        # Transpose R to get rotation from world to camera coordinates
        R_wc = R_cw.T
        # Compute the translation vector t = -R_wc * C
        t = -R_wc @ C.reshape(-1, 1)
        # Construct the projection matrix P = K * [R_wc | t]
        P = K @ np.hstack((R_wc, t))
        return P

    # Compute the projection matrices for both cameras
    P1 = compute_projection_matrix(camera_matrix, camera_rotation1, camera_position1)
    P2 = compute_projection_matrix(camera_matrix, camera_rotation2, camera_position2)

    # Number of points
    num_points = image_points1.shape[0]
    points_3d = []

    for i in range(num_points):
        p1 = image_points1[i]
        p2 = image_points2[i]

        # Build matrix A for each point
        A = np.array([
            p1[0] * P1[2, :] - P1[0, :],
            p1[1] * P1[2, :] - P1[1, :],
            p2[0] * P2[2, :] - P2[0, :],
            p2[1] * P2[2, :] - P2[1, :]
        ])

        # Solve for the 3D point using SVD
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X = X / X[3]  # Convert from homogeneous to Cartesian coordinates

        points_3d.append(X[:3])

    return np.array(points_3d)

import cv2
import numpy as np

def rotation_matrix_to_quaternion(R):
    """
    Convert a rotation matrix to a quaternion.
    
    Parameters:
    - R: A 3x3 rotation matrix.
    
    Returns:
    - A quaternion in the format [x, y, z, w].
    """
    # Make sure the matrix is a numpy array
    R = np.asarray(R)
    # Allocate space for the quaternion
    q = np.empty((4,), dtype=np.float32)
    # Compute the quaternion components
    q[3] = np.sqrt(np.maximum(0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    q[0] = np.sqrt(np.maximum(0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    q[1] = np.sqrt(np.maximum(0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
    q[2] = np.sqrt(np.maximum(0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
    q[0] *= np.sign(q[0] * (R[2, 1] - R[1, 2]))
    q[1] *= np.sign(q[1] * (R[0, 2] - R[2, 0]))
    q[2] *= np.sign(q[2] * (R[1, 0] - R[0, 1]))
    return q

def resize_rgb_image_to_depth_image(rgb_image, depth_image):
    # check if already the same size elegantly
    if rgb_image.shape == depth_image.shape:
        return rgb_image
    depth_height, depth_width = depth_image.shape
    resized_rgb = cv2.resize(rgb_image, (depth_width, depth_height), interpolation=cv2.INTER_LINEAR)
    return resized_rgb

def depth_to_local_pointcloud(depth_image, intrinsics):
    fx = intrinsics[0, 0] # Focal length in x
    fy = intrinsics[1, 1] # Focal length in y
    cx = intrinsics[0, 2] # Principal point in x
    cy = intrinsics[1, 2] # Principal point in y
    
    # Get the height (h) and width (w) of the depth image
    h, w = depth_image.shape

    # Generate a grid of coordinates corresponding to the indices of the depth image
    y_indices, x_indices = np.indices((h, w), dtype=np.float32)

    # Normalize pixel coordinates (x_indices, y_indices) around the principal point (cx, cy)
    # This normalization translates pixel coordinates into the camera's coordinate system
    # where the origin is at the principal point and scaled by the focal lengths.
    normalized_x = (x_indices - cx) / fx
    normalized_y = (y_indices - cy) / fy

    # Multiply normalized coordinates by the depth image to project to real-world scale
    # The depth acts as a scaling factor to project the normalized image plane coordinates
    # into 3D space. This step converts 2D pixel coordinates into 3D camera coordinates.
    x = normalized_x * depth_image
    y = normalized_y * depth_image
    z = depth_image  # z coordinate is the depth itself

    # Stack x, y, z coordinates along the last axis to form a single array of 3D points
    points_3d = np.stack((x, y, z), axis=-1)

    return points_3d

def keypoints_to_3D(keypoints, depth_image, intrinsics):
    points_3d = []
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])  # keypoint coordinates
        if 0 <= x < depth_image.shape[1] and 0 <= y < depth_image.shape[0]:  # ensure within bounds
            depth = depth_image[y, x]
            if depth > 0:  # valid depth value
                x_3d = (x - intrinsics[0, 2]) * depth / intrinsics[0, 0]
                y_3d = (y - intrinsics[1, 2]) * depth / intrinsics[1, 1]
                points_3d.append([x_3d, y_3d, depth])
    return np.array(points_3d, dtype=np.float32)

def get_colors_for_pointcloud(rgb_image, depth_image):
    resized_rgb_image = resize_rgb_image_to_depth_image(rgb_image, depth_image)
    # points_3d = depth_to_local_pointcloud(depth_image, intrinsics)
    h, w = depth_image.shape
    y_indices, x_indices = np.indices((h, w), dtype=np.int32)
    colors = resized_rgb_image[y_indices, x_indices].reshape(-1, 3)
    colors = rgb_image.reshape(-1, 3) / 255.0  # Normalize to [0, 1]
    # colors = 1.0 - colors  # Invert colors
    return colors

def transform_points_to_global(points_3d, pose):
    
    points_reshaped = points_3d.reshape(-1, 3)
    num_points = points_reshaped.shape[0]
    
    # Create a column of ones to extend (x, y, z) to homogeneous coordinates (x, y, z, 1)
    ones = np.ones((num_points, 1), dtype=points_3d.dtype)
    points_homogeneous = np.hstack((points_reshaped, ones))
    
    # Apply the transformation matrix to the points
    transformed_homogeneous = points_homogeneous @ pose.T
    
    # Convert back from homogeneous to Cartesian coordinates by slicing off the last column
    transformed_points = transformed_homogeneous[:, :3]
    
    return transformed_points
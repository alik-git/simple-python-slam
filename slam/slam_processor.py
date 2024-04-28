

from turtle import update
import cv2
import numpy as np
import open3d as o3d

from slam.keypoint_matching import build_pose_matrix, create_views_and_matches, estimate_pose_from_essential_matrix, estimate_pose_from_matches, estimate_pose_pnp_ransac, get_keypoint_pointcloud, get_matches, update_keypoint_tracker
from utils.geometry import depth_to_local_pointcloud, get_colors_for_pointcloud, keypoints_to_3D, transform_points_to_global
from utils.optional_rerun_wrapper import orr_log_camera, orr_log_depth_image, orr_log_global_pointcloud, orr_log_matches, orr_log_orb_keypoints, orr_log_rgb_image

def process_frame(
    frame_idx, 
    obs,
    intrinsics,
    orb,
    global_pointcloud,
    prev_pose,
    prev_keypoints,
    prev_descriptors,
    prev_opencv_image,
    prev_estimated_pose,
    keypoint_tracker
    ):
    
    print(f"Processing frame {frame_idx}")

    # Unpack observation
    pil_image = obs.pil_image
    curr_pose = obs.trajectory
    img_width, img_height = pil_image.size
    depth_tensor = obs.depth_tensor
    img_path = obs.img_path
    
    # log ground truth
    orr_log_camera(intrinsics, curr_pose, prev_pose, img_width, img_height, frame_idx)
    orr_log_rgb_image(img_path)
    orr_log_depth_image(depth_tensor.squeeze())
    
    opencv_image = np.array(pil_image)  # Directly use the RGB format from PIL without converting
    if prev_opencv_image is not None:   
        views, matches = create_views_and_matches([opencv_image, prev_opencv_image])
        k=1
    
    # Detect ORB keypoints and descriptors
    keypoints = None
    descriptors = None
    curr_estimated_pose = None
    keypoints, descriptors = orb.detectAndCompute(opencv_image, None)

    
    # Project keypoints to 3D , another ground truth log for sanity checking
    keypoint_pointcloud = get_keypoint_pointcloud(keypoints, depth_tensor.squeeze().numpy(), curr_pose.numpy(), intrinsics)
    orr_log_global_pointcloud(keypoint_pointcloud, entity_label="orb_keypoints_ground_truth")
    
    
    
    # Get matches between the current and previous frame
    matches = get_matches(
        frame_idx,
        keypoints,
        descriptors,
        opencv_image,
        prev_keypoints,
        prev_descriptors,
        prev_opencv_image,
    )
    
    keypoint_tracker = update_keypoint_tracker(matches, frame_idx, keypoint_tracker, keypoints)
    
    orr_log_orb_keypoints(keypoints, keypoint_tracker, frame_idx)

    
    # Estimate the camera pose from the matches
    # R, tvec, mask = estimate_pose_pnp_ransac(matches, prev_keypoints, keypoints, intrinsics)
    R, tvec, mask = estimate_pose_from_matches(matches, prev_keypoints, keypoints, intrinsics)
    curr_estimated_pose = build_pose_matrix(R, tvec)
    # Log the estimated camera pose if it exists
    if prev_estimated_pose is not None and curr_estimated_pose is not None:
        orr_log_camera(intrinsics, curr_estimated_pose, prev_estimated_pose, img_width, img_height, frame_idx, label="estimated_camera", trajectory_color=[0, 0, 255])
        k=1
        
    
    # get 3d points from depth and transform them to global coordinates
    points3d = depth_to_local_pointcloud(depth_tensor.squeeze().numpy(), intrinsics)
    points3d_colors = get_colors_for_pointcloud(opencv_image, depth_tensor.squeeze().numpy())
    transformed_points3d = transform_points_to_global(points3d, curr_pose.numpy())
    
    # Convert numpy array of points to Open3D point cloud
    current_pointcloud = o3d.geometry.PointCloud()
    current_pointcloud.points = o3d.utility.Vector3dVector(transformed_points3d)
    current_pointcloud.colors = o3d.utility.Vector3dVector(points3d_colors  / 255.0)  # Normalize colors to [0, 1]

    # Merge with the global point cloud
    global_pointcloud += current_pointcloud

    # Optionally downsample the point cloud to manage size
    global_pointcloud = global_pointcloud.voxel_down_sample(voxel_size=0.05)
    orr_log_global_pointcloud(global_pointcloud)
    

    return curr_pose, keypoints, descriptors, opencv_image, curr_estimated_pose, keypoint_tracker
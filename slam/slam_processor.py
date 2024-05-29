

from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
import hydra
import matplotlib.pyplot as plt


from slam.keypoint_matching import build_pose_matrix, create_views_and_matches, estimate_pose_from_essential_matrix, estimate_pose_from_matches, estimate_pose_pnp_ransac, get_keypoint_pointcloud, get_matches
from utils.geometry import depth_to_local_pointcloud, get_colors_for_pointcloud, keypoints_to_3D, transform_points_to_global
from utils.optional_rerun_wrapper import orr_log_camera, orr_log_depth_image, orr_log_global_pointcloud, orr_log_matches, orr_log_matches_image, orr_log_orb_keypoints, orr_log_rgb_image

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
    orr_log_orb_keypoints(keypoints, frame_idx)
    
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
    
    # Visualize the matches
    if prev_opencv_image is not None and matches is not None:
        def validate_keypoints_and_matches(prev_keypoints, keypoints, matches, frame_idx):
            print(f"Frame {frame_idx}:")
            print(f"Number of keypoints in previous frame: {len(prev_keypoints)}")
            print(f"Number of keypoints in current frame: {len(keypoints)}")
            print(f"Number of matches: {len(matches)}")
            if matches:
                match_distances = [m.distance for m in matches]
                print(f"Average match distance: {np.mean(match_distances)}")
                
        validate_keypoints_and_matches(prev_keypoints, keypoints, matches, frame_idx)
        match_img = cv2.drawMatches(prev_opencv_image, prev_keypoints, opencv_image, keypoints, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        match_img_path = Path(hydra_output_dir) / f"matches_frame_{frame_idx}.png"
        print(f"Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")
        cv2.imwrite(str(match_img_path), match_img)
        orr_log_matches_image(match_img_path, frame_idx=frame_idx)
        k=1
        

    
    # Estimate the camera pose from the matches
    # R, tvec, mask = estimate_pose_pnp_ransac(matches, prev_keypoints, keypoints, intrinsics)
    R, tvec, mask = estimate_pose_from_matches(matches, prev_keypoints, keypoints, intrinsics)
    curr_estimated_pose = build_pose_matrix(R, tvec)
    
    def check_translation_scale(tvec):
        if tvec is None:
            return
        scale = np.linalg.norm(tvec)
        print(f"Translation vector scale: {scale}")

    # Call this after recovering the pose
    check_translation_scale(tvec)
    
    def visualize_matches_and_inliers(prev_image, curr_image, prev_keypoints, curr_keypoints, matches, mask, frame_idx):
        if matches is None:
            return

        prev_points = np.array([prev_keypoints[m.queryIdx].pt for m in matches])
        curr_points = np.array([curr_keypoints[m.trainIdx].pt for m in matches])
        
        inlier_matches = [m for i, m in enumerate(matches) if mask[i]]
        outlier_matches = [m for i, m in enumerate(matches) if not mask[i]]
        
        match_img_inliers = cv2.drawMatches(prev_image, prev_keypoints, curr_image, curr_keypoints, inlier_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        match_img_outliers = cv2.drawMatches(prev_image, prev_keypoints, curr_image, curr_keypoints, outlier_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        match_img_inliers_path = Path(hydra_output_dir) / f"matches_inliers_frame_{frame_idx}.png"
        match_img_outliers_path = Path(hydra_output_dir) / f"matches_outliers_frame_{frame_idx}.png"
        
        cv2.imwrite(str(match_img_inliers_path), match_img_inliers)
        cv2.imwrite(str(match_img_outliers_path), match_img_outliers)
        
        print(f"Inlier matches image saved to {match_img_inliers_path}")
        print(f"Outlier matches image saved to {match_img_outliers_path}")

    # Call this after estimating the essential matrix and recovering the pose
    visualize_matches_and_inliers(prev_opencv_image, opencv_image, prev_keypoints, keypoints, matches, mask, frame_idx)
    k=1

    # Log the estimated camera pose if it exists
    if prev_estimated_pose is not None and curr_estimated_pose is not None:
        orr_log_camera(intrinsics, curr_estimated_pose, prev_estimated_pose, img_width, img_height, frame_idx, label="estimated_camera", trajectory_color=[0, 0, 255])
        k=1

    def check_coordinate_system_consistency(curr_pose, prev_pose):
        print(f"Current Pose:\n{curr_pose}")
        print(f"Previous Pose:\n{prev_pose}")

    # Check coordinate system consistency
    check_coordinate_system_consistency(curr_pose, prev_pose)
        
    
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
    global_pointcloud = global_pointcloud.voxel_down_sample(voxel_size=0.01)
    orr_log_global_pointcloud(global_pointcloud)

    # print current gt pose and estimated pose to compare for debugging
    # print the difference and other helpful stats
    if curr_pose is not None and curr_estimated_pose is not None:
        print("Current Pose: ", curr_pose)
        print("Estimated Pose: ", curr_estimated_pose)
        print("Difference: ", curr_pose - curr_estimated_pose)
        print(f"Line 75, R: {R}")
        print(f"Line 75, tvec: {tvec}")
    
    

    return curr_pose, keypoints, descriptors, opencv_image, curr_estimated_pose
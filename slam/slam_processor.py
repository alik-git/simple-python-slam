import copy
from pathlib import Path
import time
import cv2
import numpy as np
import open3d as o3d
import hydra
import matplotlib.pyplot as plt
import torch


from slam.keypoint_matching import build_pose_matrix, create_views_and_matches, estimate_pose_from_essential_matrix, estimate_pose_from_matches, estimate_pose_pnp_ransac, get_keypoint_pointcloud, get_matches
from utils.geometry import depth_to_local_pointcloud, get_colors_for_pointcloud, keypoints_to_3D, transform_points_to_global
from utils.optional_rerun_wrapper import orr_log_camera, orr_log_depth_image, orr_log_global_pointcloud, orr_log_matches, orr_log_matches_image, orr_log_orb_keypoints, orr_log_rgb_image

# open3d tutorial function
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])

# open3d tutorial function
def preprocess_point_cloud(pcd, voxel_size=0.05):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

# open3d tutorial function
def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" \
            % distance_threshold)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result


def process_obs_queue(
    obs_queue,
    intrinsics,
):
    """
    Process the observation queue to estimate the camera pose.

    Args:
        obs_queue (list): A list of observations.
        intrinsics: The camera intrinsics.

    Returns:
        torch.Tensor: The estimated camera pose as a 4x4 matrix.
    """
    pass

    # get observations
    latest_obs = obs_queue[-1]
    prev_obs = obs_queue[-2]

    # get pointclouds
    latest_points_np = depth_to_local_pointcloud(latest_obs.depth_tensor.squeeze().numpy(), intrinsics)
    prev_points_np = depth_to_local_pointcloud(prev_obs.depth_tensor.squeeze().numpy(), intrinsics)
    latest_unposed_pcd = o3d.geometry.PointCloud()
    latest_unposed_pcd.points = o3d.utility.Vector3dVector(latest_points_np.reshape(-1, 3))
    prev_unposed_pcd = o3d.geometry.PointCloud()
    prev_unposed_pcd.points = o3d.utility.Vector3dVector(prev_points_np.reshape(-1, 3))

    # add colors
    latest_pcd_colors = get_colors_for_pointcloud(np.array(latest_obs.pil_image), latest_obs.depth_tensor.squeeze().numpy())
    prev_pcd_colors = get_colors_for_pointcloud(np.array(prev_obs.pil_image), prev_obs.depth_tensor.squeeze().numpy())
    latest_unposed_pcd.colors = o3d.utility.Vector3dVector(latest_pcd_colors)
    prev_unposed_pcd.colors = o3d.utility.Vector3dVector(prev_pcd_colors)

    # downsample
    latest_unposed_pcd = latest_unposed_pcd.voxel_down_sample(voxel_size=0.005)
    prev_unposed_pcd = prev_unposed_pcd.voxel_down_sample(voxel_size=0.005)

    # # print num points
    # print(f"Number of points in latest unposed pcd: {len(latest_unposed_pcd.points)}")
    # print(f"Number of points in prev unposed pcd: {len(prev_unposed_pcd.points)}")

    # estimate normals
    print("Estimating normals")
    latest_unposed_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    prev_unposed_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    print("Normals estimated")

    # log unposed pointclouds
    orr_log_global_pointcloud(latest_unposed_pcd, entity_label="latest_unposed_pcd")
    orr_log_global_pointcloud(prev_unposed_pcd, entity_label="prev_unposed_pcd")

    ### do the registration to estimate the pose
    source = prev_unposed_pcd
    target = latest_unposed_pcd
    threshold = 0.01

    source_down, source_fpfh = preprocess_point_cloud(source)
    target_down, target_fpfh = preprocess_point_cloud(target)

    start = time.time()
    result_fast = execute_fast_global_registration(
        source_down, target_down, source_fpfh, target_fpfh, voxel_size=0.05
    )

    print("Fast global registration took %.3f sec.\n" % (time.time() - start))
    print(result_fast)
    # draw_registration_result(source_down, target_down, result_fast.transformation)

    print("Apply point-to-plane ICP")
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source, target, threshold, result_fast.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=5000))
    print(reg_p2l)
    print("Transformation is:")
    print(reg_p2l.transformation)
    # draw_registration_result(source, target, reg_p2l.transformation)

    # Estimated camera pose (4x4 matrix)
    # estimated_pose = reg_p2l.transformation
    estimated_pose = torch.tensor(reg_p2l.transformation, dtype=torch.float32)
    print("Estimated camera pose:")
    print(estimated_pose)

    # apply correction to the pose, not sure why this is needed but it is 
    P = torch.tensor([
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ]).float()
    estimated_pose = P @ estimated_pose @ P.T

    return estimated_pose

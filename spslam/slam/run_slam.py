import logging
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import open3d as o3d
from sympy import python
import torch

# Ensure the root directory of the project is in sys.path
root_path = str(Path(__file__).resolve().parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

from spslam.slam.slam_processor import process_obs_queue
from spslam.utils.geometry import depth_to_local_pointcloud, get_colors_for_pointcloud, keypoints_to_3D, transform_points_to_global
from spslam.utils.dataloader import SLAMDataset, DatasetRegistry
from spslam.utils.optional_rerun_wrapper import OptionalReRun, orr_log_camera, orr_log_depth_image, orr_log_matches, orr_log_orb_keypoints, orr_log_global_pointcloud, orr_log_rgb_image

import numpy as np
import cv2 
from collections import deque


# @hydra.main(version_base=None, config_path="../conf", config_name="record3d_config")
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO)
    print(OmegaConf.to_yaml(cfg))

    # Initialize OptionalReRun instance
    orr = OptionalReRun()
    orr.set_use_rerun(cfg.use_rerun)

    orr.init("realtime_mapping")
    orr.spawn(memory_limit='10GB') # change this as you like

    dataset_full_path = Path(cfg.datasets_base_path) / cfg.dataset_locs.folder_name
    dataset = DatasetRegistry.get_dataset(cfg.dataconfigs, dataset_full_path, cfg.dataset_locs.scene)

    cam_params = cfg.dataconfigs.camera_params
    intrinsics = np.array([
        [cam_params.fx, 0, cam_params.cx],
        [0, cam_params.fy, cam_params.cy],
        [0, 0, 1]
    ])
    print(f"Line 39, intrinsics: {intrinsics}")

    orb = cv2.ORB_create()

    gt_global_pointcloud = o3d.geometry.PointCloud()

    estimated_poses = []

    prev_gt_pose = None
    prev_keypoints = None
    prev_descriptors = None
    prev_opencv_image = None  # Store the previous frame's image for visualization
    # identity_pose = torch.tensor([
    #         [1.0, 0.0, 0.0, 0.0],
    #         [0.0, 1.0, 0.0, 0.0],
    #         [0.0, 0.0, 1.0, 0.0],
    #         [0.0, 0.0, 0.0, 1.0],
    # ])
    # prev_estimated_pose = identity_pose
    prev_estimated_pose = None

    obs_queue = deque(maxlen=5)
    gt_global_pointcloud = o3d.geometry.PointCloud()

    for frame_idx, obs in enumerate(dataset):

        if frame_idx % 10 != 0:
            continue

        if frame_idx == 10:
            pass

        if frame_idx == 200:
            pass

        if frame_idx == 50:
            pass

        # if frame_idx >= 51:
        #     break

        # Unpack observation
        obs.frame_idx = frame_idx
        pil_image = obs.pil_image
        rgb_image_np = np.array(pil_image)
        curr_gt_pose = obs.gt_pose
        img_width, img_height = pil_image.size
        depth_tensor = obs.depth_tensor
        img_path = obs.img_path

        # log ground truth
        orr_log_camera(intrinsics, curr_gt_pose, prev_gt_pose, img_width, img_height, frame_idx)
        orr_log_rgb_image(img_path)
        orr_log_depth_image(depth_tensor.squeeze())

        # log ground truth pointcloud
        current_points_np = depth_to_local_pointcloud(depth_tensor.squeeze().numpy(), intrinsics)
        gt_posed_points_np = transform_points_to_global(current_points_np, curr_gt_pose.numpy())
        gt_current_posed_pcd = o3d.geometry.PointCloud()
        gt_current_posed_pcd.points = o3d.utility.Vector3dVector(gt_posed_points_np)
        pcd_colors = get_colors_for_pointcloud(rgb_image_np, depth_tensor.squeeze().numpy())
        gt_current_posed_pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
        gt_current_posed_pcd = gt_current_posed_pcd.voxel_down_sample(voxel_size=0.001)
        gt_global_pointcloud += gt_current_posed_pcd
        gt_global_pointcloud = gt_global_pointcloud.voxel_down_sample(voxel_size=0.001)
        orr_log_global_pointcloud(gt_global_pointcloud, entity_label="pcd_ground_truth")

        obs_queue.append(obs)

        # estimate the pose of the current frame
        if frame_idx >= 1:
            relative_pose = process_obs_queue(obs_queue, intrinsics)
            curr_estimated_pose = prev_estimated_pose @ relative_pose        
        else:
            # log identity estimated pose or start off with gt pose
            curr_estimated_pose = curr_gt_pose

        # log the estimated camera pose
        orr_log_camera(intrinsics, curr_estimated_pose, prev_estimated_pose, img_width, img_height, frame_idx, label="estimated_camera", trajectory_color=[0, 255, 0])
        
        # update the 'previous' values
        prev_estimated_pose = curr_estimated_pose
        prev_gt_pose = curr_gt_pose


if __name__ == "__main__":
    main()

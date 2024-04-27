import logging
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import open3d as o3d
from sympy import python

# Ensure the root directory of the project is in sys.path
root_path = str(Path(__file__).resolve().parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

from slam.slam_processor import process_frame
from utils.geometry import depth_to_local_pointcloud, get_colors_for_pointcloud, keypoints_to_3D, transform_points_to_global
from utils.dataset_registry import DatasetRegistry
from utils.dataloader import SLAMDataset
from utils.optional_rerun_wrapper import OptionalReRun, orr_log_camera, orr_log_depth_image, orr_log_matches, orr_log_orb_keypoints, orr_log_global_pointcloud, orr_log_rgb_image
import numpy as np
import cv2 


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO)
    print(OmegaConf.to_yaml(cfg))

    # Initialize OptionalReRun instance
    orr = OptionalReRun()
    orr.set_use_rerun(cfg.use_rerun)

    orr.init("realtime_mapping")
    orr.spawn()

    dataset_full_path = Path(cfg.datasets_base_path) / cfg.dataset_locs.folder_name
    dataset = DatasetRegistry.get_dataset(cfg.dataconfigs, dataset_full_path, cfg.dataset_locs.scene)
    
    cam_params = cfg.dataconfigs.camera_params
    intrinsics = np.array([
        [cam_params.fx, 0, cam_params.cx],
        [0, cam_params.fy, cam_params.cy],
        [0, 0, 1]
    ])

    orb = cv2.ORB_create()
    
    global_pointcloud = o3d.geometry.PointCloud()
    
    # estimated_poses = []

    prev_pose = None
    prev_keypoints = None
    prev_descriptors = None
    prev_opencv_image = None  # Store the previous frame's image for visualization
    prev_estimated_pose = None
    

    for frame_idx, obs in enumerate(dataset):
        
        if frame_idx % 10 != 0:
            continue

        if frame_idx == 10:
            pass

        if frame_idx == 50:
            pass
        
        # if frame_idx >= 51:
        #     break
        
        curr_pose, keypoints, descriptors, opencv_image, curr_estimated_pose = process_frame(
            frame_idx, 
            obs,
            intrinsics,
            orb,
            global_pointcloud,
            prev_pose,
            prev_keypoints,
            prev_descriptors,
            prev_opencv_image,
            prev_estimated_pose
        )
        
        prev_pose = curr_pose
        prev_keypoints = keypoints
        prev_descriptors = descriptors
        prev_opencv_image = opencv_image
        prev_estimated_pose = curr_estimated_pose
        
        if frame_idx <2:
            prev_estimated_pose = curr_pose



if __name__ == "__main__":
    main()

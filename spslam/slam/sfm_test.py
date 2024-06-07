
import logging
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import open3d as o3d
from sympy import python
import pyceres as ceres
import numpy as np

# Ensure the root directory of the project is in sys.path
root_path = str(Path(__file__).resolve().parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

from slam.slam_processor import process_frame_queue
from utils.geometry import depth_to_local_pointcloud, get_colors_for_pointcloud, keypoints_to_3D, transform_points_to_global
from utils.dataloader import SLAMDataset, DatasetRegistry
from utils.optional_rerun_wrapper import OptionalReRun, orr_log_camera, orr_log_depth_image, orr_log_matches, orr_log_matches_image, orr_log_orb_keypoints, orr_log_global_pointcloud, orr_log_rgb_image
import numpy as np
import cv2 

def print_pose_errors(gt_poses, est_poses):
    for i, (gt_pose, est_pose) in enumerate(zip(gt_poses, est_poses)):
        gt_R, gt_t = gt_pose[:3, :3], gt_pose[:3, 3]
        est_R, est_t = est_pose[:3, :3], est_pose[:3, 3]
        rot_diff, trans_diff = calculate_pose_difference(gt_pose, est_R, est_t)
        print(f"Frame {i}: Rotation Difference: {rot_diff}, Translation Difference: {trans_diff}")


def extract_keypoints_and_descriptors(image, orb):
    # Convert the PIL image to a NumPy array and then to grayscale
    image_np = np.array(image)
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(gray_image, None)
    
    return keypoints, descriptors

def match_keypoints(descriptors1, descriptors2):
    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors
    matches = bf.match(descriptors1, descriptors2)
    
    # Sort them in the order of their distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Print distance distribution of matches
    distances = [match.distance for match in matches]
    print(f"Match distances: min = {min(distances)}, max = {max(distances)}, mean = {np.mean(distances):.2f}, median = {np.median(distances):.2f}")
    
    
    return matches

def estimate_pose(matches, keypoints1, keypoints2, intrinsics):
    # Extract point coordinates from keypoints
    pts1 = np.array([keypoints1[m.queryIdx].pt for m in matches])
    pts2 = np.array([keypoints2[m.trainIdx].pt for m in matches])
    
    # Compute the essential matrix
    E, mask = cv2.findEssentialMat(pts1, pts2, intrinsics, method=cv2.RANSAC, prob=0.999, threshold=0.5)
    
    # Recover the pose from the essential matrix
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, intrinsics)
    
    return R, t, mask_pose

def triangulate_points(matches, keypoints1, keypoints2, R, t, intrinsics):
    # Extract point coordinates from keypoints
    pts1 = np.array([keypoints1[m.queryIdx].pt for m in matches])
    pts2 = np.array([keypoints2[m.trainIdx].pt for m in matches])
    
    # Convert to homogeneous coordinates
    pts1_hom = cv2.convertPointsToHomogeneous(pts1)[:, 0, :]
    pts2_hom = cv2.convertPointsToHomogeneous(pts2)[:, 0, :]
    
    # Create projection matrices
    P1 = np.dot(intrinsics, np.hstack((np.eye(3), np.zeros((3, 1)))))
    P2 = np.dot(intrinsics, np.hstack((R, t)))
    
    # Triangulate points
    pts4D_hom = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    
    # Convert from homogeneous coordinates
    pts3D = pts4D_hom[:3] / pts4D_hom[3]
    
    return pts3D.T

def calculate_pose_difference(gt_pose, est_R, est_t):
    gt_R = gt_pose[:3, :3]
    gt_t = gt_pose[:3, 3]
    # Convert to numpy and ensure the dtype is float64
    gt_R = np.array(gt_R, dtype=np.float64)
    gt_t = np.array(gt_t, dtype=np.float64)
    
    # Convert estimated R and t to float64
    est_R = np.array(est_R, dtype=np.float64)
    est_t = np.array(est_t, dtype=np.float64).squeeze()
    
    rot_diff = cv2.norm(gt_R, est_R, cv2.NORM_L2)
    trans_diff = cv2.norm(gt_t, est_t, cv2.NORM_L2)
    
    return rot_diff, trans_diff

def create_pose_matrix(R, t):
    pose_matrix = np.eye(4)
    pose_matrix[:3, :3] = R
    pose_matrix[:3, 3] = t.flatten()
    return pose_matrix

def verify_matches(matches, keypoints1, keypoints2):
    for i, match in enumerate(matches[:10]):  # Print first 10 matches for verification
        pt1 = keypoints1[match.queryIdx].pt
        pt2 = keypoints2[match.trainIdx].pt
        print(f"Match {i}: Image 1 - {pt1}, Image 2 - {pt2}")
        
def bundle_adjustment(poses, points_3d, observations, intrinsics):
    # Prepare data for Ceres
    problem = ceres.Problem()
    
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    
    for i, obs in enumerate(observations):
        pose_idx, point_idx, x, y = obs
        camera = poses[pose_idx]
        point = points_3d[point_idx]

        # Define residual block
        cost_function = ceres.CreateReprojectionError(fx, fy, cx, cy, x, y)
        problem.AddResidualBlock(cost_function, None, camera, point)
    
    # Set solver options
    options = ceres.SolverOptions()
    options.linear_solver_type = ceres.LinearSolverType.DENSE_SCHUR
    options.minimizer_progress_to_stdout = True

    # Solve the problem
    summary = ceres.Summary()
    ceres.Solve(options, problem, summary)
    print(summary.BriefReport())
    
    return poses, points_3d



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
    print(f"Line 39, intrinsics: {intrinsics}")

    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    
    global_pointcloud = o3d.geometry.PointCloud()
    
    estimated_poses = []
    points_3d = []
    gt_poses = []
    observations = []

    prev_pose = None
    prev_keypoints = None
    prev_descriptors = None
    prev_opencv_image = None  # Store the previous frame's image for visualization
    prev_estimated_pose = None
    
    k=1
    
    # get the first 5 frames of the dataset with stride 50 to test the pipeline, 
    # put them in an array 
    frames_5 = []
    for i in range(0, 250, 50):
        frames_5.append(dataset[i])
        
    
    for frame_idx, obs in enumerate(frames_5):
        print(f"\nProcessing frame {frame_idx}")

        # Unpack observation
        pil_image = obs.pil_image
        curr_pose = obs.trajectory
        img_width, img_height = pil_image.size
        depth_tensor = obs.depth_tensor
        img_path = obs.img_path
        
        # print all of the variables
        print(f"curr_pose: {curr_pose}")
        print(f"img_width: {img_width}")
        print(f"img_height: {img_height}")
        print(f"depth_tensor: {depth_tensor[:5]}")
        print(f"img_path: {img_path}")
        
        # log ground truth
        curr_cam_label = f"gt_camera_{frame_idx}"
        orr_log_camera(intrinsics, curr_pose, prev_pose, img_width, img_height, frame_idx, label=curr_cam_label)
        orr_log_rgb_image(img_path, curr_cam_label)
        orr_log_depth_image(depth_tensor.squeeze(), curr_cam_label)
        
         # Extract keypoints and descriptors
        keypoints, descriptors = extract_keypoints_and_descriptors(pil_image, orb)
        
        # Print keypoints and descriptors for verification
        print(f"Number of keypoints in frame {frame_idx}: {len(keypoints)}")
        print(f"Descriptors shape: {descriptors.shape}")
        
        orr_log_orb_keypoints(keypoints, frame_idx, label=curr_cam_label)

        
            # Store keypoints and descriptors for matching in the next step
        if frame_idx == 0:
            prev_keypoints = keypoints
            prev_descriptors = descriptors
        else:
            # Proceed with matching in the next step
            pass
        
        if frame_idx > 0:
            # Match keypoints between the previous and current frame
            matches = match_keypoints(prev_descriptors, descriptors)
            verify_matches(matches, prev_keypoints, keypoints)
            
            # Print the number of matches for verification
            print(f"Number of matches between frame {frame_idx-1} and frame {frame_idx}: {len(matches)}")
            
            # Print coordinates of matched keypoints
            for i, match in enumerate(matches[:10]):  # Print first 10 matches for brevity
                pt1 = prev_keypoints[match.queryIdx].pt
                pt2 = keypoints[match.trainIdx].pt
                print(f"Match {i}: Image 1 - {pt1}, Image 2 - {pt2}")
                
            # Estimate pose
            R, t, mask_pose = estimate_pose(matches, prev_keypoints, keypoints, intrinsics)
            print(f"Estimated Rotation:\n{R}")
            print(f"Estimated Translation:\n{t}")
            
             # Triangulate points
            pts3D = triangulate_points(matches, prev_keypoints, keypoints, R, t, intrinsics)
            print(f"Triangulated 3D points: {pts3D[:10]}")  # Print first 10 points for brevity
            
            rot_diff, trans_diff = calculate_pose_difference(curr_pose, R, t)
            print(f"Rotation Difference: {rot_diff}")
            print(f"Translation Difference: {trans_diff}")
            
            # Create estimated pose matrix
            estimated_pose = create_pose_matrix(R, t)
            
            # Log estimated camera pose
            estimated_cam_label = f"estimated_camera_{frame_idx}"
            orr_log_camera(intrinsics, estimated_pose, prev_estimated_pose, img_width, img_height, frame_idx, label=estimated_cam_label)

            # Update the previous estimated pose
            prev_estimated_pose = estimated_pose
                        
                    
            # Visualize and log the matches
            opencv_image = np.array(pil_image)

            # Convert images from RGB to BGR for OpenCV
            prev_opencv_image_bgr = cv2.cvtColor(prev_opencv_image, cv2.COLOR_RGB2BGR)
            opencv_image_bgr = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)

            match_img = cv2.drawMatches(prev_opencv_image_bgr, prev_keypoints, opencv_image_bgr, keypoints, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            # Convert match_img back to RGB
            match_img_rgb = match_img #  cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)

            hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            match_img_path = Path(hydra_output_dir) / f"matches_frame_{frame_idx}.png"
            print(f"Output directory: {hydra_output_dir}")
            cv2.imwrite(str(match_img_path), match_img_rgb)
            orr_log_matches_image(match_img_path, frame_idx=frame_idx)

            # Store current keypoints and descriptors for the next iteration
            prev_keypoints = keypoints
            prev_descriptors = descriptors
            prev_opencv_image = opencv_image
            gt_poses.append(curr_pose)
                        # Collect data for bundle adjustment
            for m in matches:
                observations.append((frame_idx - 1, len(points_3d), prev_keypoints[m.queryIdx].pt[0], prev_keypoints[m.queryIdx].pt[1]))
                observations.append((frame_idx, len(points_3d), keypoints[m.trainIdx].pt[0], keypoints[m.trainIdx].pt[1]))
            points_3d.extend(pts3D)
        else:
            prev_keypoints = keypoints
            prev_descriptors = descriptors
            prev_opencv_image = np.array(pil_image)
        
        prev_pose = curr_pose
        
        poses = [create_pose_matrix(curr_pose[:3, :3], curr_pose[:3, 3]) for curr_pose in estimated_poses]
        poses, points_3d = bundle_adjustment(poses, points_3d, observations, intrinsics)
        
        print_pose_errors(gt_poses, poses)

        k=1
            
    k=1 
        

if __name__ == "__main__":
    main()

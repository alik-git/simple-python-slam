import cv2
import numpy as np
import open3d as o3d

from utils.geometry import keypoints_to_3D, transform_points_to_global
from utils.optional_rerun_wrapper import orr_log_matches

def get_keypoint_pointcloud(keypoints, depth_image, curr_pose, intrinsics):
    keypoint_3d_points = keypoints_to_3D(keypoints, depth_image, intrinsics)
    transformed_keypoint_3d_points = transform_points_to_global(keypoint_3d_points, curr_pose)
    
    keypoint_pointcloud = o3d.geometry.PointCloud()
    keypoint_pointcloud.points = o3d.utility.Vector3dVector(transformed_keypoint_3d_points)
    keypoint_pointcloud.colors = o3d.utility.Vector3dVector(np.tile(np.array([1, 1, 1]), (keypoint_3d_points.shape[0], 1)))  # Blue color for keypoints
    return keypoint_pointcloud

def get_matches(
    frame_idx,
    keypoints,
    descriptors,
    opencv_image,
    prev_keypoints,
    prev_descriptors,
    prev_opencv_image,
):
        # if not (prev_keypoints is not None and prev_descriptors is not None and prev_opencv_image is not None):
        if prev_keypoints is None:
            return None
        
        # Using FLANN matcher instead of brute-force
        flann_index = 6  # Index parameters for LSH
        index_params = dict(algorithm=flann_index, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)  # Search parameters

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(prev_descriptors, descriptors, k=2)

        # Apply Lowe's ratio test
        good_matches = []
        for item in matches:
            if len(item) < 2:
                continue
            m, n = item
            if m.distance < 0.75 * n.distance:  # Lowe's ratio test
                good_matches.append(m)
                
        matches = good_matches
                
        # # Initialize the Matcher for matching the keypoints and then match the keypoints
        # matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # matches = matcher.match(prev_descriptors, descriptors)
        
        orr_log_matches(prev_opencv_image, prev_keypoints, opencv_image, keypoints, matches, frame_idx)
        
        return matches
    
next_keypoint_id = 0
    
def update_keypoint_tracker(matches, current_frame, keypoint_tracker, current_keypoints):
    new_tracker = {}
    
    if not matches:
        return new_tracker
    
    for match in matches:
        query_idx = match.queryIdx  # Index in current keypoints list
        train_idx = match.trainIdx  # Index in previous keypoints list
        
        if train_idx in keypoint_tracker:
            # Continue the existing match chain from the previous keypoint
            match_chain = keypoint_tracker[train_idx]['match_chain'][:]
            match_chain.append((current_frame, query_idx))
            match_count = len(match_chain)
        else:
            # Start a new match chain for unmatched keypoints
            match_chain = [(current_frame, query_idx)]
            match_count = 1
        
        new_tracker[query_idx] = {
            'id': keypoint_tracker[train_idx]['id'] if train_idx in keypoint_tracker else query_idx,
            'origin_frame': keypoint_tracker[train_idx]['origin_frame'] if train_idx in keypoint_tracker else current_frame,
            'match_count': match_count,
            'last_seen': current_frame,
            'match_chain': match_chain
        }
    
    # Ensure new keypoints that were not matched are also tracked
    for idx, kp in enumerate(current_keypoints):
        if idx not in new_tracker:
            new_tracker[idx] = {
                'id': idx,
                'origin_frame': current_frame,
                'match_count': 1,
                'last_seen': current_frame,
                'match_chain': [(current_frame, idx)]
            }
    
    return new_tracker
    
def estimate_pose_pnp_ransac(matches, keypoints1, keypoints2, intrinsics):
    if matches is None:
        return None, None, None
    if len(matches) < 5:
        return None, None, None
    object_points = np.array([keypoints1[m.queryIdx].pt for m in matches], dtype=np.float32)
    image_points = np.array([keypoints2[m.trainIdx].pt for m in matches], dtype=np.float32)
    _, R, t, mask = cv2.solvePnPRansac(object_points, image_points, intrinsics, None)
    return R, t, mask



def estimate_pose_from_matches(matches, prev_keypoints, keypoints, intrinsics):
    if matches is None:
        return None, None, None
    if len(matches) < 5:
        return None, None, None
    
    object_points = np.array([prev_keypoints[m.queryIdx].pt for m in matches], dtype=np.float32)
    image_points = np.array([keypoints[m.trainIdx].pt for m in matches], dtype=np.float32)
    
    E, mask = cv2.findEssentialMat(object_points, image_points, intrinsics, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, tvec, mask = cv2.recoverPose(E, object_points, image_points, intrinsics)
    
    return R, tvec, mask

def build_pose_matrix(R, tvec):
    if R is None or tvec is None:
        return None
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = tvec.squeeze()
    return pose


def estimate_pose_from_essential_matrix(object_points, image_points, intrinsics):
    E, mask = cv2.findEssentialMat(object_points, image_points, intrinsics, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, tvec, mask = cv2.recoverPose(E, object_points, image_points, intrinsics)
    return R, tvec, mask

class View:
    def __init__(self, image, keypoints, descriptors, name):
        self.image = image
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.name = name
        self.R = None
        self.t = None
        
def detect_features(image):
    # Assuming ORB here, but can be SIFT, SURF, etc.
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

def match_features(descriptors1, descriptors2):
    # Create a BFMatcher or FLANN based matcher here
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches
        
def create_views_and_matches(images):
    views = []
    matches = {}  # Dictionary to hold matches between pairs of images

    for i, img in enumerate(images):
        # Here you should detect keypoints and compute descriptors
        keypoints, descriptors = detect_features(img)
        view = View(img, keypoints, descriptors, f"Image_{i}")
        views.append(view)
        
        if i > 0:
            # Matching current view with the previous view
            prev_view = views[i-1]
            match = match_features(prev_view.descriptors, descriptors)
            matches[(prev_view.name, view.name)] = match

    return views, matches
import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_and_describe(image):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

def match_keypoints(descriptors1, descriptors2):
    flann_index = 6
    index_params = dict(algorithm=flann_index, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    
    good_matches = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
    return good_matches

def update_keypoint_tracker(matches, current_frame, keypoint_tracker, current_keypoints):
    new_tracker = {}
    
    # Transfer chain data to new keypoints based on matches
    for match in matches:
        query_idx = match.queryIdx  # Index in current frame
        train_idx = match.trainIdx  # Index in previous frame

        if train_idx in keypoint_tracker:
            chain = keypoint_tracker[train_idx]['chain'] + [current_frame]
            print(f"Extend idx {query_idx} from frame {train_idx} to {current_frame}")
        else:
            chain = [current_frame]
            print(f"Starting new chain for idx {query_idx} in frame {current_frame}")

        new_tracker[query_idx] = {'chain': chain}

    # Track new keypoints that were not matched
    for idx, kp in enumerate(current_keypoints):
        if idx not in new_tracker:
            new_tracker[idx] = {'chain': [current_frame]}
            print(f"Starting new chain for untracked kp {idx} in frame {current_frame}")

    return new_tracker

def visualize_keypoints(images, keypoints, keypoint_tracker):
    output_images = []
    for img, kps in zip(images, keypoints):
        output_image = np.copy(img)
        for kp_idx, kp in enumerate(kps):
            if kp_idx in keypoint_tracker:
                chain_length = len(keypoint_tracker[kp_idx]['chain'])
                if chain_length == 1:
                    color = (0, 255, 0)  # Green for new keypoints
                elif chain_length == 2:
                    color = (0, 0, 255)  # Blue for 1 match
                elif chain_length >= 3:
                    color = (255, 0, 0)  # Red for 2+ matches
                cv2.circle(output_image, (int(kp.pt[0]), int(kp.pt[1])), 4, color, -1)
        output_images.append(output_image)

    plt.figure(figsize=(15, 5))
    for i, output_image in enumerate(output_images):
        plt.subplot(1, len(output_images), i + 1)
        plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
        plt.title(f'Frame {i * 200}')
    plt.show()

# Main sequence
image_paths = ["/home/kuwajerw/local_data/Replica/room0/results/frame000000.jpg",
               "/home/kuwajerw/local_data/Replica/room0/results/frame000200.jpg",
               "/home/kuwajerw/local_data/Replica/room0/results/frame001962.jpg"]
images = [cv2.imread(path, cv2.IMREAD_COLOR) for path in image_paths]
keypoints, descriptors = zip(*(detect_and_describe(img) for img in images))

matches_0_200 = match_keypoints(descriptors[0], descriptors[1])
matches_200_500 = match_keypoints(descriptors[1], descriptors[2])

keypoint_tracker = update_keypoint_tracker(matches_0_200, 200, {}, keypoints[0])
keypoint_tracker = update_keypoint_tracker(matches_200_500, 500, keypoint_tracker, keypoints[1])

visualize_keypoints(images, keypoints, keypoint_tracker)

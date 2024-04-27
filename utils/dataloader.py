import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from PIL import Image
from natsort import natsorted
import glob

import torchvision.transforms as T

# import sys
# # Ensure the root directory of the project is in sys.path
# root_path = str(Path(__file__).resolve().parent.parent)
# if root_path not in sys.path:
#     sys.path.append(root_path)

from utils.dataset_registry import DatasetRegistry

# # Assuming 'tensor_image' is your image tensor.
# # If your tensor was on a GPU, first bring it back to CPU memory.
# tensor_image = tensor_image.cpu()

# # If your tensor is in the form of (C, H, W), convert it to (H, W, C) for PIL.
# if tensor_image.dim() == 3:
#     tensor_image = tensor_image.permute(1, 2, 0)

# # If it has a batch dimension (B, C, H, W), remove it and convert to (H, W, C).
# elif tensor_image.dim() == 4:
#     tensor_image = tensor_image.squeeze(0).permute(1, 2, 0)

# # The tensor may also need to be converted to byte type if it's not already.
# tensor_image = tensor_image.byte()

# # Convert the tensor to a PIL image.
# pil_image = T.ToPILImage()(tensor_image)

# # Now you can display the image.
# pil_image.show()


class SlamObservation:
    """
    Represents a sample in the SLAM dataset.
    """
    def __init__(self, img_path, pil_image, image_tensor, depth_tensor=None, trajectory=None):
        self.img_path = img_path
        self.pil_image = pil_image
        self.image_tensor = image_tensor
        self.depth_tensor = depth_tensor
        self.trajectory = trajectory

class SLAMDataset(Dataset):
    """
    A general SLAM dataset class that can be inherited by specific scene datasets.
    """
    def __init__(self, dataconfig, dataset_path: Path, scene_name: str, has_depth=False):
        self.dataset_path = Path(dataset_path)
        self.scene_name = scene_name
        self.has_depth = has_depth
        self.images, self.depth_images = self.load_images()
        self.trajectory_data = self.load_trajectory_data()
        self.png_depth_scale = dataconfig["camera_params"]["png_depth_scale"]
        
    def load_images(self):
        """
        Method to load images and depth images if available. Should be implemented by subclasses.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
        
    def load_trajectory_data(self):
        """
        Method to load trajectory data. Should be implemented by subclasses.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def __len__(self):
        """Return the minimum length of images and trajectory data."""
        return min(len(self.images), len(self.trajectory_data))

    def __getitem__(self, idx):
        """Get dataset item by index."""
        img_path = self.images[idx]
        pil_image = Image.open(img_path).convert('RGB')  # Open and convert image to RGB
        image_tensor = torch.from_numpy(np.array(pil_image)).permute(2, 0, 1).float()  # Convert image to torch tensor
        
        if self.has_depth:
            depth_img_path = self.depth_images[idx]
            depth_pil_image = Image.open(depth_img_path)# .convert('L')  # Open and convert depth image to grayscale
            depth_tensor = torch.from_numpy(np.array(depth_pil_image)).unsqueeze(0).float()  # Convert depth image to torch tensor
            depth_tensor = depth_tensor / self.png_depth_scale
            k=1
        else:
            depth_tensor = None
        
        trajectory = torch.tensor(self.trajectory_data[idx], dtype=torch.float32)
        trajectory = trajectory.view(4, 4)  # Reshape trajectory to 4x4 matrix
        
        return SlamObservation(img_path, pil_image, image_tensor, depth_tensor, trajectory)

@DatasetRegistry.register('replica')
class ReplicaDataset(SLAMDataset):
    """
    A SLAM dataset class for the Replica dataset.
    Inherits from SLAMDataset and implements loading methods specific to the Replica dataset structure.
    """
    def __init__(self, dataconfig, dataset_path: Path, scene_name: str):
        super().__init__(dataconfig, dataset_path, scene_name, has_depth=True)

    def load_images(self):
        """
        Loads all RGB image file paths and depth image file paths from the 'results' directory and sorts them in natural order.
        """
        input_folder = self.dataset_path / self.scene_name / 'results'
        rgb_images = natsorted(glob.glob(f"{input_folder}/frame*.jpg"))
        depth_images = natsorted(glob.glob(f"{input_folder}/depth*.png"))  # Assuming depth images follow a similar naming convention
        return rgb_images, depth_images

    def load_trajectory_data(self):
        """
        Load trajectory data from a file named 'traj.txt' located in the scene's directory.
        """
        trajectory_file = self.dataset_path / self.scene_name / 'traj.txt'
        trajectory_data = np.loadtxt(trajectory_file, dtype=np.float32)
        return trajectory_data

if __name__ == '__main__':
    replica_dataset = ReplicaDataset('/home/kuwajerw/local_data/Replica', 'room0')
    k=1
#     print(f"Dataset length: {len(replica_dataset)}")
#     image, trajectory = replica_dataset[0]
#     print(f"Image shape: {image.shape}")
#     print(f"Trajectory shape: {trajectory.shape}")
#     k=1
#     # Assuming 'tensor_image' is your image tensor.
#     # If your tensor was on a GPU, first bring it back to CPU memory.
#     tensor_image = image.cpu()

#     # Assuming 'tensor_image' is the image tensor with shape [H, W, C]
#     tensor_image = tensor_image.cpu()

#     # Convert the tensor to a PIL image directly if it's in (H, W, C) format.
#     pil_image = Image.fromarray(tensor_image.numpy().astype('uint8'))

#     # Now you can display the image.
#     pil_image.show()
# # Usage example:
# # replica_dataset = ReplicaDataset('/home/kuwajerw/local_data/Replica', 'room0')

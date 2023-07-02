import torch
from torch.utils.data import Dataset
from PIL import Image
import open3d as o3d

class ProprioDataset(Dataset):
    def __init__(self, dataset_pth, img_tfs=None, label_tfs=None):
        self.dataset_pth = dataset_pth
        self.img_tfs = img_tfs
        self.label_tfs = label_tfs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Load image and label
        img_path = self.data[index][0]

        # Load image using PIL
        img = Image.open(f"{self.dataset_pth}/img/{index}.jpg")
        pcd = o3d.io.read_point_cloud(f"{self.dataset_pth}/pcd/{index}.ply")

        # Apply transformations if provided
        if self.img_tfs:
            img = self.img_tfs(img)
        if self.label_tfs:
            label = self.label_tfs(label)

        return img, label

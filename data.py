import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import open3d as o3d

class ProprioDataset(Dataset):
    def __init__(self, dataset_pth: Path, img_tfs=None, label_tfs=None):
        self.dataset_pth = dataset_pth
        self.img_tfs = img_tfs
        self.label_tfs = label_tfs
        self.rgb_path = self.dataset_pth / 'vine_rgb'
        self.pcd_path = self.dataset_pth / 'kinect_pcd'

    def __len__(self):
        return len(list(self.rgb_path.glob('*')))

    def __getitem__(self, index):
        # Load image using PIL
        img = Image.open(self.rgb_path / f'{index}.jpeg')

        pcd = o3d.io.read_point_cloud(str(self.pcd_path / f'{index}.ply'))

        # Apply transformations if provided
        if self.img_tfs:
            img = self.img_tfs(img)
        if self.label_tfs:
            label = self.label_tfs(pcd)

        return img.double(), label.double()

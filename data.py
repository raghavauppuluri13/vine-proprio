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

    def __len__(self):
        return len(list((self.dataset_pth / 'img').glob('*')))

    def __getitem__(self, index):
        # Load image using PIL
        img = Image.open(self.dataset_pth / 'img' / f'{index}.jpeg')
        pcd = o3d.io.read_point_cloud(str(self.dataset_pth / 'pcd' / f'{index}.ply'))

        # Apply transformations if provided
        if self.img_tfs:
            img = self.img_tfs(img)
        if self.label_tfs:
            label = self.label_tfs(pcd)

        return img, label

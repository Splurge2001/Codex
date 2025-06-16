import os
import numpy as np
import torch
from torch.utils.data import Dataset

class PointCloudDataset(Dataset):
    """Dataset for point cloud txt files organised in class folders."""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        
        if class_names is None:
            # automatically collect folder names that are actual directories
            class_names = [d for d in sorted(os.listdir(root_dir))
                           if os.path.isdir(os.path.join(root_dir, d))]
        for label, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            self.class_to_idx[class_name] = label
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith('.txt'):
                    self.samples.append((os.path.join(class_dir, fname), label))

        self.classes = list(self.class_to_idx.keys())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        data = np.loadtxt(file_path).astype(np.float32)
        # normalise each file to [0, 1] range if possible
        if data.max() > data.min():
            data = (data - data.min()) / (data.max() - data.min())
        tensor = torch.from_numpy(data).unsqueeze(0)
        if self.transform:
            tensor = self.transform(tensor)
        return tensor, label

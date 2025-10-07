import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
import torch
from torch.utils.data import Dataset


class TemporalSpectralDataset(Dataset):

    def __init__(self,
                 root_dir: str,
                 transform=None,
                 target_transform=None,
                 n_components: int = 32):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.n_components = n_components
        self.classes = sorted(os.listdir(os.path.join(root_dir, "spectral_features")))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.temporal_features = []
        all_columns_for_pca = []

        for cls in self.classes:
            cls_csv = os.path.join(root_dir, "temporal_features", cls, "output1.csv")
            df = pd.read_csv(cls_csv,
                             header=None if cls == "Healthy" else 0,
                             nrows=100_000, 
                             dtype=np.float32)

            arr = df.values
            self.temporal_features.append(arr)

            all_columns_for_pca.append(arr.T)   

        all_columns_for_pca = np.concatenate(all_columns_for_pca, axis=0)
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(all_columns_for_pca)


        self.items = []      
        for cls in self.classes:
            class_idx = self.class_to_idx[cls]
            cls_img_dir = os.path.join(root_dir, "spectral_features", cls)
            img_files = sorted(f for f in os.listdir(cls_img_dir) if f.endswith(".png"))

            for col_idx, fname in enumerate(img_files):
                img_path = os.path.join(cls_img_dir, fname)
                self.items.append((img_path, class_idx, col_idx))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        img_path, class_idx, col_idx = self.items[idx]

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        raw_column = self.temporal_features[class_idx][:, col_idx] 
        raw_column = raw_column.reshape(1, -1)                        

        pca_feat = self.pca.transform(raw_column)                   
        pca_feat = torch.from_numpy(pca_feat.squeeze()).float()     

        label = class_idx
        if self.target_transform:
            label = self.target_transform(label)

        return pca_feat, img, label

import os, gzip
import numpy as np
import torch
from torch.utils.data import Dataset


def posenc(x, l = 6):
    rets = [x]
    for i in range(l):
        for fn in [torch.sin, torch.cos]:
            rets.append(fn(2. ** i * x))
    return torch.concat(rets, -1)


def transform_viewpoint(v, use_pos_enc = False):
    """
    Transforms the viewpoint vector into a consistent
    representation
    """
    pos, z = torch.split(v, 3, dim=-1)
    y, p = torch.split(z, 1, dim=-1)

    if use_pos_enc:
        view = torch.cat([y, p], dim=-1)
        pos = posenc(pos, 10)
        view = posenc(view, 4)
    else:
        view = torch.cat([torch.cos(y), torch.sin(y), torch.cos(p), torch.sin(p)], dim=-1)

    v_hat = torch.cat([pos, view], dim=-1)

    return v_hat


class ShepardMetzler(Dataset):
    """
    Shepart Metzler mental rotation task
    dataset. Based on the dataset provided
    in the GQN paper. Either 5-parts or
    7-parts.
    :param root_dir: location of data on disc
    :param train: whether to use train of test set
    :param transform: transform on images
    :param fraction: fraction of dataset to use
    :param target_transform: transform on viewpoints
    """

    def __init__(self, root_dir, train=True, transform=None, fraction=1.0, target_transform=transform_viewpoint, use_pos_enc=False):
        super(ShepardMetzler, self).__init__()
        assert fraction > 0.0 and fraction <= 1.0
        prefix = "train" if train else "test"
        self.root_dir = os.path.join(root_dir, prefix)
        self.records = sorted([p for p in os.listdir(self.root_dir) if "pt" in p])
        self.records = self.records[:int(len(self.records) * fraction)]
        self.transform = transform
        self.target_transform = target_transform
        self.use_pos_enc = use_pos_enc

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        scene_path = os.path.join(self.root_dir, self.records[idx])
        with gzip.open(scene_path, "r") as f:
            data = torch.load(f)
            images, viewpoints = list(zip(*data))

        images = np.stack(images)
        viewpoints = np.stack(viewpoints)

        # uint8 -> float32
        images = images.transpose(0, 1, 4, 2, 3)
        images = torch.FloatTensor(images) / 255

        if self.transform:
            images = self.transform(images)

        viewpoints = torch.FloatTensor(viewpoints)
        if self.target_transform:
            viewpoints = self.target_transform(viewpoints, use_pos_enc=self.use_pos_enc)

        return images, viewpoints

from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.utils.data as data
import numpy as np
import os
import h5py
import subprocess
import shlex

BASE_DIR = './'


def _get_data_files(list_filename):
    with open(list_filename) as f:
        return [line.rstrip()[5:] for line in f]


def _load_data_file(name):
    f = h5py.File(name)
    data = f["data"][:]
    label = f["label"][:]
    return data, label


class ModelNet40(data.Dataset):
    def __init__(self, num_points, transforms=None, split='train', download=True):
        super().__init__()

        self.transforms = transforms

        self.folder = "modelnet40_ply_hdf5_2048"
        self.data_dir = os.path.join(BASE_DIR, self.folder)
        self.url = "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"

        if download and not os.path.exists(self.data_dir):
            zipfile = os.path.join(BASE_DIR, os.path.basename(self.url))
            # subprocess.check_call(
            #     shlex.split("wget {} -o {}".format(self.url, zipfile))
            # )

            # subprocess.check_call(
            #     shlex.split("unzip {} -d {}".format(zipfile, BASE_DIR))
            # )

            # subprocess.check_call(shlex.split("rm {}".format(zipfile)))
            os.system('wget https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip')
            print("DATA DOWNLOADED")
            os.system('unzip modelnet40_ply_hdf5_2048.zip')
            print("DATA UNZIPPED")

        self.split = split
        if self.split == 'train':
            self.files = _get_data_files(os.path.join(self.data_dir, "train_files.txt"))
        elif self.split == 'test':
            self.files = _get_data_files(os.path.join(self.data_dir, "test_files.txt"))

        point_list, label_list = [], []
        for f in self.files:
            points, labels = _load_data_file(os.path.join(BASE_DIR, f))
            point_list.append(points)
            label_list.append(labels)

        self.points = np.concatenate(point_list, 0)
        self.labels = np.concatenate(label_list, 0)
        self.set_num_points(num_points)

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.num_points)
        np.random.shuffle(pt_idxs)

        current_points = self.points[idx, pt_idxs].copy()
        label = torch.from_numpy(self.labels[idx]).type(torch.LongTensor)

        if self.transforms is not None:
            current_points = self.transforms(current_points)

        return current_points, label

    def __len__(self):
        return self.points.shape[0]

    def set_num_points(self, pts):
        self.num_points = min(self.points.shape[1], pts)

    def randomize(self):
        pass


class UnlabeledModelNet40(data.Dataset):
    def __init__(self, num_points, root, transforms=None, split='unlabeled'):
        super().__init__()
        self.transforms = transforms
        root = os.path.abspath(root)
        self.folder = "modelnet40_ply_hdf5_2048"
        self.data_dir = os.path.join(root, self.folder)

        self.split, self.num_points = split, num_points
        if self.split == 'unlabeled':
            self.files =  _get_data_files( \
                os.path.join(self.data_dir, 'unlabeled_files.txt'))

        point_list = []
        for f in self.files:
            points, _ = _load_data_file(os.path.join(root, self.files[-1]))
            point_list.append(points)

        self.points = np.concatenate(point_list, 0)

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.points.shape[1])
        
        current_points = self.points[idx, pt_idxs].copy()
        
        if self.transforms is not None:
            current_points = self.transforms(current_points)
        
        current_points = torch.transpose(current_points, 1, 0)
        
        return current_points

    def __len__(self):
        return self.points.shape[0]


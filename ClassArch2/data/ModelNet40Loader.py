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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _get_data_files(list_filename):
    with open(list_filename) as f:
        return [line.rstrip()[5:] for line in f]


def _load_data_file(name):
    f = h5py.File(name)
    data = f["data"][:]
    label = f["label"][:]
    return data, label

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class ModelNet40(data.Dataset):
    def __init__(self, num_points, transforms=None, split='train', download=True):
        super().__init__()

        self.transforms = transforms
        self.train = False
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
            self.train = True
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
        if self.train:
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

class ModelNet40Norm():
    def __init__(self, num_points, transforms=None, split='train', normalize=True, normal_channel = True, cache_size=15000, download=True):
        super().__init__()

        self.transforms = transforms
        self.normalize = normalize
        self.num_points = num_points
        self.train = False
        self.folder = "modelnet40_normal_resampled"
        self.data_dir = os.path.join(BASE_DIR, self.folder)

        if download and not os.path.exists(self.data_dir):
            zipfile = os.path.join(BASE_DIR, os.path.basename(self.url))
            # subprocess.check_call(
            #     shlex.split("wget {} -o {}".format(self.url, zipfile))
            # )

            # subprocess.check_call(
            #     shlex.split("unzip {} -d {}".format(zipfile, BASE_DIR))
            # )

            # subprocess.check_call(shlex.split("rm {}".format(zipfile)))
            os.system('https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip')
            print("DATA DOWNLOADED")
            os.system('modelnet40_normal_resampled.zip')
            print("DATA UNZIPPED")

        self.catfile = os.path.join(self.data_dir, 'modelnet40_shape_names.txt')
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.labels = dict(zip(self.cat, range(len(self.cat))))  
        self.normal_channel = normal_channel
        
        shape_ids = {}
        self.split = split
        if self.split == 'train':
            self.train = True
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.data_dir, 'modelnet40_train.txt'))] 
        elif self.split == 'test':
            shape_ids['test']= [line.rstrip() for line in open(os.path.join(self.data_dir, 'modelnet40_test.txt'))]
        assert(split=='train' or split=='test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
       
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath = [(shape_names[i], os.path.join(self.data_dir, shape_names[i], shape_ids[split][i])+'.txt') for i in range(len(shape_ids[split]))]

        self.cache_size = cache_size # how many data points to cache in memory
        self.cache = {} # from index to (point_set, cls) tuple

        #if self.shuffle is None:
        #    if split == 'train': self.shuffle = True
        #    else: self.shuffle = False
        #else:
        #    self.shuffle = shuffle


    def _augment_batch_data(self, batch_data):
        if self.normal_channel:
            rotated_data = provider.rotate_point_cloud_with_normal(batch_data)
            rotated_data = provider.rotate_perturbation_point_cloud_with_normal(rotated_data)
        else:
            rotated_data = provider.rotate_point_cloud(batch_data)
            rotated_data = provider.rotate_perturbation_point_cloud(rotated_data)
    
        jittered_data = provider.random_scale_point_cloud(rotated_data[:,:,0:3])
        jittered_data = provider.shift_point_cloud(jittered_data)
        jittered_data = provider.jitter_point_cloud(jittered_data)
        rotated_data[:,:,0:3] = jittered_data
        return provider.shuffle_points(rotated_data)

    
     
    def __getitem__(self, idx):  
        if idx in self.cache:
            current_points, label = self.cache[idx]
        else:
            pt_idxs = np.arange(0, self.num_points)
            if self.train:
                 np.random.shuffle(pt_idxs)
            fn = self.datapath[idx]
            label = self.labels[self.datapath[idx][0]]
            label = torch.from_numpy(np.array([label])).type(torch.LongTensor)
            current_points = np.loadtxt(fn[1],delimiter=',').astype(np.float32)
            # Take the first npoints
            current_points = current_points[pt_idxs,:]
            if self.normalize:
                current_points[:,0:3] = pc_normalize(current_points[:,0:3])
            if self.normal_channel:
                current_points = current_points[:,0:3]
            if len(self.cache) < self.cache_size:
                self.cache[idx] = (current_points, label)
        return current_points, label


    def __len__(self):
        return len(self.datapath)


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
            points, _ = _load_data_file(os.path.join(BASE_DIR, f))
            point_list.append(points)

        self.points = np.concatenate(point_list, 0)

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, min(self.points.shape[1], self.num_points))
        
        current_points = self.points[idx, pt_idxs].copy()
        
        if self.transforms is not None:
            current_points = self.transforms(current_points)

        return current_points

    def __len__(self):
        return self.points.shape[0]

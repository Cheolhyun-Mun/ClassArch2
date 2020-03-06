import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../../"))
from shutil import copyfile
import h5py
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from ClassArch2.models.rscnn_ssn_cls import RSCNN_SSN as RSCNN
from ClassArch2.models.rscnn_ssn_cls import model_fn_decorator
from ClassArch2.data.ModelNet40Loader import UnlabeledModelNet40
import ClassArch2.data.data_utils as d_utils


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

test_transforms = transforms.Compose([
    d_utils.PointcloudToTensor()
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    torch.cuda.empty_cache()
    # torch.cuda.reset_max_memory_allocated(device=device)
    ngpus_per_node = torch.cuda.device_count()
    target_folder = './data/'

    ds_test = UnlabeledModelNet40(1024, '../data/', test_transforms, split='unlabeled')
   
    model_loc = '../train/checkpoints/rscnn_cls_best'
    checkpoints = torch.load(model_loc + '.pth.tar')
    model = RSCNN(input_channels=0, num_classes=40, use_xyz=True)
    model.load_state_dict(checkpoints['model_state'])
    model = model.cuda()
    model.eval()

    label = []
    length = ds_test.__len__()

    for i in range(length):
        X = ds_test.__getitem__(i)
        X = X.contiguous().unsqueeze(0).cuda()
        pred = model(X)
        label.append(int((pred == pred.max()).nonzero().flatten()[-1]))
       
    label = np.array(label)
    label = np.expand_dims(label, axis=1)
    label = label.astype(np.uint8)
    print('LABELING DONE...')
    print('MAKING INTO HDF FILE...')

    f1 = h5py.File('../data/modelnet40_ply_hdf5_2048/ply_data_unlabeled0.h5')
    data  = f1['data'][:]
    print(f1.keys())
    
    f2 = h5py.File('../data/modelnet40_ply_hdf5_2048/ply_data_unlabeled1.h5')
    data  = data.append(f2['data'][:])
    print(f2.keys())

    final_label = h5py.File('data/modelnet40_ply_hdf5_2048/ply_data_labeled0.h5', 'w')
    print(final_label)
    final_label['data'] = data
    final_label['label'] = label

    copyfile('data/modelnet40_ply_hdf5_2048/ply_data_unlabeled_0_id2file.json', 'data/modelnet40_ply_hdf5_2048/ply_data_labeled_0_id2file.json')

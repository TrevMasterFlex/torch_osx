import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
from torch.utils.data import Dataset

class JointKeypointsDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, dimension=3):
        self.key_pts_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.dimension = dimension

    def __len__(self):
        return len(self.key_pts_frame)

    def __getitem__(self, idx):
        sample = {'image': mpimg.imread(os.path.join(self.root_dir, self.key_pts_frame.iloc[idx, 0])), 'keypoints': self.key_pts_frame.iloc[idx, 1:].to_numpy().astype('float').reshape(-1, self.dimension)}

        if self.transform:
            sample = self.transform(sample)

        return sample

class Normalize(object):      
    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        return {'image': image/255.0, 'keypoints': (np.copy(key_pts) - 100)/50.0}

class Rescale(object):
    def __init__(self, output_size, dimension):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.dimension = dimension

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        
        if(self.dimension == 2):
            key_pts = key_pts * [new_w / w, new_h / h]

        return {'image': cv2.resize(image, (new_w, new_h)), 'keypoints': key_pts}

class ToTensor(object):
    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        
        if(len(image.shape) == 2):
            image = image.reshape(image.shape[0], image.shape[1], 1)
            
        return {'image': torch.from_numpy(image.transpose((2, 0, 1))), 'keypoints': torch.from_numpy(key_pts)}

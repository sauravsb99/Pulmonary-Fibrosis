import copy
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import pydicom
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from time import time
import warnings
from scipy.ndimage.interpolation import zoom
from enum import Enum
from torchvision import transforms
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
import pytest


class CropBoundingBox:
    @staticmethod
    def bounding_box(img3d: np.array):
        mid_img = img3d[int(img3d.shape[0] / 2)]
        same_first_row = (mid_img[0, :] == mid_img[0, 0]).all()
        same_first_col = (mid_img[:, 0] == mid_img[0, 0]).all()
        if same_first_col and same_first_row:
            return True
        else:
            return False

    def __call__(self, sample):
        image = sample['image']
        if not self.bounding_box(image):
            return sample

        mid_img = image[int(image.shape[0] / 2)]
        r_min, r_max = None, None
        c_min, c_max = None, None
        for row in range(mid_img.shape[0]):
            if not (mid_img[row, :] == mid_img[0, 0]).all() and r_min is None:
                r_min = row
            if (mid_img[row, :] == mid_img[0, 0]).all() and r_max is None \
                    and r_min is not None:
                r_max = row
                break

        for col in range(mid_img.shape[1]):
            if not (mid_img[:, col] == mid_img[0, 0]).all() and c_min is None:
                c_min = col
            if (mid_img[:, col] == mid_img[0, 0]).all() and c_max is None \
                    and c_min is not None:
                c_max = col
                break

        image = image[:, r_min:r_max, c_min:c_max]
        return {
            'features': sample['features'],
            'image': image,
            'metadata': sample['metadata'],
            'target': sample['target']
        }

class ConvertToHU:
    def __call__(self, sample):
        image, data = sample['image'], sample['metadata']

        img_type = data.ImageType
        is_hu = img_type[0] == 'ORIGINAL' and not (img_type[2] == 'LOCALIZER')
        if not is_hu:
            warnings.warn(f'Patient {data.PatientID} CT Scan not cannot be'
                          f'converted to Hounsfield Units (HU).')

        intercept = data.RescaleIntercept
        slope = data.RescaleSlope
        image = (image * slope + intercept).astype(np.int16)
        return {
            'features': sample['features'],
            'image': image,
            'metadata': data,
            'target': sample['target']
        }
class Resize:
    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
        resize_factor = np.array(self.output_size) / np.array(image.shape)
        image = zoom(image, resize_factor, mode='nearest')
        return {
            'features': sample['features'],
            'image': image,
            'metadata': sample['metadata'],
            'target': sample['target']
        }
class Clip:
    def __init__(self, bounds=(-1000, 500)):
        self.min = min(bounds)
        self.max = max(bounds)

    def __call__(self, sample):
        image = sample['image']
        image[image < self.min] = self.min
        image[image > self.max] = self.max
        return {
            'features': sample['features'],
            'image': image,
            'metadata': sample['metadata'],
            'target': sample['target']
        }
class MaskMethod(Enum):
    MORPHOLOGICAL = 1
    DEEPLEARNING = 2


class Mask:
    def __init__(self, method=MaskMethod.MORPHOLOGICAL, threshold=-400,
                 root_dir='../data/test'):
        self.threshold = threshold
        self.method = method
        self.root_dir = root_dir

    def __call__(self, sample):
        image = sample['image']
        if self.method == MaskMethod.MORPHOLOGICAL:
            for slice_id in range(image.shape[0]):
                m = self.get_morphological_mask(image[slice_id])
                image[slice_id][m == False] = image[slice_id].min()
        elif self.method == MaskMethod.DEEPLEARNING:
            # m = self.get_deeplearning_mask(data.PatientID)
            raise NotImplementedError
        else:
            raise ValueError('Unrecognized masking method')

        return {
            'features': sample['features'],
            'image': image,
            'metadata': sample['metadata'],
            'target': sample['target']
        }

    def get_morphological_mask(self, image):
        m = image < self.threshold
        m = clear_border(m)
        m = label(m)
        areas = [r.area for r in regionprops(m)]
        areas.sort()
        if len(areas) > 2:
            for region in regionprops(m):
                if region.area < areas[-2]:
                    for coordinates in region.coords:
                        m[coordinates[0], coordinates[1]] = 0
        return m > 0

    def get_deeplearning_mask(self, patient_id):
        """Very slow, must be done using GPUs
        """
        list_files = [str(x) for x in (Path(self.root_dir) / patient_id).glob('*.dcm')]
        input_image = sitk.ReadImage(list_files)
        m = mask.apply(input_image) #.squeeze()
        m[m == 2] = 1
        return m
class Normalize:
    def __init__(self, bounds=(-1000, 500)):
        self.min = min(bounds)
        self.max = max(bounds)

    def __call__(self, sample):
        image = sample['image'].astype(np.float)
        image = (image - self.min) / (self.max - self.min)
        return {
            'features': sample['features'],
            'image': image,
            'metadata': sample['metadata'],
            'target': sample['target']
        }


class ToTensor:
    def __init__(self, add_channel=True):
        self.add_channel = add_channel

    def __call__(self, sample):
        image = sample['image']
        if self.add_channel:
            image = np.expand_dims(image, axis=0)

        return {
            'features': sample['features'],
            'image': torch.from_numpy(image),
            'metadata': sample['metadata'],
            'target': sample['target']
        }

    
class ZeroCenter:
    def __init__(self, pre_calculated_mean):
        self.pre_calculated_mean = pre_calculated_mean

    def __call__(self, sample):
        return {
            'features': sample['features'],
            'image': sample['image'] - self.pre_calculated_mean,
            'metadata': sample['metadata'],
            'target': sample['target']
        }
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

root_dir = Path('./files/osic-pulmonary-fibrosis-progression')
model_dir = Path('./files/working')
pretrained_weigths_dir = Path('./files/test/pretrained_models')
pretrained_ae_weigths = pretrained_weigths_dir/'barcelona-20200722.pth'
cache_dir = Path('./files/osic-cached-dataset')
latent_dir = Path('./files/results/latent')
latent_dir.mkdir(exist_ok=True, parents=True)
# num_kfolds = 5
test_size=0.2
batch_size = 42
learning_rate = 1e-3
num_epochs = 20
quantiles = (0.2, 0.5, 0.8)
model_name ='cauchy'

device = torch.device('cpu')


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # Encoder
        self.conv1 = nn.Conv3d(1, 16, 3)
        self.conv2 = nn.Conv3d(16, 32, 3)
        self.conv3 = nn.Conv3d(32, 96, 2)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.pool2 = nn.MaxPool3d(kernel_size=3, stride=3, return_indices=True)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        # Decoder
        self.deconv1 = nn.ConvTranspose3d(96, 32, 2)
        self.deconv2 = nn.ConvTranspose3d(32, 16, 3)
        self.deconv3 = nn.ConvTranspose3d(16, 1, 3)
        self.unpool1 = nn.MaxUnpool3d(kernel_size=2, stride=2)
        self.unpool2 = nn.MaxUnpool3d(kernel_size=3, stride=3)
        self.unpool3 = nn.MaxUnpool3d(kernel_size=2, stride=2)

    def encode(self, x, return_partials=True):
        # Encoder
        x = self.conv1(x)
        up3out_shape = x.shape
        x, i1 = self.pool1(x)
        x = self.conv2(x)
        up2out_shape = x.shape
        x, i2 = self.pool2(x)
        x = self.conv3(x)
        up1out_shape = x.shape
        x, i3 = self.pool3(x)

        if return_partials:
            return x, up3out_shape, i1, up2out_shape, i2, up1out_shape, i3
        else:
            return x

    def forward(self, x):
        x, up3out_shape, i1, up2out_shape, i2, up1out_shape, i3 = self.encode(x)

        # Decoder
        x = self.unpool1(x, output_size=up1out_shape, indices=i3)
        x = self.deconv1(x)
        x = self.unpool2(x, output_size=up2out_shape, indices=i2)
        x = self.deconv2(x)
        x = self.unpool3(x, output_size=up3out_shape, indices=i1)
        x = self.deconv3(x)

        return x

class ClinicalDataset(Dataset):
    def __init__(self, root_dir, ctscans_dir, mode, transform=None,
                 cache_dir=None):
        self.transform = transform
        self.mode = mode
        self.ctscans_dir = Path(ctscans_dir)
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        # print(self.cache_dir)
        # If cache_dir is set, use cached values...
        if cache_dir is not None:
            self.raw = pd.read_csv(self.cache_dir/f'tabular_{mode}.csv')
            # print(self.raw)
            with open(self.cache_dir/'features_list.pkl', "rb") as fp:
                self.FE = pickle.load(fp)
                # print(self.FE)
            return 

        # ...otherwise, pre-process
        tr = pd.read_csv(Path(root_dir)/"train.csv")
        tr.drop_duplicates(keep=False, inplace=True, subset=['Patient', 'Weeks'])
        chunk = pd.read_csv(Path(root_dir)/"test.csv")

        sub = pd.read_csv(Path(root_dir)/"sample_submission.csv")
        sub['Patient'] = sub['Patient_Week'].apply(lambda x: x.split('_')[0])
        sub['Weeks'] = sub['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))
        sub = sub[['Patient', 'Weeks', 'Confidence', 'Patient_Week']]
        sub = sub.merge(chunk.drop('Weeks', axis=1), on="Patient")

        tr['WHERE'] = 'train'
        chunk['WHERE'] = 'val'
        sub['WHERE'] = 'test'
        data = tr.append([chunk, sub])

        data['min_week'] = data['Weeks']
        data.loc[data.WHERE == 'test', 'min_week'] = np.nan
        data['min_week'] = data.groupby('Patient')['min_week'].transform('min')

        base = data.loc[data.Weeks == data.min_week]
        base = base[['Patient', 'FVC']].copy()
        base.columns = ['Patient', 'min_FVC']
        base['nb'] = 1
        base['nb'] = base.groupby('Patient')['nb'].transform('cumsum')
        base = base[base.nb == 1]
        base.drop('nb', axis=1, inplace=True)

        data = data.merge(base, on='Patient', how='left')
        data['base_week'] = data['Weeks'] - data['min_week']
        # print(base)
        del base

        COLS = ['Sex', 'SmokingStatus']
        self.FE = []
        for col in COLS:
            for mod in data[col].unique():
                self.FE.append(mod)
                data[mod] = (data[col] == mod).astype(int)

        data['age'] = (data['Age'] - data['Age'].min()) / \
                      (data['Age'].max() - data['Age'].min())
        data['BASE'] = (data['min_FVC'] - data['min_FVC'].min()) / \
                       (data['min_FVC'].max() - data['min_FVC'].min())
        data['week'] = (data['base_week'] - data['base_week'].min()) / \
                       (data['base_week'].max() - data['base_week'].min())
        data['percent'] = (data['Percent'] - data['Percent'].min()) / \
                          (data['Percent'].max() - data['Percent'].min())
        self.FE += ['age', 'percent', 'week', 'BASE']

        self.raw = data.loc[data.WHERE == mode].reset_index()
        del data
        # return getitem

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        patient_id = self.raw['Patient'].iloc[idx]
        if self.cache_dir is None:
            patient_path = self.ctscans_dir / patient_id
            image, metadata = load_scan(patient_path)
        else:
            image = torch.load(self.cache_dir / f'{patient_id}.pt')
            metadata = pydicom.read_file(self.cache_dir / f'{patient_id}.dcm')

        sample = {
            'features': self.raw[self.FE].iloc[idx].values,
            'image': image,
            'metadata': metadata,
            'target': self.raw['FVC'].iloc[idx]
        }
        if self.transform:
            sample = self.transform(sample)
        # print(sample)
        return sample

    def cache(self, cache_dir):
        Path(cache_dir).mkdir(exist_ok=True, parents=True)

        # Cache raw features table
        self.raw.to_csv(Path(cache_dir)/f'tabular_{self.mode}.csv', index=False)

        # Cache features list
        with open(Path(cache_dir)/'features_list.pkl', "wb") as fp:
            pickle.dump(self.FE, fp)

        # Cache images and metadata
        self.raw['index'] = self.raw.index
        idx_unique = self.raw.groupby('Patient').first()['index'].values
        bar = tqdm(idx_unique.tolist())
        for idx in bar:
            sample = self[idx]
            patient_id = sample['metadata'].PatientID
            torch.save(sample['image'], Path(cache_dir)/f'{patient_id}.pt')
            sample['metadata'].save_as(Path(cache_dir)/f'{patient_id}.dcm')

# Helper function that loads CT scans in a single array. 
# This is also new vs. baselie
def load_scan(path):
    slices = [pydicom.read_file(p) for p in path.glob('*.dcm')]
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except AttributeError:
        warnings.warn(f'Patient {slices[0].PatientID} CT scan does not '
                      f'have "ImagePositionPatient". Assuming filenames '
                      f'in the right scan order.')

    image = np.stack([s.pixel_array.astype(float) for s in slices])
    return image, slices[0]
class QuantModel(nn.Module):
    def __init__(self, in_tabular_features=9, in_ctscan_features=76800,
                 out_quantiles=3):
        super(QuantModel, self).__init__()
        # This line is new. We need to know a priori the number
        # of latent features to properly flatten the tensor
        self.in_ctscan_features = in_ctscan_features

        self.fc1 = nn.Linear(in_tabular_features, 512)
        self.fc2 = nn.Linear(in_ctscan_features, 512)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, out_quantiles)

    def forward(self, x1, x2):
        # Now the quant model has 2 inputs: x1 (the tabular features)
        # and x2 (the pre-computed latent features)
        x1 = F.relu(self.fc1(x1))
        
        # Flattens the latent features and concatenate with tabular features
        x2 = x2.view(-1, self.in_ctscan_features)
        x2 = F.relu(self.fc2(x2))
        x = torch.cat([x1, x2], dim=1)
        
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
class GenerateLatentFeatures:
    def __init__(self, autoencoder, latent_dir):
        self.autoencoder = autoencoder
        self.latent_dir = Path(latent_dir)
        self.cache_dir = Path(cache_dir)

    def __call__(self, sample):
        patient_id = sample['metadata'].PatientID
        cached_latent_file = self.latent_dir/f'{patient_id}_lat.pt'

        if cached_latent_file.is_file():
            latent_features = torch.load(cached_latent_file)
        else:
            with torch.no_grad():
                img = sample['image'].float().unsqueeze(0)
                latent_features = self.autoencoder.encode(
                    img, return_partials=False).squeeze(0)
            torch.save(latent_features, cached_latent_file)

        return {
            'tabular_features': sample['features'],
            'latent_features': latent_features,
            'target': sample['target']
        }

def saanam(cachePathString):
    cachePath=Path(cachePathString)
    autoencoder = AutoEncoder()
    autoencoder.load_state_dict(torch.load(
        pretrained_ae_weigths,
        map_location=torch.device('cuda:0')
    ))
    device = torch.device('cpu')
    autoencoder.to(device)
    autoencoder.eval()

    data = ClinicalDataset(
        root_dir=root_dir,
        ctscans_dir=root_dir/'test',
        cache_dir=cachePath,
        mode='test',
        transform=GenerateLatentFeatures(autoencoder, latent_dir)
    )

    model = QuantModel().to(device)
    state_dict = torch.load(model_dir/'save.pth')
    model.load_state_dict(state_dict)
    models=[model]

    avg_preds = np.zeros((len(data), len(quantiles)))
    print(len(data), len(quantiles),avg_preds)
    for model in models:
        dataloader = DataLoader(data, batch_size=batch_size,shuffle=False, num_workers=2)
        preds = []
        for batch in tqdm(dataloader):
            inputs1 = batch['tabular_features'].float()
            inputs1 = inputs1.to(device)
            inputs2 = batch['latent_features'].float()
            inputs2 = inputs2.to(device)
            with torch.no_grad():
                preds.append(model(inputs1, inputs2))

        preds = torch.cat(preds, dim=0).cpu().numpy()
        avg_preds += preds
    print("Hello ",avg_preds)
    print(len(models))
    avg_preds /= len(models)

    print("Hello ",avg_preds)

    df = pd.DataFrame(data=avg_preds, columns=list(quantiles))
    df['Patient_Week'] = data.raw['Patient_Week']
    df['FVC'] = df[quantiles[1]]
    df['Confidence'] = df[quantiles[2]] - df[quantiles[0]]
    df = df.drop(columns=list(quantiles))
    df.to_csv(cachePath/'submission.csv', index=False)
# if __name__ == '__main__':
#     saanam('./files/osic-cached-dataset2')
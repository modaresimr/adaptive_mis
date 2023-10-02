#!/usr/bin/env python
# coding: utf-8

# ## [ISIC Challenge (2016-2020)](https://challenge.isic-archive.com/)
# ---
#
# ### [Data 2018](https://challenge.isic-archive.com/data/)
#
# The input data are dermoscopic lesion images in JPEG format.
#
# All lesion images are named using the scheme `ISIC_<image_id>.jpg`, where `<image_id>` is a 7-digit unique identifier. EXIF tags in the images have been removed; any remaining EXIF tags should not be relied upon to provide accurate metadata.
#
# The lesion images were acquired with a variety of dermatoscope types, from all anatomic sites (excluding mucosa and nails), from a historical sample of patients presented for skin cancer screening, from several different institutions. Every lesion image contains exactly one primary lesion; other fiducial markers, smaller secondary lesions, or other pigmented regions may be neglected.
#
# The distribution of disease states represent a modified "real world" setting whereby there are more benign lesions than malignant lesions, but an over-representation of malignancies.

# In[2]:

from tqdm import tqdm
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
import torch.nn.functional as F


# In[3]:

class ISIC2018Dataset(Dataset):
    def __init__(self, datadir, image_size=224, one_hot=True, **kwargs):
        # pre-set variables
        self.X = None
        self.Y = None
        self.image_size = image_size
        self.data_prefix = "ISIC_"
        self.target_postfix = "_segmentation"
        self.target_fex = "png"
        self.input_fex = "jpg"
        self.class_names = ["background", 'forground']
        self.data_dir = datadir
        self.imgs_dir = os.path.join(self.data_dir, "ISIC2018_Task1-2_Training_Input")
        self.msks_dir = os.path.join(self.data_dir, "ISIC2018_Task1_Training_GroundTruth")

        # input parameters
        self.img_dirs = glob.glob(f"{self.imgs_dir}/*.{self.input_fex}")
        self.data_ids = [d.split(self.data_prefix)[1].split(f".{self.input_fex}")[0] for d in self.img_dirs]
        self.one_hot = one_hot
        self.in_channels = 3
        self.num_classes = 2
        if not os.path.exists(f"{self.data_dir}/X_tr_{self.image_size}x{self.image_size}.npy"):
            self.convert()
        self.X = torch.tensor(np.load(f"{self.data_dir}/X_tr_{self.image_size}x{self.image_size}.npy"))
        self.Y = torch.tensor(np.load(f"{self.data_dir}/Y_tr_{self.image_size}x{self.image_size}.npy"))

        self.img_transform = transforms.Compose([
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.msk_transform = transforms.Compose([

            # transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.transform = True

    def convert(self):
        size_transforms = transforms.Compose([
            transforms.Resize(
                transforms.ToTensor(),
                size=[self.image_size, self.image_size],
                interpolation=transforms.functional.InterpolationMode.BILINEAR,
                antialias=True
            ),
        ])

        X = []
        Y = []
        print("Converting Dataset...")
        for id in tqdm(self.data_ids):
            img = self.get_img_by_id_orig(id)
            img = size_transforms(img)
            X.append(img)

            msk = self.get_msk_by_id_orig(id)
            msk = size_transforms(msk)
            Y.append(msk)

        self.X = torch.stack(X)
        self.Y = torch.stack(Y)
        np.save(f"{self.data_dir}/X_tr_{self.image_size}x{self.image_size}.npy", self.X.numpy())
        np.save(f"{self.data_dir}/Y_tr_{self.image_size}x{self.image_size}.npy", self.Y.numpy())

    def get_img_by_id_orig(self, id):
        img_dir = os.path.join(self.imgs_dir, f"{self.data_prefix}{id}.{self.input_fex}")
        img = read_image(img_dir, ImageReadMode.RGB)
        return img

    def get_msk_by_id_orig(self, id):
        msk_dir = os.path.join(self.msks_dir, f"{self.data_prefix}{id}{self.target_postfix}.{self.target_fex}")
        msk = read_image(msk_dir, ImageReadMode.GRAY)

        return msk > 0

    def get_msk_by_id(self, id):
        if self.transform:
            return self.msk_transform(self.Y[id])
        return self.Y[id]

    def get_img_by_id(self, id):
        if self.transform:
            return self.img_transform(self.X[id])
        return self.X[id]

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        # data_id = self.data_ids[idx]
        data_id = idx
        img = self.get_img_by_id(data_id)
        msk = self.get_msk_by_id(data_id)

        if self.one_hot:
            msk = torch.clamp(msk, min=0)
            msk = F.one_hot(torch.squeeze(msk).to(torch.int64))
            msk = torch.moveaxis(msk, -1, 0).to(torch.float)

        return {'image': img, 'mask': msk, 'id': data_id}

    def summary(self):
        print(f"Number of images: {len(self)}")

import os
import glob
import numpy as np
import torch
import imageio.v2 as imageio
from torch.utils.data import Dataset
from torchvision import transforms, utils
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
import torch.nn.functional as F


class SegPC2021Dataset(Dataset):
    def __init__(self, datadir, title, **kwargs):
        # pre-set variables
        self.title = title
        self.dataset_dir = datadir
        self.load_dataset()

    def load_dataset(self):

        #         build_segpc_dataset(
        #             input_size = self.input_size,
        #             scale = self.scale,
        #             data_dir = self.data_dir,
        #             dataset_dir = self.dataset_dir,
        #             mode = self.mode,
        #             force_rebuild = force_rebuild,
        #         )

        x_path_list = glob.glob(self.dataset_dir + '/x/*.bmp')
        y_path = self.dataset_dir
        if not len(x_path_list):
            x_path_list = glob.glob(self.dataset_dir + '/[tv][ra]*/x/*.bmp')
            y_path = self.dataset_dir + '/[tv][ra]*/y/'
        X = []
        Y = []
        meta = []
        for xp in x_path_list:
            fn = xp.split('/')[-1].split('.bmp')[0]
            ys = glob.glob(y_path + "/{fn}*.bmp")
            img = imageio.imread(xp)
            X.append(img)
            meta.append(fn)
            labels = np.zeros_like(img)
            for yi, y in enumerate(ys):
                msk = imageio.imread(y)
                if len(msk.shape) == 3:
                    msk = msk[:, :, 0]
                cim, nim = split_c_n(msk)
                cim = np.where(cim > 0, yi + 1, 0)
                nim = np.where(nim > 0, yi + 1, 0)

                labels = np.zeros([*nim.shape, 2], dtype=np.uint8)
                labels[:, :, 0] = cim
                labels[:, :, 1] = nim
            Y.append(labels)
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.meta = np.array(meta)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]
        msk = self.Y[idx]
        meta = self.meta[idx]

        sample = {'image': img, 'mask': msk, 'id': meta}
        return sample


def split_c_n(img, nv=40, cv=20):
    nim = np.where(img >= nv, 1, 0)
    cim = np.where(img >= cv, 1, 0) - nim
    return cim, nim

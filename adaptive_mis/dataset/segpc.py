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
import cv2
from tqdm import tqdm


class SegPC2021Dataset(Dataset):
    def __init__(self, datadir, image_size, crop_scale, crop_nucleus=True, one_hot=True, **kwargs):
        self.one_hot = one_hot
        # pre-set variables
        self.dataset_dir = datadir
        self.image_size = image_size
        self.scale = crop_scale
        self.crop_nucleus = crop_nucleus
        self.num_classes = 2
        self.class_names = ["background", "cytoplasm"]
        self.base_cache_file = f"{datadir}/cache/c{crop_nucleus}_{crop_scale}_{image_size}/"
        os.makedirs(self.base_cache_file, exist_ok=True)

        if not os.path.exists(f'{self.base_cache_file}/meta.npy'):
            self.convert()
        self.X = torch.tensor(np.load(f'{self.base_cache_file}/X.npy').astype(np.float32))
        self.Y = torch.tensor(np.load(f'{self.base_cache_file}/Y.npy').astype(np.float32))
        self.meta = np.load(f'{self.base_cache_file}/meta.npy')

        self.in_channels = self.X[0].shape[0]

    def convert(self):
        img_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(
                size=[self.image_size, self.image_size],
                interpolation=transforms.functional.InterpolationMode.BILINEAR
            )
        ])
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
            y_path = self.dataset_dir + '/[tv][ra]*/y'
        self.y_path = y_path

        X = []
        Y = []
        meta = []
        for xi, xp in enumerate(tqdm(x_path_list)):
            fn = xp.replace("\\", "/").split('/')[-1].split('.bmp')[0]
            img = self.get_orig_img(xp)
            ys = self.get_orig_msk(fn)
            if (self.crop_nucleus):
                for i, y in enumerate(ys):
                    cmsk = y[:, :, 0]
                    nmsk = y[:, :, 1]
                    timg, tnmsk, tcmsk = do_crop_nucleus(img, cmsk, nmsk, self.scale)
                    timg_with_nucleus = np.concatenate([timg, np.expand_dims(tnmsk, -1)], -1)
                    timg_with_nucleus = img_transforms(timg_with_nucleus)

                    tcmsk = img_transforms(tcmsk)
                    X.append(timg_with_nucleus.numpy())
                    Y.append(tcmsk.numpy())
                    meta.append(f'{fn}_{i}')
            else:
                msk = np.zeros((img.shape[0], img.shape[1], 2))
                for i, y in enumerate(ys):
                    msk += y * i
                X.append(img)
                Y.append(msk)
                meta.append(f'{fn}')

        X = np.stack(X)
        Y = np.stack(Y)
        meta = np.array(meta)

        # print(X.shape, X.dtype)
        # print(Y.shape, Y.dtype)
        # print(meta.shape, meta.dtype)

        np.save(f"{self.base_cache_file}/X.npy", X)
        np.save(f"{self.base_cache_file}/Y.npy", Y)
        np.save(f"{self.base_cache_file}/meta.npy", meta)

    def get_orig_img(self, xp):
        img = imageio.imread(xp)
        return img

    def get_orig_msk(self, fn):
        ys = glob.glob(f"{self.y_path}/{fn}_*.bmp")
        # print(len(ys), f"{self.y_path}/{fn}_*.bmp")
        Y = []
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
        # for y in Y:
        #     print(y.shape)
        return np.array(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]
        msk = self.Y[idx]
        meta = self.meta[idx]
        if self.one_hot:
            msk = F.one_hot(torch.squeeze(msk).to(torch.int64), num_classes=self.num_classes)
            msk = torch.moveaxis(msk, -1, 0).to(torch.float)
        sample = {'image': img, 'mask': msk, 'id': meta}
        return sample

    def summary(self):
        print(f"Number of images: {len(self)} channels={self.in_channels} classes={self.num_classes}({self.class_names}) image_size={self.image_size}")


def split_c_n(img, nv=40, cv=20):
    nim = np.where(img >= nv, 1, 0)
    cim = np.where(img >= cv, 1, 0) - nim
    return cim, nim


def make_bbox_square_in_img(bbx, img):
    # maxl=INPUT_SIZE[0]#max(bbx[2]-bbx[0],bbx[3]-bbx[1])

    maxl = bbx[2] - bbx[0], bbx[3] - bbx[1]
    maxl = max(maxl), max(maxl)

    dx, dy = (maxl[0] - (bbx[2] - bbx[0])) // 2, (maxl[1] - (bbx[3] - bbx[1])) // 2
    minx = max(bbx[0] - dx, 0)
    miny = max(bbx[1] - dy, 0)
    if minx + maxl[0] > img.shape[0]:
        minx = img.shape[0] - maxl[0]
    if miny + maxl[1] > img.shape[1]:
        miny = img.shape[1] - maxl[1]
    return minx, miny, minx + maxl[0], miny + maxl[1]


def sim_resize(image, size):
    # Resizes the image to the specified size while preserving aspect ratio

    # Get the original image dimensions
    original_height, original_width = image.shape[:2]

    # Calculate the aspect ratio of the original image
    aspect_ratio = original_width / original_height

    # Determine the target width and height while maintaining the aspect ratio
    target_width, target_height = size
    if target_width / target_height > aspect_ratio:
        target_width = int(target_height * aspect_ratio)
    else:
        target_height = int(target_width / aspect_ratio)
    # print('ssssssss',image.shape, target_width, target_height)
    # Resize the image using OpenCV
    resized_image = cv2.resize(image.astype(np.uint8), (target_width, target_height))
    return resized_image


def do_crop_nucleus(img, cim, nim, scale):
    # crop nucleus
    idxs, idys = np.nonzero(nim)
    n_bbox = ([min(idxs), min(idys), max(idxs) + 1, max(idys) + 1])

    idxs, idys = np.nonzero(cim)
    c_bbox = ([min(idxs), min(idys), max(idxs) + 1, max(idys) + 1])
    bbox = n_bbox
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    dx = int((scale * w) // 2)
    dy = int((scale * h) // 2)

    scaled_bbox = make_bbox_square_in_img([
        max(0, bbox[0] - dx), max(0, bbox[1] - dy),
        min(img.shape[0], bbox[2] + dx), min(img.shape[1], bbox[3] + dy),
    ], img)

    timg = img[scaled_bbox[0]:scaled_bbox[2], scaled_bbox[1]:scaled_bbox[3]]
    tnmsk = nim[scaled_bbox[0]:scaled_bbox[2], scaled_bbox[1]:scaled_bbox[3]]
    tcmsk = cim[scaled_bbox[0]:scaled_bbox[2], scaled_bbox[1]:scaled_bbox[3]]

    return timg, tnmsk, tcmsk

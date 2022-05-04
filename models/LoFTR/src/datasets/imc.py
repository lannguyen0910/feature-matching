import os
from torch.utils.data import Dataset
from loguru import logger
import numpy as np
import torch
import torch.nn.functional as F

from src.utils.dataset import read_megadepth_gray, read_megadepth_depth


class IMCDataset(Dataset):
    def __init__(self,
                 data,
                 mode='train',
                 min_overlap_score=0.4,
                 img_resize=None,
                 df=None,
                 img_padding=False,
                 depth_padding=False,
                 augment_fn=None,
                 **kwargs):
        """
        Args:
            data (pd.DataFrame): IMC train.csv.
            mode (str): options are ['train', 'val', 'test']
            min_overlap_score (float): how much a pair should have in common. In range of [0, 1]. Set to 0 when testing.
            img_resize (int, optional): the longer edge of resized images. None for no resize. 640 is recommended.
                                        This is useful during training with batches and testing with memory intensive algorithms.
            df (int, optional): image size division factor. NOTE: this will change the final image size after img_resize.
            img_padding (bool): If set to 'True', zero-pad the image to squared size. This is useful during training.
            depth_padding (bool): If set to 'True', zero-pad depthmap to (2000, 2000). This is useful during training.
            augment_fn (callable, optional): augments images with pre-defined visual effects.
        """
        super().__init__()
        # self.root_dir = root_dir
        self.mode = mode

        # prepare scene_info and pair_info
        if mode == 'test' and min_overlap_score != 0:
            logger.warning(
                "You are using `min_overlap_score`!=0 in test mode. Set to 0.")
            min_overlap_score = 0

        # parameters for image resizing, padding and depthmap padding
        if mode == 'train':
            assert img_resize is not None and img_padding and depth_padding
        self.img_resize = img_resize
        self.df = df
        self.img_padding = img_padding
        # the upperbound of depthmaps size in megadepth.
        self.depth_max_size = 2000 if depth_padding else None

        # for training LoFTR
        self.augment_fn = augment_fn if mode == 'train' else None
        self.coarse_scale = kwargs['coarse_scale']
        self.depth0_base_path = kwargs['depth0_base_path']
        self.depth1_base_path = kwargs['depth1_base_path']

        self.path1 = data["path1"].values
        self.path2 = data["path2"].values
        self.camerainst1 = data["camerainst1"].values
        self.camerainst2 = data["camerainst2"].values
        self.rot1 = data["rot1"].values
        self.rot2 = data["rot2"].values
        self.trans1 = data["trans1"].values
        self.trans2 = data["trans2"].values

    def __len__(self):
        return len(self.path1)

    def __getitem__(self, idx):
        # read grayscale image and mask. (1, h, w) and (h, w)
        img_name0 = self.path1[idx]
        img_name1 = self.path2[idx]

        # TODO: Support augmentation & handle seeds for each worker correctly.
        image0, mask0, scale0 = read_megadepth_gray(
            img_name0, self.img_resize, self.df, self.img_padding, None)
        # np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))
        image1, mask1, scale1 = read_megadepth_gray(
            img_name1, self.img_resize, self.df, self.img_padding, None)
        # np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))
        depth_path0 = os.path.join(self.depth0_base_path,
                                   img_name0.split("/")[-3], img_name0.split("/")[-1])
        depth_path1 = os.path.join(self.depth1_base_path,
                                   img_name1.split("/")[-3], img_name1.split("/")[-1])

        # read depth. shape: (h, w)
        if self.mode in ['train', 'val']:
            depth0 = read_megadepth_depth(
                depth_path0, pad_to=self.depth_max_size)
            depth1 = read_megadepth_depth(
                depth_path1, pad_to=self.depth_max_size)
        else:
            depth0 = depth1 = torch.tensor([])

        # read intrinsics of original size
        K_0 = torch.tensor(np.asarray([float(x) for x in self.camerainst1[idx].split(
            " ")]), dtype=torch.float).reshape(3, 3)
        K_1 = torch.tensor(np.asarray([float(x) for x in self.camerainst2[idx].split(
            " ")]), dtype=torch.float).reshape(3, 3)

        # read and compute relative poses
        R0 = self.rot1[idx].replace('{', '').replace('}', '').replace("'", "")
        R0 = np.asarray([float(x) for x in R0.split(" ")]).reshape(3, 3)
        Tv0 = self.trans1[idx].replace(
            '{', '').replace('}', '').replace("'", "")
        Tv0 = np.asarray([[float(x) for x in Tv0.split(" ")]])
        T0 = np.concatenate((R0, Tv0.T), axis=1)
        T0 = np.concatenate((T0, np.asarray([[0, 0, 0, 1]])), axis=0)
        del R0
        del Tv0
        R1 = self.rot2[idx].replace('{', '').replace('}', '').replace("'", "")
        R1 = np.asarray([float(x) for x in R1.split(" ")]).reshape(3, 3)
        Tv1 = self.trans2[idx].replace(
            '{', '').replace('}', '').replace("'", "")
        Tv1 = np.asarray([[float(x) for x in Tv1.split(" ")]])
        T1 = np.concatenate((R1, Tv1.T), axis=1)
        T1 = np.concatenate((T1, np.asarray([[0, 0, 0, 1]])), axis=0)
        del R1
        del Tv1
        T_0to1 = torch.tensor(np.matmul(T1, np.linalg.inv(T0)), dtype=torch.float)[
            :4, :4]  # (4, 4)
        T_1to0 = T_0to1.inverse()

        data = {
            'image0': image0,  # (1, h, w)
            'depth0': depth0,  # (h, w)
            'image1': image1,
            'depth1': depth1,
            'T_0to1': T_0to1,  # (4, 4)
            'T_1to0': T_1to0,
            'K0': K_0,  # (3, 3)
            'K1': K_1,
            'scale0': scale0,  # [scale_w, scale_h]
            'scale1': scale1,
            'dataset_name': 'MegaDepth',
            'scene_id': idx,
            'pair_id': idx,
            'pair_names': (img_name0, img_name1),
        }

        # for LoFTR training
        if mask0 is not None:  # img_padding is True
            if self.coarse_scale:
                [ts_mask_0, ts_mask_1] = F.interpolate(torch.stack([mask0, mask1], dim=0)[None].float(),
                                                       scale_factor=self.coarse_scale,
                                                       mode='nearest',
                                                       recompute_scale_factor=False)[0].bool()
            data.update({'mask0': ts_mask_0, 'mask1': ts_mask_1})

        torch.cuda.empty_cache()
        return data

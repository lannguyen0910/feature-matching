from loguru import logger
from torch.utils.data.dataset import Dataset

import pytorch_lightning as pl
from torch import distributed as dist
from torch.utils.data import (
    Dataset,
    DataLoader,
    ConcatDataset,
)

from src.utils.augment import build_augmentor
from src.datasets.imc import IMCDataset


class MultiSceneDataModule(pl.LightningDataModule):
    def __init__(self, args, config, data_df):
        super().__init__()

        # 1. data config
        # training and validating
        self.train_data_root = data_df[:1000000]
        self.val_data_root = data_df[1000000:1200000]

        # testing
        self.test_data_root = data_df[1200000:]

        # 2. dataset config
        # general options
        # 0.4, omit data with overlap_score < min_overlap_score
        self.min_overlap_score_test = config.DATASET.MIN_OVERLAP_SCORE_TEST
        self.min_overlap_score_train = config.DATASET.MIN_OVERLAP_SCORE_TRAIN
        # None, options: [None, 'dark', 'mobile']
        self.augment_fn = build_augmentor(config.DATASET.AUGMENTATION_TYPE)

        # MegaDepth options
        self.mgdpt_img_resize = config.DATASET.MGDPT_IMG_RESIZE  # 840
        self.mgdpt_img_pad = config.DATASET.MGDPT_IMG_PAD   # True
        self.mgdpt_depth_pad = config.DATASET.MGDPT_DEPTH_PAD   # True
        self.mgdpt_df = config.DATASET.MGDPT_DF  # 8
        # 0.125. for training loftr.
        self.coarse_scale = 1 / config.LOFTR.RESOLUTION[0]
        self.depth0_base_path = config.DATASET.DEPTH0_PATH
        self.depth1_base_path = config.DATASET.DEPTH1_PATH

        # 3.loader parameters
        self.train_loader_params = {
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'pin_memory': getattr(args, 'pin_memory', True)
        }
        self.val_loader_params = {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': args.num_workers,
            'pin_memory': getattr(args, 'pin_memory', True)
        }
        self.test_loader_params = {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': args.num_workers,
            'pin_memory': True
        }

    def setup(self, stage=None):
        """
        Setup train / val / test dataset. This method will be called by PL automatically.
        Args:
            stage (str): 'fit' in training phase, and 'test' in testing phase.
        """

        assert stage in ['fit', 'test'], "stage must be either fit or test"

        self.world_size = 1
        self.rank = 0
        logger.info("Set wolrd_size=1 and rank=0")

        if stage == 'fit':
            self.train_dataset = self._setup_dataset(
                self.train_data_root,
                mode='train',
                min_overlap_score=self.min_overlap_score_train)

            self.val_dataset = self._setup_dataset(
                self.val_data_root,
                mode='val',
                min_overlap_score=self.min_overlap_score_test)
            logger.info(f'[rank:{self.rank}] Train & Val Dataset loaded!')

        else:  # stage == 'test
            self.test_dataset = self._setup_dataset(
                self.test_data_root,
                mode='test',
                min_overlap_score=self.min_overlap_score_test)
            logger.info(f'[rank:{self.rank}]: Test Dataset loaded!')

    def _setup_dataset(self,
                       data_root,
                       mode='train',
                       min_overlap_score=0.
                       ):
        """ Setup train / val / test set"""

        dataset_builder = self._build_concat_dataset
        return dataset_builder(data_root, mode=mode, min_overlap_score=min_overlap_score)

    def _build_concat_dataset(
        self,
        data_root,
        mode,
        min_overlap_score=0.,
    ):
        datasets = []
        augment_fn = self.augment_fn if mode == 'train' else None

        datasets.append(
            IMCDataset(data_root,
                       mode=mode,
                       min_overlap_score=min_overlap_score,
                       img_resize=self.mgdpt_img_resize,
                       df=self.mgdpt_df,
                       img_padding=self.mgdpt_img_pad,
                       depth_padding=self.mgdpt_depth_pad,
                       augment_fn=augment_fn,
                       coarse_scale=self.coarse_scale,
                       depth0_base_path=self.depth0_base_path,
                       depth1_base_path=self.depth1_base_path))

        return ConcatDataset(datasets)

    def train_dataloader(self):
        """ Build training dataloader for IMC dataset. """
        return DataLoader(
            self.train_dataset, drop_last=True, **self.train_loader_params)

    def val_dataloader(self):
        """ Build validation dataloader for IMC dataset. """
        return DataLoader(
            self.val_dataset, drop_last=True, **self.val_loader_params)

    def test_dataloader(self, *args, **kwargs):
        return DataLoader(self.test_dataset, **self.test_loader_params)


def _build_dataset(dataset: Dataset, *args, **kwargs):
    return dataset(*args, **kwargs)

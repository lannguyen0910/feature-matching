from models.QuadTree.src.utils.profiler import build_profiler
from models.QuadTree.src.lightning.lightning_loftr import PL_LoFTR
from models.QuadTree.configs.config import config
from models.QuadTree.utils import *

import time
import numpy as np
import os
import csv
import cv2
import gc
import torch
import argparse

from kornia_moons.feature import *
import kornia as K
import kornia.feature as KF

import gc
gc.enable()


parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str,
                    default="data/image-matching-challenge-2022", help="Path to image-matching-challenge-2022 dataset.")
parser.add_argument("--ckpt", type=str,
                    default="models/QuadTree/pretrained/outdoor_quadtree.ckpt", help="Checkpoint path to pretrained QuadTreeAttention method in LoFTR.")

args = parser.parse_args()


def main():
    def draw_img_match(img1, img2, mkpts0, mkpts1, inliers):
        display_vertical = False

        draw_LAF_matches(
            KF.laf_from_center_scale_ori(torch.from_numpy(mkpts0).view(1, -1, 2),
                                         torch.ones(mkpts0.shape[0]).view(
                1, -1, 1, 1),
                torch.ones(mkpts0.shape[0]).view(1, -1, 1)),

            KF.laf_from_center_scale_ori(torch.from_numpy(mkpts1).view(1, -1, 2),
                                         torch.ones(mkpts1.shape[0]).view(
                1, -1, 1, 1),
                torch.ones(mkpts1.shape[0]).view(1, -1, 1)),
            torch.arange(mkpts0.shape[0]).view(-1, 1).repeat(1, 2),
            K.tensor_to_image(img1),
            K.tensor_to_image(img2),
            inliers,
            draw_dict={'inlier_color': (0.2, 1, 0.2),
                       'tentative_color': None,
                       'feature_color': (0.2, 0.5, 1), 'vertical': display_vertical})

    def inf_loftr_qta(image_fpath_1, image_fpath_2, max_image_size=qta_max_img_size, divide_coef=32):
        # resize image if we need
        img1_resized, scale1 = load_resized_image(
            image_fpath_1, max_image_size)
        img2_resized, scale2 = load_resized_image(
            image_fpath_2, max_image_size)

        # add padding -> use same image padding mask because models wants it
        pad_img1, pad_img2, pad_offset_p1, pad_offset_p2 = add_zero_padding_two_img_same(
            img1_resized, img2_resized, divide_coef)

        # save temporarily
        img1_disk_path = put_img_on_disk(pad_img1, 'qta_img1')
        img2_disk_path = put_img_on_disk(pad_img2, 'qta_img2')

        # load withr loftr
        gray_img_1 = load_loftr_image_origNEW(img1_disk_path, -1, -1)
        gray_img_2 = load_loftr_image_origNEW(img2_disk_path, -1, -1)

        batch = {'image0': gray_img_1, 'image1': gray_img_2}

        # Inference
        with torch.no_grad():
            qta_matcher.eval()
            qta_matcher.to(qta_device)

            qta_matcher(batch)
            mkpts0 = batch['mkpts0_f'].cpu().numpy()
            mkpts1 = batch['mkpts1_f'].cpu().numpy()
            mconf = batch['mconf'].cpu().numpy()

        # unpad matches
        mkpts0, mkpts1 = unpad_matches(
            mkpts0, mkpts1, pad_offset_p1, pad_offset_p2)

        # scale to original im size because we used max_image_size
        mkpts0, mkpts1 = scale_to_resized(mkpts0, mkpts1, scale1, scale2)

        # cleanup
        if os.path.exists(img1_disk_path):
            os.remove(img1_disk_path)
        if os.path.exists(img2_disk_path):
            os.remove(img2_disk_path)

        return mkpts0, mkpts1, mconf

    dry_run = True

    qta_max_img_size = 1056  # -1
    qta_torch_device = torch.device(
        'cpu' if dry_run and not torch.cuda.is_available() else 'cuda')

    qta_device = "cuda" if torch.cuda.is_available() else "cpu"
    qta_profiler_name = None
    qta_profiler = build_profiler(qta_profiler_name)
    qta_model = PL_LoFTR(config,
                         # args.ckpt_path, from scratch atm
                         pretrained_ckpt=args.ckpt,
                         profiler=qta_profiler)
    qta_matcher = qta_model.matcher
    qta_matcher.eval()
    qta_matcher.to(qta_device)

    test_samples = []
    with open(f'{args.data}/test.csv') as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
            # Skip header.
            if i == 0:
                continue
            test_samples += [row]

    if dry_run:
        for sample in test_samples:
            print(sample)

    F_dict = {}
    for i, row in enumerate(test_samples):
        sample_id, batch_id, image_1_id, image_2_id = row

        image_fpath_1 = f'{args.data}/test_images/{batch_id}/{image_1_id}.png'
        image_fpath_2 = f'{args.data}/test_images/{batch_id}/{image_2_id}.png'

        if dry_run:
            st = time.time()

        mkps1, mkps2, mconf = inf_loftr_qta(
            image_fpath_1, image_fpath_2, max_image_size=qta_max_img_size, divide_coef=32)

        if len(mkps1) > 7:
            rpt_F = 0.20
            conf_F = 0.999999
            m_iters_F = 200_000
            F, inliers_F = cv2.findFundamentalMat(
                mkps1, mkps2, cv2.USAC_MAGSAC, rpt_F, conf_F, m_iters_F)

            if dry_run:
                nd = time.time()
                print("Running time: ", nd - st, " s")

            if F is None:
                F_dict[sample_id] = np.zeros((3, 3))
                continue
            else:
                F_dict[sample_id] = F
                inliers = inliers_F
        else:
            F_dict[sample_id] = np.zeros((3, 3))
            continue

        if dry_run and i < 3:
            F_inliers_total = 0 if F is None else (inliers_F == 1).sum()
            print('inliers F', F_inliers_total)

            orig_image_1 = load_torch_kornia_image(
                image_fpath_1, qta_torch_device, -1)
            orig_image_2 = load_torch_kornia_image(
                image_fpath_2, qta_torch_device, -1)
            draw_img_match(orig_image_1, orig_image_2, mkps1, mkps2, inliers)

        try:
            # cleanup
            gc.collect()
            torch.cuda.empty_cache()
            del mkps1, mkps2
        except Exception:
            pass

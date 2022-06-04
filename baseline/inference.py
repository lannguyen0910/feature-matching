from config import ds_cfg, ot_cfg
from utils import load_torch_image, FlattenMatrix
import argparse
import csv
import time
import numpy as np
import cv2
import torch
import copy

import matplotlib.cm as cm

from LoFTR.src.utils.plotting import make_matching_figure
from LoFTR.src.loftr.loftr import LoFTR

from kornia_moons.feature import *
import kornia as K
import kornia.feature as KF

import gc
gc.enable()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str,
                        default="data/image-matching-challenge-2022", help="Path to image-matching-challenge-2022 dataset.")
    parser.add_argument("--ds_ckpt", type=str,
                        default="data/kornia-loftr/outdoor_ds.ckpt", help="Checkpoint path to pretrained dual-softmax LoFTR.")
    parser.add_argument("--ot_ckpt", type=str,
                        default="data/kornia-loftr/outdoor_ot.ckpt", help="Checkpoint path to pretrained optimal transport LoFTR.")
    args = parser.parse_args()

    test_samples = []
    F_dict = {}

    with open(f'{args.data}/test.csv') as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
            # Skip header.
            if i == 0:
                continue
            test_samples += [row]

    device = torch.device('cuda')
    ds_matcher = LoFTR(config=ds_cfg)
    ds_matcher.load_state_dict(torch.load(args.ds_ckpt)['state_dict'])
    ds_matcher = ds_matcher.to(device).eval()

    ot_matcher = LoFTR(config=ot_cfg)
    ot_matcher.load_state_dict(torch.load(args.ot_ckpt)['state_dict'])
    ot_matcher = ot_matcher.to(device).eval()

    for i, row in enumerate(test_samples):
        gc.collect()
        sample_id, batch_id, image_1_id, image_2_id = row
        # Load the images.
        st = time.time()
        image_1 = load_torch_image(
            f'{args.data}/test_images/{batch_id}/{image_1_id}.png', device)
        image_2 = load_torch_image(
            f'{args.data}/test_images/{batch_id}/{image_2_id}.png', device)

        input_dict1 = {"image0": K.color.rgb_to_grayscale(image_1),
                                     "image1": K.color.rgb_to_grayscale(image_2)}
        input_dict2 = copy.deepcopy(input_dict1)

        with torch.no_grad():
            ds_matcher(input_dict1)
            mkpts0_ds = input_dict1['mkpts0_f'].cpu().numpy()
            mkpts1_ds = input_dict1['mkpts1_f'].cpu().numpy()

        gc.collect()

        with torch.no_grad():
            ot_matcher(input_dict2)
            mkpts0_ot = input_dict2['mkpts0_f'].cpu().numpy()
            mkpts1_ot = input_dict2['mkpts1_f'].cpu().numpy()
            # mconf = input_dict2['mconf'].cpu().numpy()

        mkpts0 = np.concatenate(
            [mkpts0_ds, mkpts0_ot])
        mkpts1 = np.concatenate(
            [mkpts1_ds, mkpts1_ot])

        if len(mkpts0) > 7:
            F, inliers = cv2.findFundamentalMat(
                mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.2, 0.9999, 250000)
            assert F.shape == (3, 3), 'Malformed F?'
            F_dict[sample_id] = F
        else:
            F_dict[sample_id] = np.zeros((3, 3))
            continue
        if (i < 3):
            nd = time.time()
            print(f'Image: {i+1} | Running time: {nd - st}s')

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
                K.tensor_to_image(image_1),
                K.tensor_to_image(image_2),
                inliers > 0,
                draw_dict={'inlier_color': (0.2, 1, 0.2),
                           'tentative_color': None,
                           'feature_color': (0.2, 0.5, 1), 'vertical': False})

    torch.cuda.empty_cache()

    with open('submission.csv', 'w') as f:
        f.write('sample_id,fundamental_matrix\n')
        for sample_id, F in F_dict.items():
            f.write(f'{sample_id},{FlattenMatrix(F)}\n')


if __name__ == '__main__':
    main()

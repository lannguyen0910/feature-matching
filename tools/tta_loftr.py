from models.LoFTR.configs.baseline.config import ds_cfg, ot_cfg
from models.LoFTR.utils.baseline.utils import load_torch_image, FlattenMatrix
from models.LoFTR.src.loftr.loftr import LoFTR

import argparse
import csv
import time
import numpy as np
import cv2
import torch
from multiprocessing import Pool


from kornia_moons.feature import *
import kornia as K
import kornia.feature as KF

import gc
gc.enable()

F_dict = {}
np.random.seed(42)
DEBUG = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str,
                        default="data/image-matching-challenge-2022", help="Path to image-matching-challenge-2022 dataset.")
    parser.add_argument("--ds_ckpt", type=str,
                        default="data/kornia-loftr/outdoor_ds.ckpt", help="Checkpoint path to pretrained dual-softmax LoFTR.")
    parser.add_argument("--ot_ckpt", type=str,
                        default="data/kornia-loftr/outdoor_ot.ckpt", help="Checkpoint path to pretrained optimal transport LoFTR.")
    args = parser.parse_args()

    if DEBUG == True:
        import time
        st = time.time()

    test_samples = []

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

    alpha = 15.0
    angle: torch.tensor = torch.ones(1) * alpha
    scale: torch.tensor = torch.ones(1, 2)

    def preprocess(row):
        sample_id, batch_id, image_1_id, image_2_id = row
        image_1, orig_sz1, sz1 = load_torch_image(
            f'{args.data}/test_images/{batch_id}/{image_1_id}.png')
        center: torch.tensor = torch.ones(1, 2)
        center[..., 0] = sz1[0] / 2  # x
        center[..., 1] = sz1[1] / 2  # y
        M: torch.tensor = K.geometry.get_rotation_matrix2d(
            center, angle, scale)  # 1x2x3
        Minv: torch.tensor = K.geometry.get_rotation_matrix2d(
            center, -angle, scale)
        image_1_rot = K.geometry.warp_affine(
            image_1, M.to(image_1.device), dsize=(sz1[1], sz1[0]))
        image_1_irot = K.geometry.warp_affine(
            image_1, Minv.to(image_1.device), dsize=(sz1[1], sz1[0]))
        image_1_flip = K.geometry.transform.hflip(image_1)

        image_2, orig_sz2, sz2 = load_torch_image(
            f'{args.data}/test_images/{batch_id}/{image_2_id}.png')
        image_2_rot = K.geometry.warp_affine(
            image_2, M.to(image_2.device), dsize=(sz2[1], sz2[0]))
        image_2_irot = K.geometry.warp_affine(
            image_2, Minv.to(image_2.device), dsize=(sz2[1], sz2[0]))
        image_2_flip = K.geometry.transform.hflip(image_2)

        return {
            'sample_id': sample_id,
            'image_1_info': (orig_sz1, sz1),
            'image_2_info': (orig_sz2, sz2),
            'images_1': K.color.rgb_to_grayscale(torch.cat((image_1, image_1_rot, image_1_irot, image_1_flip))),
            'images_2': K.color.rgb_to_grayscale(torch.cat((image_2, image_2_rot, image_2_irot, image_2_flip))),
            'image_1': image_1,
            'image_2': image_2,
            'affine_mats': (M, Minv),
            'st': time.perf_counter()
        }

    def _matching(model, feat_c0, feat_c1, feat_f0, feat_f1, data):
        feat_c0, feat_c1 = model.loftr_coarse(feat_c0, feat_c1, None, None)

        # 3. match coarse-level
        model.coarse_matching(feat_c0, feat_c1, data,
                              mask_c0=None, mask_c1=None)

        # 4. fine-level refinement
        feat_f0_unfold, feat_f1_unfold = model.fine_preprocess(
            feat_f0, feat_f1, feat_c0, feat_c1, data)
        if feat_f0_unfold.size(0) != 0:  # at least one coarse level predicted
            feat_f0_unfold, feat_f1_unfold = model.loftr_fine(
                feat_f0_unfold, feat_f1_unfold)

        # 5. match fine-level
        model.fine_matching(feat_f0_unfold, feat_f1_unfold, data)

    def deep(data):
        with torch.no_grad():
            M, Minv = data['affine_mats']
            orig_sz1, sz1 = data['image_1_info']
            orig_sz2, sz2 = data['image_2_info']
            c0s, f0s = ds_matcher.backbone(data['images_1'].to(device).half())
            c1s, f1s = ds_matcher.backbone(data['images_2'].to(device).half())
            meta = {
                'bs': 1,
                'hw0_i': [sz1[1], sz1[0]],
                'hw1_i': [sz2[1], sz2[0]],
                'hw0_c': c0s.shape[2:], 'hw1_c': c1s.shape[2:],
                'hw0_f': f0s.shape[2:], 'hw1_f': f1s.shape[2:],
            }

            mkpts0, mkpts1, batch_indexes = [], [], []
            queries = [(0, 0), (1, 0), (2, 0), (3, 3), (0, 1), (0, 2)]
            for bi, (i, j) in enumerate(queries):
                feat_c0 = c0s[i:i+1]
                feat_c1 = c1s[j:j+1]
                feat_f0 = f0s[i:i+1]
                feat_f1 = f1s[j:j+1]

                feat_c0 = ds_matcher.pos_encoding(feat_c0).permute(0, 2, 3, 1)
                n, h, w, c = feat_c0.shape
                feat_c0 = feat_c0.reshape(n, -1, c)

                feat_c1 = ds_matcher.pos_encoding(feat_c1).permute(0, 2, 3, 1)
                n1, h1, w1, c1 = feat_c1.shape
                feat_c1 = feat_c1.reshape(n1, -1, c1)
                _matching(ds_matcher, feat_c0, feat_c1, feat_f0, feat_f1, meta)

                mkpts0.append(meta['mkpts0_f'].float().cpu().numpy())
                mkpts1.append(meta['mkpts1_f'].float().cpu().numpy())
                batch_indexes.append(
                    np.full(len(meta['mkpts0_f']), bi, dtype=np.int32))

        mkpts0 = np.concatenate(mkpts0)
        mkpts1 = np.concatenate(mkpts1)
        batch_indexes = np.concatenate(batch_indexes)

        mask = (batch_indexes == 1)
        mkpts0[mask] = np.concatenate([mkpts0[mask], np.ones(
            (np.sum(mask), 1))], axis=1) @ Minv[0].cpu().numpy().T
        mask = (batch_indexes == 2)
        mkpts0[mask] = np.concatenate([mkpts0[mask], np.ones(
            (np.sum(mask), 1))], axis=1) @ M[0].cpu().numpy().T
        mask = (batch_indexes == 3)
        mkpts0[mask, 0] = sz1[0] - mkpts0[mask, 0]
        mkpts1[mask, 0] = sz2[0] - mkpts1[mask, 0]
        mask = (batch_indexes == 4)
        mkpts1[mask] = np.concatenate([mkpts1[mask], np.ones(
            (np.sum(mask), 1))], axis=1) @ Minv[0].cpu().numpy().T
        mask = (batch_indexes == 5)
        mkpts1[mask] = np.concatenate([mkpts1[mask], np.ones(
            (np.sum(mask), 1))], axis=1) @ M[0].cpu().numpy().T

        mkpts0_orig = mkpts0 / sz1 * orig_sz1
        mkpts1_orig = mkpts1 / sz2 * orig_sz2

        data['mkpts0'] = mkpts0
        data['mkpts1'] = mkpts1
        data['mkpts0_orig'] = mkpts0_orig
        data['mkpts1_orig'] = mkpts1_orig
        return data

    def ransac(data):
        if len(data['mkpts0']) > 7:
            F, inliers = cv2.findFundamentalMat(
                data['mkpts0_orig'], data['mkpts1_orig'],
                cv2.USAC_MAGSAC, 0.200, 0.9999, 250000)
            inliers = inliers > 0
            assert F.shape == (3, 3), 'Malformed F?'
            data['F'] = F
            data['inliers'] = inliers
        else:
            data['F'] = np.zeros((3, 3))
            data['inliers'] = []
        return data

    processed_samples = map(preprocess, test_samples)
    matched_samples = map(deep, processed_samples)

    with Pool(2) as pool:
        for i, data in enumerate(pool.imap(ransac, matched_samples, chunksize=4)):
            F_dict[data['sample_id']] = data['F']
            inliers = data['inliers']
            nd = time.perf_counter()
            if (i < 3) and len(inliers) > 0:
                mkpts0 = data['mkpts0']
                mkpts1 = data['mkpts1']
                image_1 = data['image_1']
                image_2 = data['image_2']

                print(f"Running time: {nd-data['st']:.3f}s")
                draw_LAF_matches(
                    KF.laf_from_center_scale_ori(torch.from_numpy(mkpts0).view(1,-1, 2),
                                                torch.ones(mkpts0.shape[0]).view(1,-1, 1, 1),
                                                torch.ones(mkpts0.shape[0]).view(1,-1, 1)),

                    KF.laf_from_center_scale_ori(torch.from_numpy(mkpts1).view(1,-1, 2),
                                                torch.ones(mkpts1.shape[0]).view(1,-1, 1, 1),
                                                torch.ones(mkpts1.shape[0]).view(1,-1, 1)),
                    torch.arange(mkpts0.shape[0]).view(-1,1).repeat(1,2),
                    K.tensor_to_image(image_1),
                    K.tensor_to_image(image_2),
                    inliers,
                    draw_dict={'inlier_color': (0.2, 1, 0.2),
                            'tentative_color': None, 
                            'feature_color': (0.2, 0.5, 1), 'vertical': False})
            if DEBUG == True:
                print("The number of TTA_LoFTR keypoints: ",
                      len(mkpts0))
                print("Fundamental matrix: ")
                print(F_dict[data['sample_id']])

            gc.collect()
            torch.cuda.empty_cache()
    
    if DEBUG == True:
        ed = time.time()
        print(f"Total time: {ed - st:.2f}s")


if __name__ == '__main__':
    main()

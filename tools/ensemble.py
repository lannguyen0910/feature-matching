from models.DKM.dkm import dkm_base
from models.LoFTR.utils.baseline.utils import resize_img_loftr
from data.SuperGluePretrainedNetwork.models.matching import Matching
from data.SuperGluePretrainedNetwork.models.utils import frame2tensor
from models.SuperGluePretrainedNetwork.config import config
from models.SuperGluePretrainedNetwork.utils import *

from kornia_moons.feature import *
import kornia as K
import kornia.feature as KF

import numpy as np
import cv2
import csv
import torch
import gc
import argparse
from PIL import Image

gc.enable()

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str,
                    default="data/image-matching-challenge-2022", help="Path to image-matching-challenge-2022 dataset.")
parser.add_argument("--ds_ckpt", type=str,
                    default="data/kornia-loftr/outdoor_ds.ckpt", help="Checkpoint path to pretrained dual-softmax LoFTR.")
args = parser.parse_args()

F_dict = {}

scales_lens_superglue = [[1.2, 1200, 1.0], [
    1.2, 1600, 1.6], [0.8, 2000, 2], [1, 2800, 3]]
scales_lens_loftr = [[1.1, 1000, 1.0], [1, 1200, 1.3], [0.9, 1400, 1.6]]
w_h_muts_dkm = [[680 * 510, 1]]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = dkm_base(pretrained=True, version="v11")
matching_superglue = Matching(config).eval().to(device)
matcher_loftr = KF.LoFTR(pretrained=None)
matcher_loftr.load_state_dict(torch.load(args.ds_ckpt)['state_dict'])
matcher_loftr = matcher_loftr.to(device).eval()

np.random.seed(42)

DEBUG = True


def main():
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

    with torch.no_grad():
        for i, row in enumerate(test_samples):
            sample_id, batch_id, image_0_id, image_1_id = row

            image_0_BGR = cv2.imread(
                f'{args.data}/test_images/{batch_id}/{image_0_id}.png')
            image_1_BGR = cv2.imread(
                f'{args.data}/test_images/{batch_id}/{image_1_id}.png')

            image_0_GRAY = cv2.cvtColor(image_0_BGR, cv2.COLOR_BGR2GRAY)
            image_1_GRAY = cv2.cvtColor(image_1_BGR, cv2.COLOR_BGR2GRAY)

            # ===========================
            #           LoFTR
            # ===========================
            mkpts0_loftr_all = []
            mkpts1_loftr_all = []

            for variant_scale, max_len, enlarge_scale in scales_lens_loftr:

                image_0_resize, scale_0, isResized_0 = resize_img_loftr(
                    image_0_GRAY, max_len, enlarge_scale, variant_scale, device)
                image_1_resize, scale_1, isResized_1 = resize_img_loftr(
                    image_1_GRAY, max_len, enlarge_scale, variant_scale, device)

                if isResized_0 == False or isResized_1 == False:
                    continue

                input_dict = {"image0": image_0_resize,
                              "image1": image_1_resize}
                correspondences = matcher_loftr(input_dict)
                confidence = correspondences['confidence'].cpu().numpy()

                if len(confidence) < 1:
                    continue

                confidence_quantile = np.quantile(confidence, 0.6)
                idx = np.where(confidence >= confidence_quantile)

                mkpts0_loftr = correspondences['keypoints0'].cpu().numpy()[idx]
                mkpts1_loftr = correspondences['keypoints1'].cpu().numpy()[idx]

                if DEBUG == True:
                    print("loftr scale_0", scale_0)
                    print("loftr scale_1", scale_1)

                mkpts0_loftr = mkpts0_loftr / scale_0
                mkpts1_loftr = mkpts1_loftr / scale_1

                mkpts0_loftr_all.append(mkpts0_loftr)
                mkpts1_loftr_all.append(mkpts1_loftr)

            mkpts0_loftr_all = np.concatenate(mkpts0_loftr_all, axis=0)
            mkpts1_loftr_all = np.concatenate(mkpts1_loftr_all, axis=0)

            # ===========================
            #          SuperGlue
            # ===========================
            mkpts0_superglue_all = []
            mkpts1_superglue_all = []

            for variant_scale, max_len, enlarge_scale in scales_lens_superglue:
                image_0, scale_0, isResized_0 = resize_img_superglue(
                    image_0_GRAY, max_len, enlarge_scale, variant_scale)
                image_1, scale_1, isResized_1 = resize_img_superglue(
                    image_1_GRAY, max_len, enlarge_scale, variant_scale)

                if isResized_0 == False or isResized_1 == False:
                    break

                image_0 = frame2tensor(image_0, device)
                image_1 = frame2tensor(image_1, device)

                pred = matching_superglue(
                    {"image0": image_0, "image1": image_1})
                pred = {k: v[0].detach().cpu().numpy()
                        for k, v in pred.items()}
                kpts0, kpts1 = pred["keypoints0"], pred["keypoints1"]
                matches, conf = pred["matches0"], pred["matching_scores0"]

                valid = matches > -1
                mkpts0_superglue = kpts0[valid]
                mkpts1_superglue = kpts1[matches[valid]]

                if DEBUG == True:
                    print("superglue scale_0", scale_0)
                    print("superglue scale_1", scale_1)

                mkpts0_superglue /= scale_0
                mkpts1_superglue /= scale_1

                mkpts0_superglue_all.append(mkpts0_superglue)
                mkpts1_superglue_all.append(mkpts1_superglue)

            if len(mkpts0_superglue_all) > 0:
                mkpts0_superglue_all = np.concatenate(
                    mkpts0_superglue_all, axis=0)
                mkpts1_superglue_all = np.concatenate(
                    mkpts1_superglue_all, axis=0)

            # ===========================
            #            DKM
            # ===========================
            img0PIL = Image.fromarray(
                cv2.cvtColor(image_0_BGR, cv2.COLOR_BGR2RGB))
            img1PIL = Image.fromarray(
                cv2.cvtColor(image_1_BGR, cv2.COLOR_BGR2RGB))

            mkpts0_dkm_all = []
            mkpts1_dkm_all = []

            for w_h_mut, param in w_h_muts_dkm:

                ratio = (image_0_BGR.shape[0] + image_1_BGR.shape[0]) / (
                    image_0_BGR.shape[1] + image_1_BGR.shape[1]) * param  # 根据图0的高宽比确定计算参数

                model.w_resized = int(np.sqrt(w_h_mut / ratio))
                model.h_resized = int(ratio * model.w_resized)

                dense_matches, dense_certainty = model.match(
                    img0PIL, img1PIL, do_pred_in_og_res=True)
                dense_certainty = dense_certainty.pow(0.6)

                sparse_matches, sparse_certainty = model.sample(dense_matches, dense_certainty, max(min(
                    500, (len(mkpts0_loftr_all) + len(mkpts0_superglue_all)) // int(4 * len(w_h_muts_dkm))), 100), 0.01)
                mkpts0_dkm = sparse_matches[:, :2]
                mkpts1_dkm = sparse_matches[:, 2:]
                h, w, c = image_0_BGR.shape
                mkpts0_dkm[:, 0] = ((mkpts0_dkm[:, 0] + 1) / 2) * w
                mkpts0_dkm[:, 1] = ((mkpts0_dkm[:, 1] + 1) / 2) * h
                h, w, c = image_1_BGR.shape
                mkpts1_dkm[:, 0] = ((mkpts1_dkm[:, 0] + 1) / 2) * w
                mkpts1_dkm[:, 1] = ((mkpts1_dkm[:, 1] + 1) / 2) * h

                mkpts0_dkm_all.append(mkpts0_dkm)
                mkpts1_dkm_all.append(mkpts1_dkm)

            mkpts0_dkm_all = np.concatenate(mkpts0_dkm_all, axis=0)
            mkpts1_dkm_all = np.concatenate(mkpts1_dkm_all, axis=0)

            # ensemble of all mkpts
            mkpts0 = []
            mkpts1 = []

            if len(mkpts0_loftr_all) > 0:
                mkpts0.append(mkpts0_loftr_all)
                mkpts1.append(mkpts1_loftr_all)

            if len(mkpts0_superglue_all) > 0:
                mkpts0.append(mkpts0_superglue_all)
                mkpts1.append(mkpts1_superglue_all)

            mkpts0.append(mkpts0_dkm_all)
            mkpts1.append(mkpts1_dkm_all)

            mkpts0 = np.concatenate(mkpts0, axis=0)
            mkpts1 = np.concatenate(mkpts1, axis=0)

            if len(mkpts0) > 8:
                F, inliers = cv2.findFundamentalMat(
                    mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.15, 0.9999, 20000)
                F_dict[sample_id] = F
            else:
                F_dict[sample_id] = np.zeros((3, 3))
                continue

            if (i < 3):
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
                    cv2.cvtColor(image_0_BGR, cv2.COLOR_BGR2RGB),
                    cv2.cvtColor(image_1_BGR, cv2.COLOR_BGR2RGB),
                    inliers > 0,
                    draw_dict={'inlier_color': (0.2, 1, 0.2),
                               'tentative_color': None,
                               'feature_color': (0.2, 0.5, 1), 'vertical': False})

            if DEBUG == True:
                print("The number of loftr keypoints: ", len(mkpts0_loftr_all))
                print("The number of superglue keypoints: ",
                      len(mkpts0_superglue_all))
                print("The number of dkm keypoints: ", len(mkpts0_dkm_all))
                print("The number of all keypoints: ", len(mkpts0))
                print("Fundamental Matrix: ")
                print(F_dict[sample_id])

            gc.collect()
            torch.cuda.empty_cache()

    if DEBUG == True:
        ed = time.time()
        print(f"Total time: {ed - st:.2f}s")


if __name__ == '__main__':
    main()

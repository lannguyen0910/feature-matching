import argparse
from data.SuperGluePretrainedNetwork.models.matching import Matching
from data.SuperGluePretrainedNetwork.models.utils import frame2tensor
from models.SuperGluePretrainedNetwork.config import config
from models.SuperGluePretrainedNetwork.utils import *

import numpy as np
import cv2
import csv
import torch
import gc
gc.enable()

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str,
                    default="data/image-matching-challenge-2022", help="Path to image-matching-challenge-2022 dataset.")
args = parser.parse_args()

F_dict = {}

scales_lens_superglue = [[1.2, 1200, 1.0], [
    1.2, 1600, 1.6], [0.8, 2000, 2], [1, 2800, 3]]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
matching_superglue = Matching(config=config).eval().to(device)

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
            #          SuperGlue
            # ===========================
            mkpts0 = []
            mkpts1 = []

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

                mkpts0.append(mkpts0_superglue)
                mkpts1.append(mkpts1_superglue)

            if len(mkpts0) > 0:
                mkpts0 = np.concatenate(mkpts0, axis=0)
                mkpts1 = np.concatenate(mkpts1, axis=0)

            if len(mkpts0) > 8:
                F, inliers = cv2.findFundamentalMat(
                    mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.15, 0.9999, 20000)
                F_dict[sample_id] = F
            else:
                F_dict[sample_id] = np.zeros((3, 3))
                continue

            if DEBUG == True:
                print("The number of superglue keypoints: ",
                      len(mkpts0))
                print("The number of all keypoints: ", len(mkpts0))
                print("Fundamental matrix: ")
                print(F_dict[sample_id])

            gc.collect()
            torch.cuda.empty_cache()

    if DEBUG == True:
        ed = time.time()
        print(f"Total time: {ed - st:.2f}s")


if __name__ == '__main__':
    main()

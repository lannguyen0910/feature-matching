> [Data Description](https://www.kaggle.com/competitions/image-matching-challenge-2022/data)

Aligning photographs of the same scene is a problem of longstanding interest to computer vision researchers. Your challenge in this competition is to generate mappings between pairs of photos from various cities.

This competition uses a hidden test. When your submitted notebook is scored, the actual test data (including a sample submission) will be made available to your notebook.

<h2> Files </h2>

**train/*/calibration.csv**

- **image_id:** The image filename.

- **camera_intrinsics:** The 3×3 calibration matrix K for this image, flattened into a vector by row-major indexing.

- **rotation_matrix:** The 3×3 rotation matrix R for this image, flattened into a vector by row-major indexing.

- **translation_vector:** The translation vector T.

---

**train/*/pair_covisibility.csv**

- **pair:** A string identifying a pair of images, encoded as two image filenames (without the extension) separated by a hyphen, as key1-key2, where key1 > key2.

- **covisibility:** An estimate of the overlap between the two images. Higher numbers indicate greater overlap. We recommend using all pairs with a covisibility estimate of 0.1 or above. The procedure used to derive this number is described in Section 3.2 and Figure 5 of this paper.

- **fundamental_matrix:** The target column as derived from the calibration files. Please see the problem definition page for more details.

---

**train/scaling_factors.csv**: The poses for each scene where reconstructed via Structure-from-Motion, and are only accurate up to a scaling factor. This file contains a scalar for each scene which can be used to convert them to meters. For code examples, please refer to this notebook.


**train/*/images/:** A batch of images all taken near the same location.

**train/LICENSE.txt:** Records of the specific source of and license for each image.

---

**sample_submission.csv:** A valid sample submission.

- **sample_id:** The unique identifier for the image pair.

- **fundamental_matrix:** The target column. Please see the problem definition page for more details. The default values are randomly generated.

---

**test.csv:** Expect to see roughly 10,000 pairs of images in the hidden test set.

- **sample_id:** The unique identifier for the image pair.

- **batch_id:** The batch ID.

- **image_[1/2]_id:** The filenames of each image in the pair.

---

**test_images:** The test set. The test data comes from a different source than the train data and contains photos of mostly urban scenes with variable degrees of overlap. The two images forming a pair may have been collected months or years apart, but never less than 24 hours. Bridging this domain gap is part of the competition. The images have been resized so that the longest edge is around 800 pixels, may have different aspect ratios (including portrait and landscape), and are upright.
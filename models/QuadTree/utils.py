import cv2
import torch
import numpy as np
import kornia as K


def load_loftr_image_origNEW(fname, img_w=480, img_h=832):
    img0_raw = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    if img_w > 0 and img_h > 0:
        img0_raw = cv2.resize(img0_raw, (img_w, img_h))
    img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
    return img0


def scale_to_resized(mkpts0, mkpts1, scale1, scale2):
    # scale to original im size because we used max_image_size
    # first point
    mkpts0[:, 0] = mkpts0[:, 0] / scale1
    mkpts0[:, 1] = mkpts0[:, 1] / scale1

    # second point
    mkpts1[:, 0] = mkpts1[:, 0] / scale2
    mkpts1[:, 1] = mkpts1[:, 1] / scale2

    return mkpts0, mkpts1


def put_img_on_disk(img, output_img_tag):
    img_path_on_disk = f'/kaggle/working/{output_img_tag}.png'
    cv2.imwrite(img_path_on_disk, img)
    return img_path_on_disk


def calc_divide_size_smallest(im_size, coef):
    # select size dividable by coef
    if im_size % coef == 0:
        # already dividable, just return original
        return im_size
    return round(((im_size / coef) + 0.5)) * coef


def add_zero_padding_two_img_same(img1, img2, div_coef=32):
    img1_height, img1_width, img1_channels = img1.shape
    img2_height, img2_width, img2_channels = img2.shape

    # fit both images on canvas
    max_width = max(img1_width, img2_width)
    max_height = max(img1_height, img2_height)

    # use own width and height for image with zero-padding
    result1, offset1 = create_zero_padding_img(
        img1, max_width, max_height, img1_channels, div_coef)
    result2, offset2 = create_zero_padding_img(
        img2, max_width, max_height, img2_channels, div_coef)

    return result1, result2, offset1, offset2


def create_zero_padding_img(img, max_im_width, max_im_height, channels, div_coef=32):
    # create new image of desired size and color (black) for padding
    new_area_image_width = calc_divide_size_smallest(max_im_width, div_coef)
    new_area_image_height = calc_divide_size_smallest(max_im_height, div_coef)

    # it is different from what we have in max_im_width and max_im_height
    # max_im_height, max_im_width define an area for image and maybe include extra mask
    im_height, im_width, im_channels = img.shape
    x_offset = (new_area_image_width - im_width) // 2
    y_offset = (new_area_image_height - im_height) // 2

    im_right = x_offset + im_width  # right of image corner where ends
    im_bottom = y_offset + im_height  # right of image corner where ends

    color = (0, 0, 0)
    result = np.full((new_area_image_height, new_area_image_width,
                      im_channels), color, dtype=np.uint8)

    # copy img image into center of result image
    result[y_offset:im_bottom, x_offset:im_right] = img

    # return image and x,y of old image in a new image (frame)
    return result, (x_offset, y_offset)


def add_zero_padding(img, div_coef=32):
    old_image_height, old_image_width, channels = img.shape
    return create_zero_padding_img(img, old_image_width, old_image_height, channels, div_coef)


def unpad_matches(mkpts0, mkpts1, offset_point1, offset_point2):
    offset_x1, offset_y1 = offset_point1
    offset_x2, offset_y2 = offset_point2

    # remove offeset
    mkpts0[:, 0] = mkpts0[:, 0] - offset_x1
    mkpts0[:, 1] = mkpts0[:, 1] - offset_y1

    mkpts1[:, 0] = mkpts1[:, 0] - offset_x2
    mkpts1[:, 1] = mkpts1[:, 1] - offset_y2
    return mkpts0, mkpts1


def load_resized_image(fname, max_image_size):
    img = cv2.imread(fname)
    if max_image_size == -1:
        # no resize
        return img, 1.0
    scale = max_image_size / max(img.shape[0], img.shape[1])
    w = int(img.shape[1] * scale)
    h = int(img.shape[0] * scale)
    img = cv2.resize(img, (w, h))
    return img, scale


def load_torch_kornia_image(fname, device, max_image_size):
    img, _ = load_resized_image(fname, max_image_size)
    img = K.image_to_tensor(img, False).float() / 255.
    img = K.color.bgr_to_rgb(img)
    orig_img_device = img.to(device)
    return orig_img_device

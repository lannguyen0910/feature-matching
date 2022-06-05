import cv2


def resize_img_superglue(img, max_len, enlarge_scale, variant_scale):
    if max_len == -1:
        scale = 1
    else:
        scale = max(max_len, max(
            img.shape[0], img.shape[1]) * enlarge_scale) / max(img.shape[0], img.shape[1])
    w = int(round(img.shape[1] * scale))
    h = int(round(img.shape[0] * scale))

    isResized = False
    if w >= h:
        if int(h * variant_scale) <= w:
            isResized = True
            h = int(h * variant_scale)
    else:
        if int(w * variant_scale) <= h:
            isResized = True
            w = int(w * variant_scale)
    img_resize = cv2.resize(img, (w, h))
    return img_resize, (w / img.shape[1], h / img.shape[0]), isResized

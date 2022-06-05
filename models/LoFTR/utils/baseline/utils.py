import cv2
import kornia as K

def FlattenMatrix(M, num_digits=8):
    '''Convenience function to write CSV files.'''

    return ' '.join([f'{v:.{num_digits}e}' for v in M.flatten()])


def load_torch_image(fname, device):
    img = cv2.imread(fname)
    scale = 840 / max(img.shape[0], img.shape[1])
    w = int(img.shape[1] * scale)
    h = int(img.shape[0] * scale)
    img = cv2.resize(img, (w, h))
    img = K.image_to_tensor(img, False).float() / 255.
    img = K.color.bgr_to_rgb(img)
    return img.to(device)

def resize_img_loftr(img, max_len, enlarge_scale, variant_scale, device):
    if max_len == -1:
        scale = 1
    else:
        scale = max(max_len, max(img.shape[0], img.shape[1]) * enlarge_scale) / max(img.shape[0], img.shape[1])
    w = int(round(img.shape[1] * scale) / 8) * 8
    h = int(round(img.shape[0] * scale) / 8) * 8
    
    isResized = False
    if w >= h:
        if int(h * variant_scale) <= w:
            isResized = True
            h = int(h * variant_scale / 8) * 8
    else:
        if int(w * variant_scale) <= h:
            isResized = True
            w = int(w * variant_scale / 8) * 8
    img_resize = cv2.resize(img, (w, h)) 
    img_resize = K.image_to_tensor(img_resize, False).float() / 255.
    
    return img_resize.to(device), (w / img.shape[1], h / img.shape[0]), isResized

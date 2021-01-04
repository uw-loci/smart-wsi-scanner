from skimage import io, img_as_float, img_as_ubyte, exposure, transform, color
import numpy as np
import glob

def thres_saturation(img, t=5):
    img = color.rgb2hsv(img)
    h, w, c = img.shape
    sat_img = img[:, :, 1]
    sat_img = img_as_ubyte(sat_img)
    ave_sat = np.sum(sat_img)/(h*w)
    return ave_sat >= t

def estimate_bg(path):
    bgs = glob.glob(os.path.join(path, '*.tif'))
    sum_img = img_as_float(io.imread(bgs[0]))
    for bg in bgs:
        bg_data = np.array(img_as_float(io.imread(bg)))
        sum_img = sum_img + bg_data
    sum_img = exposure.rescale_intensity(sum_img / len(bgs), out_range=(0, 1))
    return sum_img

def white_balance(img, bg, gain=0.8):
    img = img_as_float(img)
    bg = img_as_float(bg)
    r = np.mean(bg[:, :, 0])
    g = np.mean(bg[:, :, 1])
    b = np.mean(bg[:, :, 2])
    mm = max(r, g, b)
    img[:, :, 0] = np.clip(img[:, :, 0] * mm / r * gain, 0, 1)
    img[:, :, 1] = np.clip(img[:, :, 1] * mm / g * gain, 0, 1)
    img[:, :, 2] = np.clip(img[:, :, 2] * mm / b * gain, 0, 1)
    return img

def flat_field(img, bg, gain=1.25):
    img = img_as_float(img)
    bg = img_as_float(bg)
    r = np.mean(bg[:, :, 0])
    g = np.mean(bg[:, :, 1])
    b = np.mean(bg[:, :, 2])
    img[:, :, 0] = 0.95 * exposure.rescale_intensity(np.clip(np.divide(img[:, :, 0], bg[:, :, 0] + 0.00) * r * gain, 0, 1), in_range=(0, 0.85), out_range=(0, 1))
    img[:, :, 1] = 0.95 * exposure.rescale_intensity(np.clip(np.divide(img[:, :, 1], bg[:, :, 1] + 0.00) * g * gain, 0, 1), in_range=(0, 0.85), out_range=(0, 1))
    img[:, :, 2] = 0.95 * exposure.rescale_intensity(np.clip(np.divide(img[:, :, 2], bg[:, :, 2] + 0.00) * b * gain, 0, 1), in_range=(0, 0.85), out_range=(0, 1))
    return img
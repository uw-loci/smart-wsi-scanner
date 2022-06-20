import os, glob, sys

from skimage import io, img_as_float, color, transform, exposure
from skimage.filters import threshold_mean
from skimage.util import view_as_windows, crop
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def bounding_image(config, image, box=None):
        img_g = color.rgb2gray(img)
        img_t = transform.resize(img_g, (int(img_g.shape[0]/100), int(img_g.shape[1]/100)), anti_aliasing=None, order=0)
        img_ct = crop(img_t, ((10, 10), (20, 20)))
        img_t = transform.resize(img_ct, (img_t.shape[0], img_t.shape[1]))
        if box is None:
            thresh = threshold_mean(img_t)
            img_d = 1 - (img_t > thresh) # 1 at valid pixels
            imgh = np.sum(img_d, axis = 1) # a column
            imgw = np.sum(img_d, axis = 0) # a row
            imgh_m = np.mean(imgh)
            imgw_m = np.mean(imgw)
            box_s_x = min(np.argwhere(imgw>imgw_m*1.5))[0] * 100
            box_s_y = min(np.argwhere(imgh>imgh_m*1.5))[0] * 100
            box_e_x = max(np.argwhere(imgw>imgw_m*1.5))[0] * 100
            box_e_y = max(np.argwhere(imgh>imgh_m*1.5))[0] * 100
            start = config["slide-box"]
            low_box_bounded = (config["slide-box"][0] + config["pixel-size-bf-4x"] * box_s_x,
                               config["slide-box"][1] + config["pixel-size-bf-4x"] * box_s_y,
                               config["slide-box"][0] + config["pixel-size-bf-4x"] * box_e_x,
                               config["slide-box"][1] + config["pixel-size-bf-4x"] * box_e_y,
                              )
        else:
            box_s_x = box[0] / config["pixel-size-bf-4x"]
            box_s_y = box[1] / config["pixel-size-bf-4x"]
            box_e_x = box[2] / config["pixel-size-bf-4x"]
            box_e_y = box[3] / config["pixel-size-bf-4x"]
            low_box_bounded = (box[0], box[1], box[2], box[3])
        fig, ax = plt.subplots(1)
        ax.imshow(img_t, cmap='gray')
        rect = patches.Rectangle((int(box_s_x/100),int(box_s_y/100)), int((box_e_x-box_s_x)/100), int((box_e_y-box_s_y)/100), linewidth=2,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
        plt.show()
        return low_box_bounded # bounding box in real stage position (x, y, x, y)
    
def is_background(img, t=0.3, tt=0.32):
    patch_h = int(img.shape[0]/8)
    patch_w = int(img.shape[1]/8)
    img = color.rgb2hsv(img)
    img_windows = np.squeeze(view_as_windows(img, (patch_h, patch_w, 3), step=(patch_h, patch_w, 3)))
    img_windows = np.reshape(img_windows, (img_windows.shape[0]*img_windows.shape[1], patch_h, patch_w, 3)) # nxm, 256, 256, 3
    img_max = np.max(img_windows, axis=0) # 256x256x3
    img_min = np.min(img_windows, axis=0) # 256x256x3
    sat_img = img_max[:, :, 1]
    bright_img = 1-img_min[:, :, 2]
    ave_sat = np.sum(sat_img)/(patch_h*patch_w)
    ave_bright = np.sum(bright_img)/(patch_h*patch_w)
#     print(ave_sat)
#     print(ave_bright)
    return ave_sat < t and ave_bright < tt

def estimate_background(config, save_path, acq_name, position_list=None, mda=True):
    sum_img = np.zeros((config["camera-resolution"][0], config["camera-resolution"][1], 3))
    sum_count = 0
    if mda:
        data_path = glob.glob(save_path+'/'+acq_name+'*')[-1]
        dataset = Dataset(data_path)
    else:
        image_list = glob.glob(os.path.join(glob.glob(save_path+'/'+acq_name+'*')[-1], '*.tiff'))
    bg_stack = []
    for pos_row in range(position_list.shape[0]):
        for pos_col in range(position_list.shape[1]):
            if mda:
                img = dataset.read_image(position=pos_row*position_list.shape[1]+pos_col)
            else:
                img = io.imread(image_list[pos_row*position_list.shape[1]+pos_col])
            if is_background(img):
                bg_stack.append(img)
                sum_count = sum_count + 1
    bg_stack= np.stack(bg_stack)
    median = np.median(bg_stack, axis=0)
    median = img_as_float(median)
    return median


def white_balance(img, bg, gain=1.0):
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

def flat_field(img, bg, gain=1):
    img = img_as_float(img)
    bg = img_as_float(bg)
    r = np.mean(bg[:, :, 0])
    g = np.mean(bg[:, :, 1])
    b = np.mean(bg[:, :, 2])
    img[:, :, 0] = 1 * exposure.rescale_intensity(np.clip(np.divide(img[:, :, 0], bg[:, :, 0] + 0.00) * r * gain, 0, 1), in_range=(0, 0.95), out_range=(0, 1))
    img[:, :, 1] = 1 * exposure.rescale_intensity(np.clip(np.divide(img[:, :, 1], bg[:, :, 1] + 0.00) * g * gain, 0, 1), in_range=(0, 0.95), out_range=(0, 1))
    img[:, :, 2] = 1 * exposure.rescale_intensity(np.clip(np.divide(img[:, :, 2], bg[:, :, 2] + 0.00) * b * gain, 0, 1), in_range=(0, 0.95), out_range=(0, 1))
    return img

def montage_blend(outputs, patch_size, padded_image_size, overlap, rows, cols):
    outputs = outputs.reshape(rows, cols, patch_size[0], patch_size[1], patch_size[2])
    canvas_size = padded_image_size
    canvas = np.zeros(canvas_size)
    decay = np.linspace(1, 0, overlap)
    grow = np.linspace(0, 1, overlap)
    mask_left = np.ones(patch_size)
    mask_top = np.ones(patch_size)
    mask_right = np.ones(patch_size)
    mask_bottom = np.ones(patch_size)
    mask_left[:, 0:overlap] = np.repeat(np.repeat(grow.reshape(1, -1), patch_size[0], axis=0).reshape(patch_size[0], -1, 1), patch_size[2], 2)
    mask_top[0:overlap, :] = np.repeat(np.repeat(grow.reshape(-1, 1), patch_size[1], axis=1).reshape(-1, patch_size[1], 1), patch_size[2], 2)
    mask_right[:, -overlap:] = np.repeat(np.repeat(decay.reshape(1, -1), patch_size[0], axis=0).reshape(patch_size[0], -1, 1), patch_size[2], 2)
    mask_bottom[-overlap:, :] = np.repeat(np.repeat(decay.reshape(-1, 1), patch_size[1], axis=1).reshape(-1, patch_size[1], 1), patch_size[2], 2)
    for id_y in range(rows):
        for id_x in range(cols):
            mask = np.ones(patch_size)
            cursor_x = id_x * step_size
            cursor_y = id_y * step_size
            if id_x != 0:
                mask = mask * mask_left
            if id_x != (cols-1):
                mask = mask * mask_right
            if id_y != 0:
                mask = mask * mask_top
            if id_y != (rows-1):
                mask = mask * mask_bottom
            paint = np.zeros(canvas_size)
            paint[cursor_y:cursor_y+patch_size[0], cursor_x:cursor_x+patch_size[1], :] = outputs[id_y, id_x] * mask
            canvas = canvas + paint
    return canvas


def lsm_process_fn(config):
    if config["snr-level"]=="low":
        def img_process_fn(image, metadata):
            image = exposure.rescale_intensity(image, in_range=(6000, 8500), out_range=(0, 1))
            image = exposure.adjust_gamma(image, 0.5)
            image = img_as_uint(image)
            if config["enhancement-type"] is not None:
                image = config["enhancer"].compute(image)
                image = img_as_uint(image)
            return image, metadata
    if config["snr-level"]=="mid":
        def img_process_fn(image, metadata):
            image = exposure.rescale_intensity(image, in_range=(6300, 9200), out_range=(0, 1))
            image = exposure.adjust_gamma(image, 0.6)
            image = img_as_uint(image)
            if config["enhancement-type"] is not None:
                image = config["enhancer"].compute(image)
                image = img_as_uint(image)
            return image, metadata
    if config["snr-level"]=="high":
        def img_process_fn(image, metadata):
            image = exposure.rescale_intensity(image, in_range=(6300, 11000), out_range=(0, 1))
            image = exposure.adjust_gamma(image, 0.8)
            image = img_as_uint(image)
            if config["enhancement-type"] is not None:
                image = config["enhancer"].compute(image)
                image = img_as_uint(image)
            return image, metadata
    if config["snr-level"]=="extreme":
        def img_process_fn(image, metadata):
            image = exposure.rescale_intensity(image, in_range=(6000, 14000), out_range=(0, 1))
            image = exposure.adjust_gamma(image, 0.9)
            image = img_as_uint(image)
            if config["enhancement-type"] is not None:
                image = config["enhancer"].compute(image)
                image = img_as_uint(image)
            return image, metadata
    return None
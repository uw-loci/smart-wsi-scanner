import os, glob, shutil, sys, copy, time, json, copy
from skimage import io, img_as_ubyte, img_as_float, color, transform, exposure
from skimage.filters import threshold_mean
from skimage.measure import shannon_entropy
from skimage.util import view_as_windows, crop
import imagej
from PIL import Image
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
    #         print(imgh.tolist())
    #         print(imgh_m)
            box_s_x = min(np.argwhere(imgw>imgw_m*1.5))[0] * 100
            box_s_y = min(np.argwhere(imgh>imgh_m*1.5))[0] * 100
            box_e_x = max(np.argwhere(imgw>imgw_m*1.5))[0] * 100
            box_e_y = max(np.argwhere(imgh>imgh_m*1.5))[0] * 100
            start = config["slide-start"]
            low_box_bounded = (config["slide-start"][0] + config["pixel-size-bf-4x"] * box_s_x,
                               config["slide-start"][1] + config["pixel-size-bf-4x"] * box_s_y,
                               config["slide-start"][0] + config["pixel-size-bf-4x"] * box_e_x,
                               config["slide-start"][1] + config["pixel-size-bf-4x"] * box_e_y,
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
    
def is_background(img, t=10):
#     img = transform.resize(img, (1024, 1024))
    img = color.rgb2hsv(img)
    img_windows = np.squeeze(view_as_windows(img, (256, 256, 3), step=256))
    empty=True
    img_windows = np.reshape(img_windows, (img_windows.shape[0]*img_windows.shape[1], 256, 256, 3)) # nxm, 256, 256, 3
    img_max = np.max(img_windows, axis=0) # 256x256x3
    sat_img = img_max[:, :, 1]
    ave_sat = np.sum(sat_img)/(256*256)
#     print(ave_sat)
    return ave_sat < 0.35
#     img_gray = color.rgb2gray(color.rgba2rgb(img))
#     etp = shannon_entropy(img_gray)
#     img = color.rgb2hsv(img)
#     h, w, c = img.shape
#     sat_img = img[:, :, 1]
#     sat_img = img_as_ubyte(sat_img)
#     ave_sat = np.sum(sat_img)/(h*w)
#     return ave_sat < 2*t and etp < t

def estimate_background(config, save_path, acq_name, position_list=None, mda=True):
    sum_img = np.zeros((config["camera-resolution"][0], config["camera-resolution"][1], 3))
    sum_count = 0
    if mda:
        data_path = glob.glob(save_path+'/'+acq_name+'*')[-1]
        dataset = Dataset(data_path)
    else:
        image_list = glob.glob(os.path.join(glob.glob(save_path+'/'+acq_name+'*')[-1], '*.tiff'))
    for pos_row in range(position_list.shape[0]):
        for pos_col in range(position_list.shape[1]):
            if mda:
                img = dataset.read_image(position=pos_row*position_list.shape[1]+pos_col)
            else:
                img = io.imread(image_list[pos_row*position_list.shape[1]+pos_col])
            if is_background(img):
                sum_img = np.array(img_as_float(img)) + sum_img
                sum_count = sum_count + 1
    return sum_img / sum_count

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
    img[:, :, 0] = 1 * exposure.rescale_intensity(np.clip(np.divide(img[:, :, 0], bg[:, :, 0] + 0.00) * r * gain, 0, 1), in_range=(0, 0.85), out_range=(0, 1))
    img[:, :, 1] = 1 * exposure.rescale_intensity(np.clip(np.divide(img[:, :, 1], bg[:, :, 1] + 0.00) * g * gain, 0, 1), in_range=(0, 0.85), out_range=(0, 1))
    img[:, :, 2] = 1 * exposure.rescale_intensity(np.clip(np.divide(img[:, :, 2], bg[:, :, 2] + 0.00) * b * gain, 0, 1), in_range=(0, 0.85), out_range=(0, 1))
    return img
    
def stitching(config, ij, save_path, acq_name, mod='bf', mag='4x', mda=True, z_stack=False, position_list=None, flip_x=False, flip_y=False, correction=False, background_image=None):
    position_list_flat = position_list.reshape(-1, 2)
    stitch_folder = os.path.join('data/stitching/tiles', acq_name)
    os.makedirs(stitch_folder, exist_ok=True)
    out_folder = os.path.join('data/stitching/stitched', acq_name)
    os.makedirs(out_folder, exist_ok=True)
    if mod == 'bf':
        if mag == '20x':
            pixel_size = config["pixel-size-bf-20x"]
        if mag == '4x':
            pixel_size = config["pixel-size-bf-4x"]
    else:
        pixel_size = config["pixel-size-shg"]
    if mda:
        data_path = glob.glob(save_path+'/'+acq_name+'*')[-1]
        dataset = Dataset(data_path)
    else:
        image_list = glob.glob(os.path.join(glob.glob(save_path+'/'+acq_name+'*')[-1], '*.tiff'))
        image_list.sort(key=lambda x: os.path.getmtime(x))
        image_list = image_list[0:-1]
    if correction is True and background_image is not None:
        bg_img = white_balance(copy.deepcopy(background_image), copy.deepcopy(background_image))
    with open(os.path.join(stitch_folder, 'TileConfiguration.txt'), 'w') as text_file:
        if z_stack:
            print('dim = {}'.format(3), file=text_file)
        else:
            print('dim = {}'.format(2), file=text_file)
        for pos_row in range(position_list.shape[0]):
            for pos_col in range(position_list.shape[1]):
                x = int(position_list[pos_row, pos_col, 0] / pixel_size)
                y = int(position_list[pos_row, pos_col, 1] / pixel_size)
                if z_stack:
                    print('{}_{}.tiff; ; ({}, {}, {})'.format(pos_row, (position_list.shape[1] - pos_col), x, y, 0), file=text_file)
                    z_idx = 0
                    img_z_list = []
                    while(dataset.has_image(position=pos_row*position_list.shape[1]+pos_col, z=z_idx)):
                        img_z_list.append(dataset.read_image(position=pos_row*position_list.shape[1]+pos_col, z=z_idx))
                        z_idx = z_idx+1
                    img = np.stack(img_z_list, axis=0)
                else:    
                    print('{}_{}.tiff; ; ({}, {})'.format(pos_row, (position_list.shape[1] - pos_col), x, y), file=text_file)
                    if mda:
                        img = dataset.read_image(position=pos_row*position_list.shape[1]+pos_col)
                    else:
                        img = io.imread(image_list[pos_row*position_list.shape[1]+pos_col])
                    if correction is True and background_image is not None:
                        img = white_balance(img, background_image)
                        img = flat_field(img, bg_img)
                if flip_y:
                    img = img[::-1, :]
                if flip_x:
                    img = img[:, ::-1]
                io.imsave(stitch_folder+'/{}_{}.tiff'.format(pos_row, (position_list.shape[1] - pos_col)), img_as_ubyte(img))
    sys.stdout.write('stitching, please wait...')
    temp_channel_folder = 'data/stitching/channel_temp'
    os.makedirs(temp_channel_folder, exist_ok=True)
    params = {'type': 'Positions from file', 'order': 'Defined by TileConfiguration', 
            'directory':stitch_folder, 'ayout_file': 'TileConfiguration.txt', 
            'fusion_method': 'Linear Blending', 'regression_threshold': '0.30', 
            'max/avg_displacement_threshold':'2.50', 'absolute_displacement_threshold': '3.50', 
            'compute_overlap':False, 'computation_parameters': 'Save computation time (but use more RAM)', 
            'image_output': 'Write to disk', 'output_directory': temp_channel_folder}
    plugin = "Grid/Collection stitching"
    ij.py.run_plugin(plugin, params)
    if mod == 'bf':
        list_channels = [f for f in os.listdir(temp_channel_folder)]
        if z_stack:
            fused_list = []
            for channel in list_channels:
                fused_list.append(io.imread(os.path.join(temp_channel_folder, channel)))
            img_to_save = np.stack(fused_list, axis=0)
        else:
            if len(list_channels) == 1:
                img_to_save = io.imread(os.path.join(temp_channel_folder, list_channels[0]))
            else:
                c1 = io.imread(os.path.join(temp_channel_folder, list_channels[0]))
                c2 = io.imread(os.path.join(temp_channel_folder, list_channels[1]))
                c3 = io.imread(os.path.join(temp_channel_folder, list_channels[2]))
                img_to_save = np.stack((c1, c2, c3)).transpose((1, 2, 0))
        io.imsave(os.path.join(out_folder, 'fused.tiff'), img_as_ubyte(img_to_save))
    shutil.rmtree(temp_channel_folder)
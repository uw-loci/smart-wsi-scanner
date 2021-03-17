import os, glob, shutil, sys, copy, time, json, copy
from pycromanager import Dataset
from skimage import io, img_as_ubyte, img_as_float, img_as_uint, color, transform, exposure
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
    
def is_background(img, t=0.18):
#     img = transform.resize(img, (1024, 1024))
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
    return ave_sat < t and ave_bright < t*2

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
    img[:, :, 0] = 1 * exposure.rescale_intensity(np.clip(np.divide(img[:, :, 0], bg[:, :, 0] + 0.00) * r * gain, 0, 1), in_range=(0, 0.95), out_range=(0, 1))
    img[:, :, 1] = 1 * exposure.rescale_intensity(np.clip(np.divide(img[:, :, 1], bg[:, :, 1] + 0.00) * g * gain, 0, 1), in_range=(0, 0.95), out_range=(0, 1))
    img[:, :, 2] = 1 * exposure.rescale_intensity(np.clip(np.divide(img[:, :, 2], bg[:, :, 2] + 0.00) * b * gain, 0, 1), in_range=(0, 0.95), out_range=(0, 1))
    return img
    
def stitching(config, ij, save_path, acq_name, mag='4x', mda=True, z_stack=False, position_list=None, flip_x=False, flip_y=False, rotate=None, correction=False, background_image=None, move_stitched_image=True):
#     position_list_flat = position_list.reshape(-1, 2)
    stitch_folder = os.path.join('data/stitching/tiles', acq_name)
    os.makedirs(stitch_folder, exist_ok=True)
    out_folder = os.path.join('data/stitching/stitched', acq_name)
    os.makedirs(out_folder, exist_ok=True)
    slide_folder = os.path.join('data/slides', mag)
    os.makedirs(slide_folder, exist_ok=True)
    if mag == '20x':
        pixel_size = config["pixel-size-bf-20x"]
    if mag == '4x':
        pixel_size = config["pixel-size-bf-4x"]
    if mag == 'mp':
        pixel_size = config["pixel-size-shg"]
    if mda:
        data_path = glob.glob(save_path+'/'+acq_name+'*')[-1]
        dataset = Dataset(data_path)
    else:
        image_list = glob.glob(os.path.join(glob.glob(save_path+'/'+acq_name+'*')[-1], '*.tiff'))
        image_list.sort(key=lambda x: os.path.getmtime(x))
    if correction is True and background_image is not None:
        image_list = image_list[0:-1]
        bg_img = white_balance(copy.deepcopy(background_image), copy.deepcopy(background_image))
    with open(os.path.join(stitch_folder, 'TileConfiguration.txt'), 'w') as text_file:
        if z_stack:
            print('dim = {}'.format(3), file=text_file)
        else:
            print('dim = {}'.format(2), file=text_file)
        for pos in range(position_list.shape[0]):
            x = int(position_list[pos, 0] / pixel_size)
            y = int(position_list[pos, 1] / pixel_size)
            if z_stack:
                print('{}.tiff; ; ({}, {}, {})'.format(pos, x, y, 0), file=text_file)
                z_idx = 0
                img_z_list = []
                while(dataset.has_image(position=pos, z=z_idx)):
                    img = dataset.read_image(position=pos, z=z_idx)
                    if correction == 'high' and background_image is None:
                        img = exposure.rescale_intensity(img, in_range=(6000, 14000), out_range=(0, 1))
                        img = exposure.adjust_gamma(img, 0.9)
                    if correction == 'mid' and background_image is None:
                        img = exposure.rescale_intensity(img, in_range=(6150, 11000), out_range=(0, 1))
                        img = exposure.adjust_gamma(img, 0.8)
                    if correction == 'low' and background_image is None:
                        img = exposure.rescale_intensity(img, in_range=(6450, 9200), out_range=(0, 1))
                        img = exposure.adjust_gamma(img, 0.6)
                    if correction is None:
                        img = img_as_float(img)
                    if rotate is not None:
                        img = transform.rotate(np.array(img), rotate)
                    img_z_list.append(img)
                    z_idx = z_idx+1
                img = np.stack(img_z_list, axis=0)
            else:    
                print('{}.tiff; ; ({}, {})'.format(pos, x, y), file=text_file)
                if mda:
                    img = dataset.read_image(position=pos)
                else:
                    img = io.imread(image_list[pos])
                if correction is True and background_image is not None:
                    img = white_balance(img, background_image)
                    img = flat_field(img, bg_img)
                if rotate is not None:
                    img = transform.rotate(img, rotate)
#             print(img)
            if flip_y:
                img = img[::-1, :]
            if flip_x:
                img = img[:, ::-1]
            sys.stdout.write('\r Processing tiles: {}/{}'.format(pos+1, position_list.shape[0]))
            io.imsave(stitch_folder+'/{}.tiff'.format(pos), img_as_ubyte(img))
    sys.stdout.write('\n stitching, please wait...')
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
    
    if mag=='20x' or mag=='4x': 
        macro = """
            #@ String inDir
            #@ String outDir
            slices = getFileList(inDir);
            for (i=0;i<lengthOf(slices);i=i+1) {
                filePath = inDir + '/' + slices[i];
                open(filePath);
            }
            run("Merge Channels...", "c1=img_t1_z1_c1 c2=img_t1_z1_c2 c3=img_t1_z1_c3 create");
            saveAs("Tiff", outDir);
            close("*");
            """
    if mag=='mp':
        macro = """
            #@ String inDir
            #@ String outDir
            slices = getFileList(inDir);
            for (i=0;i<lengthOf(slices);i=i+1) {
                filePath = inDir + '/' + slices[i];
                open(filePath);
            }
            run("Images to Stack", "name=Stack title=[] use");;
            saveAs("Tiff", outDir);
            run("Z Project...", "projection=[Max Intensity]");
            saveAs("Tiff", outDir);
            close("*");
            """

    if move_stitched_image:
        args = {
                'inDir' : os.path.join(os.getcwd(), temp_channel_folder),
                'outDir' : os.path.join(os.getcwd(), os.path.join(slide_folder, acq_name+'.tif'))
            }
    else:
        args = {
                'inDir' : os.path.join(os.getcwd(), temp_channel_folder),
                'outDir' : os.path.join(os.getcwd(), os.path.join(out_folder, 'fused.tif'))
            }

    result = ij.py.run_macro(macro, args)
    shutil.rmtree(temp_channel_folder)
#     shutil.rmtree(stitch_folder)

def lsm_process_fn(config):
    if config["snr-level"]=="low":
        def img_process_fn(image, metadata):
            image = exposure.rescale_intensity(image, in_range=(6600, 9200), out_range=(0, 1))
            image = exposure.adjust_gamma(image, 0.6)
            image = img_as_uint(image)
            return image, metadata
    if config["snr-level"]=="mid":
        def img_process_fn(image, metadata):
            image = exposure.rescale_intensity(image, in_range=(6200, 11000), out_range=(0, 1))
            image = exposure.adjust_gamma(image, 0.8)
            image = img_as_uint(image)
            return image, metadata
    if config["snr-level"]=="high":
        def img_process_fn(image, metadata):
            image = exposure.rescale_intensity(image, in_range=(6000, 14000), out_range=(0, 1))
            image = exposure.adjust_gamma(image, 0.9)
            image = img_as_uint(image)
            return image, metadata
    return img_process_fn
from .lsm_enhancer import LSMEnhancer
from .bf_enhancer import BFEnhancer
from .acquisition_utils import limit_stage, distance
import numpy as np
from .image_utils import is_background, estimate_background, white_balance, flat_field
from pycromanager import Acquisition, multi_d_acquisition_events
import matplotlib.pyplot as plt
from IPython import display
import glob, os, sys, copy, json
from skimage import color
from skimage import io, img_as_ubyte, img_as_float, img_as_uint, color, transform, exposure
from skimage.filters import threshold_mean, sobel
from skimage.measure import shannon_entropy
from skimage.util import view_as_windows, crop
import scipy
import pandas as pd
from .image_utils import lsm_process_fn
from tqdm import tqdm

class SPAcquisition:
    def __init__(self, 
            config, 
            mmcore, 
            mmstudio, 
            bf_4x_bg=None, 
            bf_20x_bg=None
        ):
        
        print('Load LSM presets.')
        config = self.config_preset(config)
        if config['lsm-enhancer'] is not None:
            print('Configuring LSM enhancer')
            self.lsm_enhancer = LSMEnhancer(config)
        elif config['bf-enhancer'] is not None:
            print('Configuring BF enhancer')
            self.bf_enhancer = BFEnhancer(config)
        
        self.config = config
        self.core = mmcore
        self.studio = mmstudio
        self.last_img = None
        self.bf_4x_bg = bf_4x_bg
        self.bf_20x_bg = bf_20x_bg
        self.position_list_4x = None
        self.position_list_20x = None
        self.position_list_mp = None
        self.z_list_4x = None
        self.z_list_20x = None
        self.z_list_mp = None
        self.bf_process_fn = None
        self.lsm_process_fn = None
        
            
    def config_preset(self, config):
        # if config['exposure-level']=="low":
        #     config['lsm-scan-rate'] = '500000.0000'
        #     config['lsm-pc-power'] = 0.30
        #     config['lsm-pmt-gain'] = 0.40
        # if config['exposure-level']=="mid":
        #     config['lsm-scan-rate'] = '400000.0000'
        #     config['lsm-pc-power'] = 0.45
        #     config['lsm-pmt-gain'] = 0.45
        # if config['exposure-level']=="high":
        #     config['lsm-scan-rate'] = '250000.0000'
        #     config['lsm-pc-power'] = 0.60
        #     config['lsm-pmt-gain'] = 0.50
        # if config['exposure-level']=="extreme":
        #     config['lsm-scan-rate'] = '200000.0000'
        #     config['lsm-pc-power'] = 0.75
        #     config['lsm-pmt-gain'] = 0.55
        config['pixel-size-shg'] = config['pixel-size-shg-base'] * 256 / config["lsm-resolution"]
        return config
    
    
    def update_slide_box(self, box):
        """
            args: box (tuple): x_start, x_end, y_start, y_end
        """
        assert len(box)==4, "Needs 4 values for the bounding box."
        self.config['slide-box'] = (box[0], box[1], box[2], box[3])
        
        
    def set_bf_4x_focus(self, focus_device):
        """
            args: focus_device (string): name of the z focus device.
        """
        core = self.core
        config = self.config
        core.set_fucos_device(focus_device)
        config['Z-stage-4x'] = core.get_position()
        aimed_z = config['Z-stage-4x'] + config['Z-bf-offset']
        config['Z-stage-20x'] = limit_stage(aimed_z, (config['hard-limit-z'][0], config['hard-limit-z'][1]), default=config['Z-stage-20x'])
        self.config = config
        
    
    def dump_configuration(self, acq_name, path=None):
        config = self.config
        if path is not None:
            save_path = path    
        else:
            save_path = 'acquisition-configs'
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, acq_name+'.json'), 'w') as fp:
            config_save = copy.deepcopy(config)
            json.dump(config_save, fp)
            
            
    def switch_objective(self, mag='4x'):
        config = self.config
        core = self.core
        if mag == '4x':
            core.set_property('Turret:O:35', 'Label', 'Position-2')
            core.set_focus_device(config['condensor-device'])
            core.set_position(config['F-stage-4x'])
            core.set_focus_device(config['focus-device'])
            focus_z = limit_stage(config['Z-stage-4x'], (config['hard-limit-z'][0], config['hard-limit-z'][1]), default=config['Z-stage-4x'])
            core.set_position(focus_z)
            core.set_property(config['led-device'][0], config['led-device'][1], config['led-4x'])
            core.wait_for_system()
            print("Imaging objective set as 4x")
        if mag == '20x':
            core.set_property(config['obj-device'][0], config['obj-device'][1], 'Position-1')
            core.set_focus_device(config['condensor-device'])
            core.set_position(config['F-stage-20x'])
            core.set_focus_device(config['focus-device'])
            focus_z = limit_stage(config['Z-stage-20x'], (config['hard-limit-z'][0], config['hard-limit-z'][1]), default=config['Z-stage-20x'])
            core.set_position(focus_z)
            core.set_property(config['led-device'][0], config['led-device'][1], config['led-20x'])
            core.wait_for_system()
            print("Imaging objective set as 20x")
            
            
    def switch_mod(self, mod='shg'):
        core = self.core
        config = self.config
        current_objective = core.get_property('Turret:O:35', 'Label')
        if mod == 'shg':
            if current_objective == 'Position-2':
                print('Current magnification not supported for LSM. Switching magnification.')
                core.set_property(config['obj-device'][0], config['obj-device'][1], 'Position-1')
                core.wait_for_system()
            core.set_property('Turret:O:35', 'Label', 'Position-1')
            core.set_focus_device(config['condensor-device'])
            core.set_position(config['F-stage-laser']) # new value
            core.set_focus_device(config['focus-device'])
            focus_z = limit_stage(config['Z-stage-laser'], (config['hard-limit-z'][0], config['hard-limit-z'][1]), default=config['Z-stage-laser'])
            core.set_position(focus_z) #
            # core.set_config('Imaging', 'LSM')
            core.set_property(config['led-device'][0], config['led-device'][1], 0.0)
            core.set_property('Shutters-DigitalIODev1', 'State', 3)
            core.set_property('Core', 'Camera', 'OSc-LSM')
            core.set_property('Core', 'Shutter', 'UniblitzShutter')
            core.set_property('QCamera', 'Color', 'OFF')
            core.set_property('OSc-LSM', 'LSM-Resolution', config['lsm-resolution'])
            core.set_property('OSc-LSM', 'LSM-PixelRateHz', config['lsm-scan-rate'])
            core.set_property('PockelsCell-Dev1ao1', 'Voltage', config['lsm-pc-power'])
            core.set_property('DCC100', 'DCC100 status', 'On')
            core.set_property('DCC100', 'ClearOverload', 'Clear')
            core.wait_for_system()
            core.set_property('DCC100', 'DCC100 status', 'On')
            core.set_property('DCC100', 'Connector1GainHV_Percent', config['lsm-pmt-gain']*100)
            core.wait_for_system()
            print("Imaging mode set as SHG")
        if mod == 'bf':
            core.set_property('DCC100', 'DCC100 status', 'On')
            core.set_property('DCC100', 'ClearOverload', 'Clear')
            core.wait_for_system()
            core.set_property('DCC100', 'DCC100 status', 'Off')
            core.wait_for_system()
            # core.set_config('Imaging', 'Camera')
            core.set_property('Shutters-DigitalIODev1', 'State', 0)
            core.set_property('Core', 'Camera', 'QCamera')
            core.set_property('Core', 'Shutter', 'WhiteLED')
            core.set_property('QCamera', 'Color', 'ON')
            if current_objective == 'Position-2':
                self.switch_objective('4x')
            if current_objective == 'Position-1': 
                self.switch_objective('20x')  
            print("Imaging mode set as Brightfield")
            
            
    def snap_image(self, rgb=True, flip_channel=True):
        core = self.core
        core.snap_image()
        tagged_image = core.get_tagged_image()
        if rgb == True:
            pixels = np.reshape(
                tagged_image.pix,
                newshape=[tagged_image.tags["Height"], tagged_image.tags["Width"], 4],
                )
            pixels = pixels[:, :, 0:3]
            if flip_channel:
                pixels = np.flip(pixels, 2)
            if self.bf_process_fn is not None:
                pixels = self.bf_process_fn(pixels)
        else:
            pixels = np.reshape(
                tagged_image.pix,
                newshape=[tagged_image.tags["Height"], tagged_image.tags["Width"]],
                )
        self.last_bf_img = pixels
        return pixels
    
            
    def live(self, mod='bf', flip_channel=True):
        config = self.config
#         self.switch_mod(mod=mod)
        fig = plt.figure(figsize=(8, 6))
        plt.axis('off')
        if mod == 'bf':
            show = plt.imshow(np.zeros((config['camera-resolution'][1], config['camera-resolution'][0])))
            try:
                while(1):
                    pixels = self.snap_image(rgb=True, flip_channel=True)
                    show.set_data(pixels)
                    display.display(plt.gcf())
                    display.clear_output(wait=True)
            except KeyboardInterrupt:
                pass
        if mod == 'shg':
            show = plt.imshow(np.zeros((config['lsm-resolution'], config['lsm-resolution'])), cmap='gray', vmin=0, vmax=255)
            try:
                while(1):
                    pixels = self.snap_image(rgb=False)
                    show.set_data(img_as_ubyte(exposure.rescale_intensity(pixels, in_range=(5000, 10000), out_range=(0, 1))))
                    display.display(plt.gcf())
                    display.clear_output(wait=True)
            except KeyboardInterrupt:
                plt.close('all')
                pass
        return pixels
    
    def update_focus_presets(self, mod='bf', mag='4x'):
        config = self.config
        core = self.core
        if mod=='bf' and mag=='4x':
            pos_z_4x = core.get_position()         
            config["Z-stage-4x"] = limit_stage(pos_z_4x, (config['hard-limit-z'][0], config['hard-limit-z'][1]), default=config["Z-stage-4x"])
            pos_z_20x = config["Z-stage-4x"] + config["Z-bf-offset"]
            config["Z-stage-20x"] = limit_stage(pos_z_20x, (config['hard-limit-z'][0], config['hard-limit-z'][1]), default=config["Z-stage-20x"])
            self.config = config
        elif mod=='bf' and mag=='20x':
            pos_z_20x = core.get_position()         
            config["Z-stage-20x"] = limit_stage(pos_z_20x, (config['hard-limit-z'][0], config['hard-limit-z'][1]), default=config["Z-stage-20x"])
            pos_z_shg = config["Z-stage-20x"] + config["Z-laser-offset"]
            config["Z-stage-laser"] = limit_stage(pos_z_shg, (config['hard-limit-z'][0], config['hard-limit-z'][1]), default=config["Z-stage-laser"])
            self.config = config
        elif mod=='shg':
            pos_z_shg = core.get_position()         
            config["Z-stage-laser"] = limit_stage(pos_z_shg, (config['hard-limit-z'][0], config['hard-limit-z'][1]), default=config["Z-stage-laser"])
            self.config = config
    
    
    def resample_z_pos(self, mag='20x', xy_pos=None, xyz_pos_list_4x=None, xyz_pos_list_20x=None):
        """
            this function transfer an annotation made on 4x BF to the target `mag` modality based on focus map of either 4x BF or 20x BF.
            args: mag (string): target modality (`'20x'|'mp'`). xy_pos (list): annotations position list generated from QuPath. 
                xyz_pos_list_4x (list): reference 4x focus map. xyz_pos_list_20x (list): reference 30x focus map.
            rReturns: (list): transferred resampled focus map for the annotation position list.
        """
        config = self.config
        if xyz_pos_list_4x is not None:
            xy_pos_list = xyz_pos_list_4x[:, :, :2] # x, y, z
            z_pos_list = xyz_pos_list_4x[:, :, 2]
            list_h = xyz_pos_list_4x.shape[0]
            list_w = xyz_pos_list_4x.shape[1]
            dense_xy = transform.resize(xy_pos_list, (list_h*100, list_w*100), order=1, preserve_range=True, mode='edge')
            dense_z = transform.resize(z_pos_list, (list_h*100, list_w*100), order=3, preserve_range=True, mode='edge')
            dense_xyz = np.concatenate((dense_xy, dense_z[:, :, None]), axis=2)
            xyz_list = np.ones((xy_pos.shape[0], 3))
            for i in range(xy_pos.shape[0]):
                x_pos_source = xy_pos[i, 0]
                y_pos_source = xy_pos[i, 1]
                if mag=='20x': # transfer back to input grid position (4x)
                    z_offset = config['Z-bf-offset']
                    x_pos = x_pos_source - config['20x-bf-offset'][0]
                    y_pos = y_pos_source - config['20x-bf-offset'][1]
                if mag=='mp':
                    z_offset = config['Z-laser-offset'] + config['Z-bf-offset']
                    x_pos = x_pos_source - config['shg-offset'][0]
                    y_pos = y_pos_source - config['shg-offset'][1]
                x_idx = np.abs(dense_xyz[0, :, 0] - x_pos).argmin()
                y_idx = np.abs(dense_xyz[:, 0, 1] - y_pos).argmin()
                z_pos = dense_xyz[y_idx, x_idx, 2] + z_offset
                z_pos = limit_stage(z_pos, (config['hard-limit-z'][0], config['hard-limit-z'][1]), default=None)
                xyz_list[i] = [x_pos_source, y_pos_source, z_pos]
        if xyz_pos_list_20x is not None:
            xyz_list = np.ones((xy_pos.shape[0], 3))
            for i in range(xy_pos.shape[0]):
                x_pos_source = xy_pos[i, 0] 
                y_pos_source = xy_pos[i, 1]       
                if mag=='20x': # transfer back to input grid position (20x)
                    z_offset = 0
                    x_pos = x_pos_source
                    y_pos = y_pos_source
                if mag=='mp':
                    z_offset = config['Z-laser-offset']
                    x_pos = x_pos_source - config['shg-offset'][0] + config['20x-bf-offset'][0] 
                    y_pos = y_pos_source - config['shg-offset'][1] + config['20x-bf-offset'][1]
                distance = np.sqrt((x_pos-xyz_pos_list_20x[:, 0])**2 + (y_pos-xyz_pos_list_20x[:, 1])**2)
                idx = np.argmin(distance)
                z_pos = xyz_pos_list_20x[idx, 2] + z_offset
                z_pos = limit_stage(z_pos, (config['hard-limit-z'][0], config['hard-limit-z'][1]), default=None)
                xyz_list[i, :] = np.array([x_pos_source, y_pos_source, z_pos])
        return xyz_list

    def annotations_positionlist(self, image_name, in_mag='4x', out_mag='4x'):
        config = self.config
        pos_lists = []
        if in_mag == '4x':
            pixel_size = config["pixel-size-bf-4x"]
            if out_mag == '4x':
                annotations = glob.glob(os.path.join('qupath-projects', '4x-tiles', image_name+'*.csv'))
                off_set = (0, 0)
            if out_mag == '20x':
                annotations = glob.glob(os.path.join('qupath-projects', '20x-tiles', image_name+'*.csv'))
                off_set = config["20x-bf-offset"]
            if out_mag == 'mp':
                annotations = glob.glob(os.path.join('qupath-projects', 'mp-tiles', image_name+'*.csv'))
                off_set = config["shg-offset"]
        annotations.sort()
        annotation_names = []
        for annotation in annotations:
            df = pd.read_csv(annotation)
            pos_list = np.array(df)
            pos_list[:, 0] = pos_list[:, 0]*pixel_size + off_set[0] + config["slide-box"][0]
            pos_list[:, 1] = pos_list[:, 1]*pixel_size + off_set[1] + config["slide-box"][1]
            pos_lists.append(pos_list)
            annotation_name = annotation.split(image_name)[-1].split('.')[0]
            annotation_names.append(annotation_name)
        return pos_lists, annotation_names # list of (x_pos, y_pos) array
    
    
    
    def generate_grid(self, mag='4x', overlap=50):
        """
            this function generate a rectangular position list for modality `mag`.
            args: mag (string): modality to generate position list (`'4x'|'20x'|'mp'`). overlap (int): number of overlapped pixels between tiles.
            returns: (ndarray): position list shaped in a 2D array. Each entry contains (x, y) position of this location.
        """
        config = self.config
        s_x = config['slide-box'][0]
        s_y = config['slide-box'][1]
        e_x = config['slide-box'][2]
        e_y = config['slide-box'][3]
        if mag == '20x':
            pixel_size = config['pixel-size-bf-20x']
            ield_w = config['camera-resolution'][0] * pixel_size
            field_h = config['camera-resolution'][1] * pixel_size
        elif mag == '4x':
            pixel_size = config['pixel-size-bf-4x']
            field_w = config['camera-resolution'][0] * pixel_size
            field_h = config['camera-resolution'][1] * pixel_size
        elif mag == 'mp':
            pixel_size = config['pixel-size-shg']
            field_w = config['lsm-resolution'] * pixel_size
            field_h = config['lsm-resolution'] * pixel_size
        field_o = overlap * pixel_size
        grid_w = int(np.floor((e_x - s_x) / (field_w - field_o))) # number of fields in x direction
        grid_h = int(np.floor((e_y - s_y) / (field_h - field_o))) # number of fields in y direction
        xy_list = np.zeros((grid_h, grid_w, 2)) # h x w x 2, xy_list[i, j, :]
        for x in range(grid_w): # row
            for y in range(grid_h): # column intergers 0~grid_h but not grid_h
                if y % 2 == 0:
                    x_pos = x * (field_w - field_o) + s_x
                    y_pos = y * (field_h - field_o) + s_y
                else:
                    x_pos = (grid_w-(x+1)) * (field_w - field_o) + s_x
                    y_pos = y * (field_h - field_o) + s_y
                xy_list[y, x] = [x_pos, y_pos] # x, y
        return xy_list # row, col
    
    
    def set_background(self, mag='4x', img=None):
        if img is None:
            print('Use the last snapped/live image as background.')
            img_to_set = self.last_img
        else:
            img_to_set = img
            
        if mag=='4x':
            self.bf_4x_bg = img_to_set
        elif mag=='20x':
            self.bf_20x_bg = img_to_set
        else:
            raise ValueError('Background image not supported for current magnification.')
            
            
    def autofocus(self, method='edge', mag='4x', interpolation='quadratic', rgb=True, search_range=45, steps=3, snap=True, flip_channel=True, 
                  check_background=True, offset=0, preset=None):
        """
            this function conducts autofocus at current position. It can also check if the current tile is a background and return a snap.
            args: method (string): focus metric (`'entropy'|'edge'`). interpolation (string): interpolation method of the focus score curve.
                search_range (float): search range in um. steps (int): number of steps in the search range. flip_channel (bool): whether to flip RBGA channels.
                check_background (bool): whether to check if the current tile is a background. offset (float): a constant offset added to the returned optimal focus.
                preset (float): a preset focus for the current tile if it is a background.
        """
        core = self.core
        config = self.config
        if mag=='4x':
            drift_origin = config["Z-stage-4x"]
        if mag=='20x':
            drift_origin = config["Z-stage-20x"]
        core.set_focus_device(config["focus-device"])   
        current_z = core.get_position()
        interval_z = search_range/steps
        scores = []
        positions = []
        count = 0
        for step in range(-int(np.floor(steps/2)), int(np.ceil(steps/2))):
            position_z = step * interval_z + current_z
            position_z = limit_stage(position_z, (config['hard-limit-z'][0], config['hard-limit-z'][1]), default=drift_origin)
            core.set_position(position_z)
            core.wait_for_system()
            count = count + 1
            pixels = self.snap_image(rgb=rgb, flip_channel=True)
            if check_background and step==-int(np.floor(steps/2)):
                bg_flag = is_background(pixels, t=0.28, tt=0.28)
                if bg_flag:
                    if preset is not None:
                        preset = limit_stage(preset, (config['hard-limit-z'][0], config['hard-limit-z'][1]), default=drift_origin)
                        core.set_position(preset)
                    else:
                        drift_origin = limit_stage(drift_origin, (config['hard-limit-z'][0], config['hard-limit-z'][1]), default=drift_origin)
                        core.set_position(drift_origin)
                    core.wait_for_system()
                    print("Is background")
                    return drift_origin, pixels, bg_flag # TODO: return center z instead of top
            img_gray = color.rgb2gray(pixels)
            sys.stdout.write("\r Diving focus at " + str(step))
            if method == 'entropy':
                score = shannon_entropy(img_gray)
            if method == 'edge':
                score = np.mean(sobel(img_gray))
            scores.append(score)
            positions.append(position_z)
            print('Score: {}, Position {}'.format(score, position_z))
        scores_array = np.asarray(scores)
        positions_array = np.asarray(positions) 
        new_length = len(positions) * 100
        new_x = np.linspace(positions_array.min(), positions_array.max(), new_length)
        new_y = scipy.interpolate.interp1d(positions_array, scores_array, kind=interpolation)(new_x)
        idx = np.argmax(new_y)
        focus_z = new_x[idx]
        focus_z = limit_stage(focus_z, (config['hard-limit-z'][0], config['hard-limit-z'][1]), default=position_z)
        if np.abs(focus_z-drift_origin) > 200:
            print("Large change in z-stage , reset focus")
            focus_z = drift_origin
            core.set_position(drift_origin)
            core.wait_for_system()
        else:
            core.set_position(focus_z)
            core.wait_for_system()
        if snap:
            pixels = self.snap_image(rgb=rgb, flip_channel=True)
            return focus_z+offset, pixels, False
        else:
            return focus_z+offset, None, False 
        
    
    def whole_slide_bf_scan(self, 
            save_path, 
            acq_name, 
            position_list, 
            mag='4x', 
            estimate_background=True, 
            focus_dive=True
        ):
        """
            this function conducts a BF scan at 4x or 20x magnification.
            args: save_path (string): path for saving the acquisition data. acq_name (string): name of the acquisition. mag (string): magnification (`'4x'|'20x'`).
                estimate_background (bool): estimate the background during acquisition. focus_dive (bool): use autofocus during acquisition.
            returns: (ndarray): estimated/used background image. (list): z positions in a flattened list.
        """
        config = self.config
        core = self.core
        fig = plt.figure(figsize=(8, 6))
        plt.axis("off")
        show = plt.imshow(np.zeros((config['camera-resolution'][1], config['camera-resolution'][0])))
        acq_id = len(glob.glob(os.path.join(save_path, acq_name+"*")))
        acq_path = os.path.join(save_path, acq_name+"_{}".format(acq_id+1))
        os.makedirs(acq_path, exist_ok=True)
        bg_flag = False
        sp_flag = False
        bg_t = 0.28
        bg_tt = 0.28
        returns = {}
        background_image = None
        if estimate_background:
            bg_stack = []
        else:
            if mag=='4x':
                background_image = self.bf_4x_bg
                if self.bf_process_fn is not None: background_image = self.bf_process_fn(background_image)
            elif mag=='20x':
                background_image = self.bf_20x_bg
                if self.bf_process_fn is not None: background_image = self.bf_process_fn(background_image)
            if background_image is None:
                raise ValueError('Background image not set nor to be estimated.')
            bg_img = white_balance(copy.deepcopy(background_image), copy.deepcopy(background_image))
            
        if mag == '4x':
            pos_z = config['Z-stage-4x']
        elif mag == '20x':
            pos_z = config["Z-stage-20x"]
            
        support_points = [(99999999, 99999999)] # dummy support point
        support_focus = [pos_z]
        
        if position_list.shape[1] == 3:
            tile_count = 0
            z_positions=np.ones(position_list.shape[0]) * core.get_position()
            core.set_focus_device(config['focus-device'])
            for pos in tqdm(range(position_list.shape[0])):
                z_pos = position_list[pos, 2]
                x_pos = position_list[pos, 0]
                y_pos = position_list[pos, 1]
                z_pos = pos_z
                z_pos = limit_stage(z_pos, (config['hard-limit-z'][0], config['hard-limit-z'][1]), default=None)
                x_pos = limit_stage(x_pos, (config['hard-limit-x'][0], config['hard-limit-x'][1]), default=None)
                y_pos = limit_stage(y_pos, (config['hard-limit-y'][0], config['hard-limit-y'][1]), default=None)
                core.set_position(z_pos)
                core.set_xy_position(x_pos, y_pos)
                xy_device = core.get_xy_stage_device()
                z_device = core.get_focus_device()
                core.wait_for_device(xy_device)
                core.wait_for_device(z_device)
                sp_flag = False
                
                if focus_dive and mag=='4x':
                    support_distance = config['pixel-size-bf-4x'] * config['camera-resolution'][1] * config['autofocus-speed']
                    idx, min_distance = distance((x_pos, y_pos), support_points)
                    if min_distance <= support_distance:
                        pos_z = support_focus[idx]
                        pos_z = limit_stage(pos_z, (config['hard-limit-z'][0], config['hard-limit-z'][1]), default=config['Z-stage-4x'])
                        core.set_position(pos_z)
                        core.wait_for_device(z_device)
                        pixels = self.snap_image(rgb=True, flip_channel=True)
                        bg_flag = is_background(pixels, t=bg_t, tt=bg_tt)
                    else:
                        pos_z, pixels, bg_flag = self.autofocus(mag='4x', rgb=True, search_range=160, steps=5, snap=True, preset=z_pos) # snap at top but return center z
                        if bg_flag:
                            if len(support_points)>=2:
                                pos_z = limit_stage(pos_z, (config['hard-limit-z'][0], config['hard-limit-z'][1]), default=config["Z-stage-4x"])
                                core.set_position(pos_z)
                            else:
                                pos_z = z_pos
                            core.wait_for_device(z_device)
                            pixels = self.snap_image(rgb=True, flip_channel=True)
                        else:
                            support_points.append((x_pos, y_pos))
                            support_focus.append(pos_z)
                            sp_flag = True
                    z_positions[pos] = pos_z
                           
                if focus_dive and mag=='20x':
                    support_distance = config['pixel-size-bf-20x'] * config['camera-resolution'][1] * config['autofocus-speed']
                    idx, min_distance = distance((x_pos, y_pos), support_points)
                    if min_distance <= support_distance:
                        pos_z = support_focus[idx]
                        pos_z = limit_stage(pos_z, (config['hard-limit-z'][0], config['hard-limit-z'][1]), default=config["Z-stage-20x"])
                        core.set_position(pos_z)
                        core.wait_for_device(z_device)
                        pixels = self.snap_image(rgb=True, flip_channel=True)
                        bg_flag = is_background(pixels, t=bg_t, tt=bg_tt)
                    else:
                        pos_z, pixels, bg_flag = self.autofocus(mag='20x', rgb=True, search_range=120, steps=4, snap=True, preset=z_pos) # snap at top but return center z
                        if bg_flag:
                            if len(support_points)>=2:
                                pos_z = limit_stage(pos_z, (config['hard-limit-z'][0], config['hard-limit-z'][1]), default=config["Z-stage-20x"])
                                core.set_position(pos_z)
                            else:
                                pos_z = z_pos
                            core.wait_for_device(z_device)
                            pixels = self.snap_image(rgb=True, flip_channel=True)
                        else:
                            pos_z, pixels, _ = self.autofocus(mag='20x', rgb=True, search_range=42, steps=7, snap=True, check_background=False, preset=z_pos) 
                            support_points.append((x_pos, y_pos))
                            support_focus.append(pos_z)
                            sp_flag = True
                    z_positions[pos] = pos_z 
                    
                pixels = img_as_float(pixels)   
                
                if estimate_background:
                    if focus_dive:
                        bg_flag = bg_flag
                    else:
                        bg_flag = is_background(pixels, t=0.28, tt=0.28)
                        print('hard check')
                    if bg_flag:
                        print(' (background tile)')
                        redive_flag=True
                        bg_stack.append(pixels)
                    else:
                        redive_flag=False                
                if background_image is not None and not estimate_background:
                    pixels = white_balance(pixels, background_image)
                    pixels = flat_field(pixels, bg_img)
                    
                show.set_data(pixels)
                display.display(plt.gcf())
                display.clear_output(wait=True)
                ### Use tifffile to write out a tile with metadata?
                io.imsave(acq_path+'/{}-{}-{}.tiff'.format(pos, bg_flag, sp_flag), img_as_ubyte(pixels), check_contrast=False)
                tile_count = tile_count + 1
                sys.stdout.write('\r {}/{} tiles done'.format(tile_count, position_list.shape[0]))
                plt.close('all')
            
        if position_list.shape[1] == 2:
            tile_count = 0
            core.set_focus_device(config['focus-device'])
            z_positions=np.ones(position_list.shape[0]) * core.get_position()
            for pos in tqdm(range(position_list.shape[0])):
                x_pos = position_list[pos, 0]
                y_pos = position_list[pos, 1]
                
                x_pos = limit_stage(x_pos, (config['hard-limit-x'][0], config['hard-limit-x'][1]), default=None)
                y_pos = limit_stage(y_pos, (config['hard-limit-y'][0], config['hard-limit-y'][1]), default=None)
                    
                xy_device = core.get_xy_stage_device()
                z_device = core.get_focus_device()
                core.set_xy_position(x_pos, y_pos)
                core.wait_for_device(xy_device)
                sp_flag = False
                    
                if focus_dive and mag=='4x':
                    support_distance = config['pixel-size-bf-4x'] * config['camera-resolution'][1] * config['autofocus-speed']
                    idx, min_distance = distance((x_pos, y_pos), support_points)
                    if min_distance <= support_distance:
                        pos_z = support_focus[idx]
                        pos_z = limit_stage(pos_z, (config['hard-limit-z'][0], config['hard-limit-z'][1]), default=config['Z-stage-4x'])
                        core.set_position(pos_z)
                        core.wait_for_device(z_device)
                        pixels = self.snap_image(rgb=True, flip_channel=True)
                        bg_flag = is_background(pixels, t=bg_t, tt=bg_tt)
                    else:
                        pos_z, pixels, bg_flag = self.autofocus(mag='4x', rgb=True, search_range=100, steps=3, snap=True) # snap at top but return center z
                        if not bg_flag:
                            support_points.append((x_pos, y_pos))
                            support_focus.append(pos_z)
                            sp_flag = True
                    z_positions[pos] = pos_z
                    
                if focus_dive and mag=='20x':
                    support_distance = config["pixel-size-bf-20x"] * config["camera-resolution"][1] * config["autofocus-speed"]
                    idx, min_distance = distance((x_pos, y_pos), support_points)
                    if min_distance <= support_distance:
                        pos_z = support_focus[idx]
                        pos_z = limit_stage(pos_z, (config['hard-limit-z'][0], config['hard-limit-z'][1]), default=config['Z-stage-20x'])
                        core.set_position(pos_z)
                        core.wait_for_device(z_device)
                        pixels = self.snap_image(rgb=True, flip_channel=True)
                        bg_flag = is_background(pixels, t=bg_t, tt=bg_tt)
                    else:
                        pos_z, pixels, bg_flag = self.autofocus(mag='20x', rgb=True, search_range=50, steps=3, snap=False) # snap at top but return center z
                        pos_z, pixels, bg_flag = self.autofocus(mag='20x', rgb=True, search_range=10, steps=3, snap=True) # snap at top but return center z
                        if not bg_flag:
                            support_points.append((x_pos, y_pos))
                            support_focus.append(pos_z)
                            sp_flag = True
                    z_positions[pos] = pos_z    
                    
                pixels = img_as_float(pixels)                   
                
                if estimate_background:
                    if focus_dive:
                        bg_flag = bg_flag
                    else:
                        bg_flag = is_background(pixels, t=bg_t, tt=bg_tt)
                        print('hard check')
                    if bg_flag:
                        print(' (background tile)')
                        redive_flag=True
                        bg_stack.append(pixels)
                    else:
                        redive_flag=False                
                if background_image is not None and not estimate_background:
                    pixels = white_balance(pixels, background_image)
                    pixels = flat_field(pixels, bg_img)
                    
                show.set_data(pixels)
                display.display(plt.gcf())
                display.clear_output(wait=True)
                ### Use tifffile to create tile with metadata?
                io.imsave(acq_path+'/{}-{}-{}.tiff'.format(pos, bg_flag, sp_flag), img_as_ubyte(pixels), check_contrast=False)
                tile_count = tile_count + 1
                sys.stdout.write('\r {}/{} tiles done'.format(tile_count, position_list.shape[0]))
                plt.close('all')
        if estimate_background:
            if len(bg_stack)==0:
                if mag=='4x':
                    background_image = self.bf_4x_bg
                    if self.bf_process_fn is not None: background_image = self.bf_process_fn(background_image)
                elif mag=='20x':
                    background_image = self.bf_20x_bg
                    if self.bf_process_fn is not None: background_image = self.bf_process_fn(background_image)
                returns['Background image'] = background_image
                # io.imsave(acq_path+'/bg_img.tiff', img_as_ubyte(background_image))
            else:
                bg_stack= np.stack(bg_stack)
                median = np.median(bg_stack, axis=0)
                median = img_as_float(median)
                returns['Background image'] = median
                # io.imsave(acq_path+'/bg_img.tiff', img_as_ubyte(median))
        else:
            returns['Background image'] = background_image
        if focus_dive:
            z_positions = z_positions.reshape(position_list.shape[0], 1)
            returns['Z positions'] = z_positions
        return returns

    def define_lsm_processor(self, func):
        def img_process_fn(image, metadata):
            image = func(image)
            image = img_as_uint(image)
            return image, metadata
        self.lsm_process_fn = img_process_fn

    # def lsm_reset_PMT(self):
    #     config = self.config
    #     def lsm_hook_fn(event, bridge, event_queue):
    #         core = bridge.get_core()
    #         core.set_property('DCC100', 'DCC100 status', 'On')
    #         core.set_property('DCC100', 'ClearOverload', 'Clear')
    #         core.wait_for_system()
    #         core.set_property('DCC100', 'DCC100 status', 'On')
    #         core.set_property('DCC100', 'Connector1GainHV_Percent', config['lsm-pmt-gain']*100)
    #         return event
    #     self.lsm_hook_fn=lsm_hook_fn
    
    def whole_slide_lsm_scan(self, save_path=None, acq_name=None, position_list=None, z_stack=False, z_center=None, sample_depth=20, z_step=4):
        config = self.config
        if position_list.shape[1] == 3:
            if z_stack:
                with Acquisition(save_path, acq_name, self.lsm_process_fn) as acq:
                    events = multi_d_acquisition_events(xyz_positions=position_list.reshape(-1, 3), 
                                                        z_start=-int(sample_depth/2), z_end=int(sample_depth/2), z_step=z_step)
                    acq.acquire(events)      
            else:
                with Acquisition(save_path, acq_name) as acq:
                    events = multi_d_acquisition_events(xyz_positions=position_list.reshape(-1, 3))
                    acq.acquire(events)
        elif position_list.shape[1] == 2:
            if z_center is None:
                z_center = config["Z-stage-laser"]
            if z_stack:
                with Acquisition(save_path, acq_name) as acq:
                    events = multi_d_acquisition_events(xy_positions=position_list.reshape(-1, 2), 
                                                        z_start=-int(sample_depth/2) + z_center, z_end=int(sample_depth/2) + z_center, z_step=z_step)
                    acq.acquire(events)
            else:
                with Acquisition(save_path, acq_name) as acq:
                    events = multi_d_acquisition_events(xy_positions=position_list.reshape(-1, 2))
                    acq.acquire(events)
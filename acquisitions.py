import os, glob, shutil, sys, copy, time, json, copy
from IPython import display
from tqdm import tqdm
from pycromanager import Acquisition, Bridge, Dataset, multi_d_acquisition_events
from skimage import io, img_as_ubyte, img_as_float, color, transform, exposure
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def snap_image(core, rgb, flip_channel=True):
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
    else:
        pixels = np.reshape(
            tagged_image.pix,
            newshape=[tagged_image.tags["Height"], tagged_image.tags["Width"]],
            )
    return pixels

def live(config, core, mod='bf', flip_channel=True):
    switch_mod(config, core, mod=mod)
    fig = plt.figure(figsize=(8, 6))
    plt.axis("off")
    if mod == 'bf':
        show = plt.imshow(np.zeros((config["camera-resolution"][1], config["camera-resolution"][0])))
        try:
            while(1):
                pixels = snap_image(core, rgb=True, flip_channel=True)
                show.set_data(pixels)
                display.display(plt.gcf())
                display.clear_output(wait=True)
        except KeyboardInterrupt:
            pass
    if mod == 'shg':
        show = plt.imshow(np.zeros((config["lsm-resolution"], config["lsm-resolution"])), cmap='gray', vmin=0, vmax=255)
        try:
            while(1):
                pixels = snap_image(core, rgb=False)
                show.set_data(img_as_ubyte(exposure.rescale_intensity(pixels, in_range=(5000, 10000), out_range=(0, 1))))
                display.display(plt.gcf())
                display.clear_output(wait=True)
        except KeyboardInterrupt:
            pass
    return pixels

def switch_objective(config, core, mag='4x'): # brightfield
    if mag == '4x':
        core.set_property('Turret:O:35', 'Label', 'Position-2')
        core.set_focus_device(config["condensor-device"])
        core.set_position(config["F-stage-4x"])
        core.set_focus_device(config["focus-device"])
        core.set_position(config["Z-stage-4x"])
        core.set_property(config["led-device"][0], config["led-device"][1], config["led-4x"])
        core.wait_for_system()
    if mag == '20x':
        core.set_property(config["obj-device"][0], config["obj-device"][1], 'Position-1')
        core.set_focus_device(config["condensor-device"])
        core.set_position(config["F-stage-20x"])
        core.set_focus_device(config["focus-device"])
        core.set_position(config["Z-stage-20x"])
        core.set_property(config["led-device"][0], config["led-device"][1], config["led-20x"])
        core.wait_for_system() 
        
def switch_mod(config, core, mod='shg'):
    current_objective = core.get_property('Turret:O:35', 'Label')
    if mod == 'shg':
        if current_objective == 'Position-2':
            print('Not supported magnification for LSM')
            core.set_property(config["obj-device"][0], config["obj-device"][1], 'Position-1')
            core.wait_for_system()
        core.set_property('Turret:O:35', 'Label', 'Position-1')
        core.set_focus_device(config["condensor-device"])
        core.set_position(config["F-stage-laser"]) # new value
        core.set_focus_device(config["focus-device"])
        core.set_position(config["Z-stage-laser"]) #
        core.set_property(config["led-device"][0], config["led-device"][1], 0.0)
        core.set_config('Imaging', 'LSM')
        core.set_property(config["led-device"][0], config["led-device"][1], 0.0)
        core.set_property("OSc-LSM", "Resolution", config["lsm-resolution"])
        core.set_property("OSc-LSM", "Bin Factor", config["lsm-bin-factor"])
        core.set_property("OSc-LSM", "PixelRateHz", config["lsm-scan-rate"])
        core.set_property("PockelsCell-Dev1ao1", "Voltage", config["lsm-pc-power"])
        core.set_property("DCC100", "DCC100 status", "On")
        core.set_property("DCC100", "ClearOverload", "Clear")
        core.wait_for_system()
        core.set_property("DCC100", "DCC100 status", "On")
        core.set_property("DCC100", "Connector3GainHV_Percent", config["lsm-pmt-gain"])
        core.wait_for_system()
        print('Imaging mode set as SHG')
    if mod == 'bf':
        core.set_property("DCC100", "DCC100 status", "On")
        core.set_property("DCC100", "ClearOverload", "Clear")
        core.wait_for_system()
        core.set_property("DCC100", "DCC100 status", "Off")
        core.wait_for_system()
        core.set_config('Imaging', 'Camera')
        if current_objective == 'Position-2':
            switch_objective(config, core, '4x')
        if current_objective == 'Position-1': 
            switch_objective(config, core, '20x')  
        print('Imaging mode set as Brightfield')
        
def resample_z_pos(config, mag='20x', xy_pos=None, xyz_pos_list_4x=None, xyz_pos_list_20x=None):
    if xyz_pos_list_4x is not None:
        xy_pos_list = xyz_pos_list_4x[:, :, :2] # x, y, z
        z_pos_list = xyz_pos_list_4x[:, :, 2]
    #     list_w = xyz_pos_list[-1, -1, 0] - xyz_pos_list[0, 0, 0] # first pos, width
    #     list_h = xyz_pos_list[-1, -1, 1] - xyz_pos_list[0, 0, 1] # last pos, height
    #     resample_w = int(np.rint(list_w / 50))
    #     resample_h = int(np.rint(list_h / 50))
        list_h = xyz_pos_list_4x.shape[0]
        list_w = xyz_pos_list_4x.shape[1]
        dense_xy = transform.resize(xy_pos_list, (list_h*100, list_w*100), order=1, preserve_range=True, mode='edge')
        dense_z = transform.resize(z_pos_list, (list_h*100, list_w*100), order=3, preserve_range=True, mode='edge')
        dense_xyz = np.concatenate((dense_xy, dense_z[:, :, None]), axis=2)
    #     print(dense_z)
        xyz_list = np.ones((xy_pos.shape[0], 3))
        for i in range(xy_pos.shape[0]):
            x_pos_source = xy_pos[i, 0]
            y_pos_source = xy_pos[i, 1]
            if mag=='20x': # transfer back to input grid position (4x)
                z_offset = config["Z-stage-20x"] - config["Z-stage-4x"]
                x_pos = x_pos_source - config["20x-bf-offset"][0]
                y_pos = y_pos_source - config["20x-bf-offset"][1]
            if mag=='mp':
                z_offset = config["Z-stage-laser"] - config["Z-stage-4x"]
                x_pos = x_pos_source - config["shg-offset"][0]
                y_pos = y_pos_source - config["shg-offset"][1]
            x_idx = np.abs(dense_xyz[0, :, 0] - x_pos).argmin()
            y_idx = np.abs(dense_xyz[:, 0, 1] - y_pos).argmin()
            z_pos = dense_xyz[y_idx, x_idx, 2] + z_offset
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
                z_offset = config["Z-stage-laser"] - config["Z-stage-20x"]
                x_pos = x_pos_source - config["shg-offset"][0] + config["20x-bf-offset"][0] 
                y_pos = y_pos_source - config["shg-offset"][1] + config["20x-bf-offset"][1]
            distance = np.sqrt((x_pos-xyz_pos_list_20x[:, 0])**2 + (y_pos-xyz_pos_list_20x[:, 1])**2)
            idx = np.argmin(distance)
            z_pos = xyz_pos_list_20x[idx, 2] + z_offset
            xyz_list[i, :] = np.array([x_pos_source, y_pos_source, z_pos])
    return xyz_list # x, y, z
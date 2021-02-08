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

def live(core, mod='bf', flip_channel=True):
    switch_mod(config, mod=mod)
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

def switch_objective(config, mag='4x'): # brightfield
    if mag == '4x':
        core.set_property('Turret:O:35', 'Label', 'Position-2')
        core.set_focus_device(config["condensor-device"])
        core.set_position(config["F-stage-4x"])
        core.set_focus_device(config["focus-device"])
        core.set_position(config["Z-stage-4x"])
        core.set_property(config["led-device"][0], config["led-device"][1], 5.0)
        core.wait_for_system()
    if mag == '20x':
        core.set_property(config["obj-device"][0], config["obj-device"][1], 'Position-1')
        core.set_focus_device(config["condensor-device"])
        core.set_position(config["F-stage-20x"])
        core.set_focus_device(config["focus-device"])
        core.set_position(config["Z-stage-20x"])
        core.set_property(config["led-device"][0], config["led-device"][1], 5.0)
        core.wait_for_system() 
        
def switch_mod(config, mod='shg'):
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
        core.wait_for_system()
        print('Imaging mode set as SHG')
    if mod == 'bf':
        core.set_config('Imaging', 'Camera')
        if current_objective == 'Position-2':
            switch_objective(config, '4x')
        if current_objective == 'Position-1': 
            switch_objective(config, '20x')  
        print('Imaging mode set as Brightfield')
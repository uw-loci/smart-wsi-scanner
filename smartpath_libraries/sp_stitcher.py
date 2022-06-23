import os, glob, shutil, sys, copy
from pycromanager import Dataset
from skimage import io, img_as_float, img_as_uint, color, transform, exposure
import imagej
import numpy as np
from .image_utils import white_balance, flat_field
import subprocess



class SPStitcher:
    def __init__(self, config, ij, working_dir, qupath_dir):
        self.config = config
        self.ij = ij
        self.working_dir = working_dir
        self.qupath_dir = qupath_dir
        
        
    def stitch_bf(self, 
            acq_name, 
            mag, 
            position_list=None, 
            flip_x=False, 
            flip_y=False, 
            rotate=None, 
            correction=False, 
            background_image=None,
            slide_path='data/slides'
        ):
        ij = self.ij
        config = self.config
        save_path = self.working_dir
        stitch_folder = os.path.join('data/stitching/tiles', acq_name)
        out_folder = os.path.join('data/stitching/stitched', acq_name)
        slide_folder = os.path.join(slide_path, mag)
        os.makedirs(stitch_folder, exist_ok=True)
        os.makedirs(out_folder, exist_ok=True)
        os.makedirs(slide_folder, exist_ok=True)
        if mag == '20x':
            pixel_size = config["pixel-size-bf-20x"]
        elif mag == '4x':
            pixel_size = config["pixel-size-bf-4x"]
        image_list = glob.glob(os.path.join(glob.glob(save_path+'/'+acq_name+'*')[-1], '*.tiff'))
        image_list.sort(key=lambda x: os.path.getmtime(x))
        if correction is True and background_image is not None:
            image_list = image_list[0:-1]
            bg_img = white_balance(copy.deepcopy(background_image), copy.deepcopy(background_image))
        with open(os.path.join(stitch_folder, 'TileConfiguration.txt'), 'w') as text_file:
            print('dim = {}'.format(2), file=text_file)
            for pos in range(position_list.shape[0]):
                x = int(position_list[pos, 0] / pixel_size)
                y = int(position_list[pos, 1] / pixel_size)
                print('{}.tiff; ; ({}, {})'.format(pos, x, y), file=text_file)
                img = io.imread(image_list[pos])
                if correction is True and background_image is not None:
                    img = white_balance(img, background_image)
                    img = flat_field(img, bg_img)
                if rotate is not None:
                    img = transform.rotate(img, rotate)
                if flip_y:
                    img = img[::-1, :]
                if flip_x:
                    img = img[:, ::-1]
                sys.stdout.write('\r Processing tiles: {}/{}'.format(pos+1, position_list.shape[0]))
                io.imsave(stitch_folder+'/{}.tiff'.format(pos), img_as_uint(img), check_contrast=False)
        print('\n Calling ImageJ for stitching, please wait...')
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
        global macroRGB
        args = {
                'indir' : os.path.join(os.getcwd(), temp_channel_folder),
                'outdir' : os.path.join(os.getcwd(), os.path.join(slide_folder, acq_name+'.tif'))
        }
        result = ij.py.run_macro(macroRGB, args)
        
    def stitch_lsm(self, save_path, acq_name, n_stack=3, position_list=None, flip_x=False, flip_y=False, rotate=None):
        ij = self.ij
        config = self.config
        stitch_folder = os.path.join('data/stitching/tiles', acq_name)
        out_folder = os.path.join('data/stitching/stitched', acq_name)
        slide_folder = os.path.join('data/slides/mp')
        os.makedirs(stitch_folder, exist_ok=True)
        os.makedirs(out_folder, exist_ok=True)
        os.makedirs(slide_folder, exist_ok=True)
        pixel_size = config["pixel-size-shg"]
        data_paths = glob.glob(save_path+'/'+acq_name+'*')
        data_paths.sort()
        data_path = data_paths[-1]
        print('Stitching: ' + data_path)
        dataset = Dataset(data_path)
        with open(os.path.join(stitch_folder, 'TileConfiguration.txt'), 'w') as text_file:
            if n_stack==3 or n_stack==1: # 3 slices in z-stack would be treated as RGB
                print('dim = {}'.format(2), file=text_file)
            else:
                print('dim = {}'.format(3), file=text_file)

            for pos in range(position_list.shape[0]):
                x = int(position_list[pos, 0] / pixel_size)
                y = int(position_list[pos, 1] / pixel_size)
                if n_stack==3 or n_stack==1:
                    print('{}.tiff; ; ({}, {})'.format(pos, x, y), file=text_file)
                else:
                    print('{}.tiff; ; ({}, {}, {})'.format(pos, x, y, 0), file=text_file)
                z_idx = 0
                img_z_list = []
                while(dataset.has_image(position=pos, z=z_idx)):
                    img = dataset.read_image(position=pos, z=z_idx)
                    # img = img_as_float(img)
                    if rotate is not None:
                        img = transform.rotate(np.array(img), rotate)
                    img_z_list.append(img)
                    z_idx = z_idx+1
                    img = np.stack(img_z_list, axis=0)
                    if flip_y:
                        img = img[::-1, :]
                    if flip_x:
                        img = img[:, ::-1]
                    sys.stdout.write('\r Processing tiles: {}/{}'.format(pos+1, position_list.shape[0]))
                    io.imsave(stitch_folder+'/{}.tiff'.format(pos), img_as_uint(img), check_contrast=False)
        print('\n Calling ImageJ for stitching, please wait...')
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
        global marcroLSM
        args = {
                'indir' : os.path.join(os.getcwd(), temp_channel_folder),
                'outdir' : os.path.join(os.getcwd(), os.path.join(slide_folder, acq_name+'.tif'))
        }
        result = ij.py.run_macro(marcroLSM, args)
        
    def convert_slide(self, mag, remove_file=True):
        print('Converting slide to ome.tif')
        if mag=='4x':
            script = os.path.join('qupath-projects', 'scripts', 'export-ometif-metadata-4x.groovy')
        if mag=='20x':
            script = os.path.join('qupath-projects', 'scripts', 'export-ometif-metadata-20x.groovy')
        if mag=='mp':
            script = os.path.join('qupath-projects', 'scripts', 'export-ometif-metadata-mp.groovy')       
        image_dirs = glob.glob(os.path.join('data', 'slides', mag, '*.tif'))
        for img_dir in image_dirs:
            if img_dir.find("ome") != -1:
                continue
            subprocess.run([self.qupath_dir, "script", script, "-i", img_dir], shell=True)
            if remove_file:
                os.remove(img_dir)
        
    def clean_folders(self, acq_name):
        stitch_folder = os.path.join('data/stitching/tiles', acq_name)
        temp_channel_folder = 'data/stitching/channel_temp'
        shutil.rmtree(temp_channel_folder)
        shutil.rmtree(stitch_folder)

marcroLSM="""
#@ String indir
#@ String outdir
slices = getFileList(indir);
for (i=0;i<lengthOf(slices);i=i+1) {
    filePath = indir + '/' + slices[i];
    open(filePath);
}
run("Images to Stack", "name=Stack title=[] use");
run("Z Project...", "projection=[Max Intensity]");
saveAs("Tiff", outdir);
close("*");
"""

macroRGB = """
#@ String indir
#@ String outdir
slices = getFileList(indir);
for (i=0;i<lengthOf(slices);i=i+1) {
    filePath = indir + '/' + slices[i];
    open(filePath);
}
run("Merge Channels...", "c1=img_t1_z1_c1 c2=img_t1_z1_c2 c3=img_t1_z1_c3 create");
run("RGB Color");
saveAs("Tiff", outdir);
close("*");
"""
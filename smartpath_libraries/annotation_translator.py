import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from os.path import join
from os.path import splitext
import os
import cv2
from tqdm import tqdm
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import xml.etree.ElementTree as ET
import shutil
# from ctypes import *
# _lib = cdll.LoadLibrary('C:\\Program Files\\openslide-win64-20171122\\bin\\libopenslide-0.dll')
import openslide
from PIL import Image
Image.MAX_IMAGE_PIXELS = None


class AnnotationTranslator:
    def __init__(self, aperios, camms, annotations, down_factor=32, output_path='registered'):
        """
            Args: aperios (string): list of paths to aperio macros (moving images). camms (string): list of paths to camms macros (fixed images). 
                    annotations (annotations): list of paths to annotations files. down_factor (int): downsampling factor used to generate the image macros. 
                    output_path (string): output path.
        """
        self.aperios = aperios
        self.camms = camms
        self.annotations = annotations
        self.scale = 1/down_factor
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
        
    def update_aperios(self, aperios):
        self.aperios = aperios
    
    def update_camms(self, camms):
        self.camms = camms
        
    def update_annotations(self, annotations):
        self.annotations = annotations
        
    def generate_thumbnails(self, down_factor=32):
        aperios = self.aperios
        camms = self.camms
        files = aperios + camms
        for aperio in files:
            slide = openslide.open_slide(aperio)
            w, h = slide.dimensions
            thumbnail = np.array(slide.get_thumbnail(size=(w/down_factor, h/down_factor)))
            save_path = splitext(aperio)[0]+'.png'
            cv2.imwrite(save_path, thumbnail)
            
        
    def transform_polygons(self, xml_path, tfm, scale=(1/32, 1, 1)):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        annotations = root.findall('Annotation/Regions/Region')
        polygons = []
        t_polygons = []
        for annotation in annotations:
            coords = annotation.findall('Vertices/Vertex')
            polygon = []
            t_polygon = []
            for coord in coords:
                x = float(coord.attrib['X'])
                y = float(coord.attrib['Y'])
                mov = [x*scale[0]*scale[1], y*scale[0]*scale[2], 1]
                t_mov = tfm @ mov
                t_x = int(np.round((t_mov[0]-1)/scale[0]))
                t_y = int(np.round((t_mov[1]-1)/scale[0]))
                polygon.append((int(np.round(x)), int(np.round(y))))
                t_polygon.append((t_x, t_y))
                coord.set('X', str(t_x))
                coord.set('Y', str(t_y))
            polygons.append(polygon)
            t_polygons.append(t_polygon)
        tree.write(join(self.output_path, xml_path.split(os.sep)[-1]))
        return polygons, t_polygons
    
    def register(self, img_path_moving, img_path_fixed):
        img1_color = cv2.imread(img_path_moving)  # moving
        img2_color = cv2.imread(img_path_fixed)   # fixed

        # resize the image
        h_aperio, w_aperio = img1_color.shape[:2]
        h_camm, w_camm = img2_color.shape[:2]

        x_scale = w_camm/w_aperio
        y_scale = h_camm/h_aperio

        img1_color = cv2.resize(img1_color, (w_camm, h_camm))


        # Convert to grayscale
        img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)
        height, width = img2.shape

        # Create ORB detector with 5000 features.
        orb_detector = cv2.ORB_create(5000)

        # Find keypoints and descriptors.
        # The first arg is the image, second arg is the mask
        #  (which is not required in this case).
        kp1, d1 = orb_detector.detectAndCompute(img1, None)
        kp2, d2 = orb_detector.detectAndCompute(img2, None)

        # Match features between the two images.
        # We create a Brute Force matcher with
        # Hamming distance as measurement mode.
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

        # Match the two sets of descriptors.
        matches = matcher.match(d1, d2)

        # Sort matches on the basis of their Hamming distance.
        matches.sort(key = lambda x: x.distance)

        # Take the top 90 % matches forward.
        matches = matches[:int(len(matches)*0.9)]
        no_of_matches = len(matches)

        # Define empty matrices of shape no_of_matches * 2.
        p1 = np.zeros((no_of_matches, 2))
        p2 = np.zeros((no_of_matches, 2))

        for i in range(len(matches)):
            p1[i, :] = kp1[matches[i].queryIdx].pt
            p2[i, :] = kp2[matches[i].trainIdx].pt


        homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
        transformed_img = cv2.warpPerspective(img1_color,
                            homography, (width, height))
        img_name = splitext(img_path_moving.split(os.sep)[-1])[0]
        cv2.imwrite(join(self.output_path, img_name+'.tiff'), transformed_img)
        with open(join(self.output_path, img_name+'.npy'), 'wb') as f:
            np.save(f, homography)

        return homography, x_scale, y_scale
    
    def translate(self, get_polygons=False, trust_order=False):
        """
            Args: get_polygons (bool): return transformed polygon points in a dict. trust_order (bool): use file order to match files, otherwise use file names.
            Returns (optional): A dict containing aperio image names and original/transformed polygon points pairs.
        """
        aperios = self.aperios
        camms = self.camms
        annotations = self.annotations
        returns = {}
        for aperio, camm, annotation in tqdm(zip(aperios, camms, annotations), total=len(aperios)):
            if not trust_order:
                if not (splitext(aperio.split(os.sep)[-1])[0] == splitext(camm.split(os.sep)[-1])[0] == splitext(annotation.split(os.sep)[-1])[0]):
                    raise ValueError('File names do not match!')
                img_name = splitext(aperio.split(os.sep)[-1])[0]
            else:
                if not (len(aperios) == len(camms) == len(annotations)):
                    raise ValueError('Numbers of files do not match!')
                img_name = splitext(aperio.split(os.sep)[-1])[0]
            tfm, x_scale, y_scale = self.register(aperio, camm)
            polygons, t_polygons = self.transform_polygons(annotation, tfm, (self.scale, x_scale, y_scale))
            returns[img_name] = ((polygons, t_polygons))
        if get_polygons:
            return returns
        
        
def check_tumor(point, polygons, physical_size=224):
    points = ((point[0]+physical_size, point[1]), 
              (point[0], point[1]+physical_size), 
              (point[0]+physical_size, point[1]+physical_size), 
              (point[0]+physical_size/2, point[1]+physical_size/2), 
              (point[0], point[1]))
    inside = False
    for polygon in polygons:
        try:
            polygon = Polygon(polygon)
            for p in points:
                p = Point(p)
                if polygon.contains(p):
                    inside = True
        except:
            continue
    return inside


def get_polygons(xml_path, annotation_tool='Aperio'):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    if annotation_tool == 'Aperio':
        annotations = root.findall('Annotation/Regions/Region')
    else:
        annotations = root.findall('Annotations/Annotation')
    polygons = []
    for annotation in annotations:
        if annotation_tool == 'Aperio':
            coords = annotation.findall('Vertices/Vertex')
        else:
            coords = annotation.findall('Coordinates/Coordinate')
        polygon = []
        for coord in coords:
            x = float(coord.attrib['X'])
            y = float(coord.attrib['Y'])
            polygon.append((int(np.round(x)), int(np.round(y))))
        polygons.append(polygon)
    return polygons


def group_patches(out_path, tumor_bags, tumor_annotations, pixel_size=1, patch_size=224, ext='jpeg'):
    """This function takes in a list of bags of patches and annotation xmls and group patches into `tumor` or `normal` folder.
        Args: out_path (string): output folder. tumor_bags (list): list of folders of bags. tumor_annotations (list): list of file paths of xml (names must match).
            pixel_size (flot): pixel size in um, use `1` if annotations using pixel unit. patch_size (int): image patch size. ext (string): image patch extension.
    """
    for tumor_bag, tumor_annotation in tqdm(zip(tumor_bags, tumor_annotations), total=len(tumor_bags)):
        polygons = get_polygons(tumor_annotation)
#         return polygons
        imgs = glob(join(tumor_bag, '*.'+ext))
        imgs.sort()
        slide_name = tumor_bag.split(os.sep)[-1]
        os.makedirs(os.path.join(out_path, slide_name, 'tumor'), exist_ok=True)
        os.makedirs(os.path.join(out_path, slide_name, 'normal'), exist_ok=True)
        for img in imgs:
            img_name = splitext(os.path.basename(img))[0]
            y = int(img_name.split('_')[1]) * patch_size * pixel_size
            x = int(img_name.split('_')[0]) * patch_size * pixel_size
            if check_tumor((x, y), polygons, physical_size=patch_size*pixel_size):
                save_name = os.path.join(out_path, slide_name, 'tumor', img_name+'.'+ext)
                shutil.copyfile(img, save_name)
            else:
                save_name = os.path.join(out_path, slide_name, 'normal', img_name+'.'+ext)
                shutil.copyfile(img, save_name)
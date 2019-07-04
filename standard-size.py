import os
import cv2
import sys
import math
import numpy as np
from shutil import copy2
from wand.image import Image

# Global variables
img_path = '/home/user/class/mc906/p4/dataset/organized'
tgtdim = 224

groups = ['test', 'training', 'validation']
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

for group in groups:
    for emotion in emotions:
        x_path = os.path.join(img_path, group, emotion)
        for original in os.listdir(x_path):
            if original.endswith(".jpg"):
                i_path = os.path.join(x_path, original)
                with Image(filename=i_path) as img:
                    if img.width or img.height != tgtdim:
                        #Rescale
                        min_dim = min(img.width, img.height)
                        scale_factor = tgtdim/min_dim

                        new_width = int(scale_factor*img.width)
                        new_height = int(scale_factor*img.height)
                        img.resize(new_width, new_height)

                        #Center crop
                        img.crop(width=tgtdim, height=tgtdim, gravity='center')

                        img.save(filename=i_path)

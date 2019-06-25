import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Global variables
img_path = '/home/user/class/mc906/p4/dataset/croped'

annotation_file = '/home/user/class/mc906/p4/dataset/data/label/label.lst'

f = open(os.path.join(annotation_file), "r")

for original in os.listdir(img_path):
    if original.endswith(".jpg"):
        found = False
        without_fcrop = original[6:]

        # Find label for image
        f.seek(0)
        for line in f:
            name = line.split(' ')[0]
            if name == without_fcrop:
                found = True
                break
        if found == False:
            os.remove(original)

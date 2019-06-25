import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Global variables
img_path = '/home/user/class/mc906/p4/dataset/data/image/origin'

annotation_file = '/home/user/class/mc906/p4/dataset/data/label/label.lst'

f = open(os.path.join(annotation_file), "r")
labels = f.readlines()

emotion_dict = {}
for line in labels:
    split = line.split(' ', 1)
    emotion_dict[split[0]] = split[1]

for original in os.listdir(img_path):
    if original.endswith(".jpg"):
        # Read image
        image = cv2.imread(os.path.join(img_path, original))
        
        # Find label for image
        if original in emotion_dict:
            # Read face coordinates
            top_x, top_y, bot_y, bot_x = emotion_dict[name].split(' ')[1:5]
            
            top_x = int(top_x)
            top_y = int(top_y)
            bot_y = int(bot_y)
            bot_x = int(bot_x)

            # Crop face
            crop = image[top_x:bot_x, top_y:bot_y]

            # Save cropped face
            name = 'fcrop_'+original
            cv2.imwrite(name, crop)

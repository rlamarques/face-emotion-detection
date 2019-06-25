import os
import cv2
import math
import numpy as np
from shutil import copy2
import matplotlib.pyplot as plt

# Global variables
img_path = '/home/user/class/mc906/p4/dataset/croped/'

counter = 0
numbers = {}
for original in os.listdir(img_path):
    if original.endswith(".jpg"):
        image = cv2.imread(original)
        x, y = image.shape[:2]
        if x == y:
            counter += 1
        # if x not in numbers:
        #     numbers[x] = 1
        # else:
        #     numbers[x] += 1

print(counter)
exit()

f = open('plot-sizes.txt', 'w')

for length, count in numbers.items():
    f.write(str(length) + ' ' + str(count) + '\n')
    
import os
import cv2
import math
import numpy as np
from shutil import copy2
import matplotlib.pyplot as plt

# Global variables
img_path = '/home/user/class/mc906/p4/dataset/croped/'

annotation_file = '/home/user/class/mc906/p4/dataset/data/label/label.lst'

f = open(os.path.join(annotation_file), "r")
labels = f.readlines()

emotion_dict = {}
for line in labels:
    split = line.split(' ')
    emotion_dict[split[0]] = split[7][:-1]

print('Emotion Dict', len(emotion_dict))

names = []
for original in os.listdir(img_path):
    if original.endswith(".jpg"):
        no_crop = original[6:]
        names.append(no_crop)

print('Names', len(names))

# Makes sure that the cropped images are present in the label
new_dict = {}
for name in emotion_dict:
    if name in names:
        new_dict[name] = emotion_dict[name]

print('New dict', len(new_dict))

emotion_count = [0,0,0,0,0,0,0]
for emotion in new_dict.values():
    emotion_count[int(emotion)] += 1

print(emotion_count)
print(sum(x for x in emotion_count))

translation = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
txt = open('../image-count.txt', 'w')

distribution = []
for i, count in enumerate(emotion_count):
    training = math.floor(0.6*count)
    validation = test = math.floor(0.2*count)
    total = training + validation + test
    training += count - total

    row = [training, validation, test]
    distribution.append(row)

    print(translation[i], 'training:', training, 'validation:', validation, 'test:', test)
    txt.write(str(translation[i] +' ' + str(training) + ' ' + str(validation) + ' ' + str(test) + '\n'))
txt.close()

training_path = '/home/user/class/mc906/p4/dataset/organized/training/'
validation_path = '/home/user/class/mc906/p4/dataset/organized/validation/'
test_path = '/home/user/class/mc906/p4/dataset/organized/test/'

for cropped in os.listdir(img_path):
    if cropped.endswith('.jpg'):
        fname = cropped[6:]
        emotion = int(emotion_dict[fname])

        if distribution[emotion][0] > 0:
            copy2(cropped, os.path.join(training_path, translation[emotion]))
            distribution[emotion][0] -= 1
        elif distribution[emotion][1] > 0:
            copy2(cropped, os.path.join(validation_path, translation[emotion]))
            distribution[emotion][1] -= 1
        else:
            copy2(cropped, os.path.join(test_path, translation[emotion]))
            distribution[emotion][2] -= 1

print(distribution)

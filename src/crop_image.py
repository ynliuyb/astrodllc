import cv2
import numpy as np
import os
from PIL import Image
import matplotlib.image as mpig

def crop_image(pic_path, pic_num, cut_x, cut_y):
    picture = Image.open(pic_path)
    picture = np.array(picture)
    # picture = np.array(picture.convert('L'))
    (x, y) = picture.shape
    pic = np.zeros((cut_x, cut_x))
    num_x = int(x / cut_x)
    num_y = int(y / cut_y)
    for i in range(0, num_x):
        for j in range(0, num_y):
            pic = picture[i * cut_x: (i + 1) * cut_x, j * cut_y: (j + 1) * cut_y]
            dir_target = dir_name + "/" + pic_num + '({}_{}).jpg'.format(i + 1, j + 1)
            cv2.imwrite(dir_target, pic)
    # os.remove(pic_path)

path = "hmid-512"
dir_name = "hmid-256/1"

for parent, dirnames, filenames in os.walk(path):  
    for filename in filenames:
        pic_num = filename.split(".")[0]
        currentPath = os.path.join(parent, filename)
        print(currentPath)
        crop_image(currentPath, pic_num, 256, 256)

print("done!!!")


import numpy as np
from PIL import Image

####### step1 #######
data1 = np.load(path1)
data2 = np.load(path2)
data = data2 - data1

####### step2 #######
def normal(path, num):
    data = np.load(path)
    x, y = data.shape
    for i in range(x):
        for j in range(y):
            img = data[i][j]
            if img <= 127:
                data[i][j] = img + 128
            else:
                data[i][j] = img - 128
    np.save('Tio/2160_2560_2/T_' + str(num) +'.npy', data)

for i in range(49):
    path = 'Tio/2160_2560_1/T_' + str(i+2) +'.npy'
    normal(path, i+2)
print('over!')
import os
from PIL import Image
import numpy as np


crop_size = 200
scr_path = '../raw-img/'

folder = os.listdir(scr_path)


# first split images to train test and validation directory

os.system("mkdir -p {}".format('test'))
os.system("mkdir -p {}".format('valid'))
os.system("mkdir -p {}".format('train'))

ind = 0
for i in folder:

    for j in os.listdir(scr_path + i):

        if ind%12 == 0 :

            os.system('mv {} {}'.format(scr_path + '/' + i + '/' + j, 'test'))

        elif ind%51 == 0 :

            os.system('mv {} {}'.format(scr_path + '/' + i + '/' + j, 'valid'))

        else:

            os.system('mv {} {}'.format(scr_path + '/' + i + '/' + j, 'train'))

        ind += 1


# now center randomly crop images.
# first split images to train test and validation directory


os.system("mkdir -p {}".format('test_crop'))
os.system("mkdir -p {}".format('valid_crop'))
os.system("mkdir -p {}".format('train_crop'))

for i in ['test', 'valid', 'train']:

    ind = 0
    loop = 1

    if i == 'train':
        loop = 1      # epoch to augment the data

    for _ in range(loop):

        for j in os.listdir(i):

            if j.split('.')[-1].lower() not in ['jpg', 'png', 'jpeg']:
                continue

            im = Image.open(i + '/' + j)

            width, height = im.size

            if height <= crop_size:
                top = 0
                bottom = height

            else:
                top = np.random.randint(0, height-crop_size)
                bottom = top + crop_size

            if width <= crop_size:
                left = 0
                right = width

            else:
                left = np.random.randint(0, width-crop_size)
                right = left + crop_size

            im = im.crop((left, top, right, bottom))
            im = im.resize((crop_size, crop_size))

            try:
                im.save('{}_crop/{}.jpg'.format(i,ind))
            except:
                pass

            ind += 1


os.system("rm -r {}".format('test'))
os.system("rm -r {}".format('valid'))
os.system("rm -r {}".format('train'))

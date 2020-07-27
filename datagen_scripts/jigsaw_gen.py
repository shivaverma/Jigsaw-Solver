import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


def cuts(path, channel, dim, cuts, out_dir, epoch, name):

    ind = -1

    data = {'image': [], 'label': []}

    for e in range(epoch):

        im_list = os.listdir(path)
        cut_len = dim//cuts

        if cut_len*cuts != dim:
            return "provide a valid cut value"

        for im_name in im_list:

            ind += 1

            try:
                im = Image.open(path + im_name)
            except:
                continue

            im = np.array(im)

            if im.shape != (dim, dim, 3):
                continue

            new_im = np.zeros((dim, dim, channel))

            rand_pos = np.random.permutation([i for i in range(cuts*cuts)])

            for i in range(cuts):

                hor_cut = im[i*cut_len:(i+1)*cut_len]

                for j in range(cuts):

                    piece = hor_cut[:, j*cut_len:(j+1)*cut_len]

                    pos = rand_pos[i*cuts+j]
                    x = pos//cuts
                    y = pos%cuts

                    new_im[x*cut_len:(x+1)*cut_len,y*cut_len:(y+1)*cut_len] = piece

            label = np.zeros(cuts*cuts).astype('int')

            for i, val in enumerate(rand_pos):
                label[val] = int(i)

            new_label = ''
            for i in label:
                new_label += str(i) + ' '

            data['image'].append(str(ind)+'.jpg')
            data['label'].append(new_label)

            plt.imsave(out_dir + str(ind) + '.jpg', new_im/255)

    df = pd.DataFrame(data)
    df.to_csv('data_2x2/{}.csv'.format(name))


if __name__ == '__main__':

    os.system("mkdir -p {}".format('data_2x2'))
    os.system("mkdir -p {}".format('data_2x2/test'))
    os.system("mkdir -p {}".format('data_2x2/valid'))
    os.system("mkdir -p {}".format('data_2x2/train'))

    cuts(path='valid_crop/', channel=3, dim=200, cuts=2, out_dir='data_2x2/valid/', epoch=1, name='valid')
    cuts(path='test_crop/', channel=3, dim=200, cuts=2, out_dir='data_2x2/test/', epoch=1, name='test')
    cuts(path='train_crop/', channel=3, dim=200, cuts=2, out_dir='data_2x2/train/', epoch=4, name='train')
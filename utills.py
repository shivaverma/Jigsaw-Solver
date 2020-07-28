import copy
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


def swap_piece(im, pos, true_pos, cuts=2, dim=200):

    """
        swap 2 pieces and returns image
    """

    im = im

    cut_len = dim // cuts

    x = pos // cuts
    y = pos % cuts

    x_true = true_pos // cuts
    y_true = true_pos % cuts

    piece = copy.copy(im[x * cut_len:(x + 1) * cut_len, y * cut_len:(y + 1) * cut_len, ])
    swap_piece = copy.copy(im[x_true * cut_len:(x_true + 1) * cut_len, y_true * cut_len:(y_true + 1) * cut_len, ])

    im[x * cut_len:(x + 1) * cut_len, y * cut_len:(y + 1) * cut_len, ] = swap_piece
    im[x_true * cut_len:(x_true + 1) * cut_len, y_true * cut_len:(y_true + 1) * cut_len, ] = piece

    return im


def rearrange(im, label, cuts=2, dim=200, channel=3):

    """
        rearranges the image according to the label
    """

    cut_len = dim // cuts

    new_im = np.zeros((dim, dim, channel))
    for i in range(cuts):

        hor_cut = im[i * cut_len:(i + 1) * cut_len]

        for j in range(cuts):
            piece = hor_cut[:, j * cut_len:(j + 1) * cut_len]

            pos = label[i * cuts + j]
            x = pos // cuts
            y = pos % cuts

            new_im[x * cut_len:(x + 1) * cut_len, y * cut_len:(y + 1) * cut_len] = piece

    plt.imshow(new_im)
    plt.show()


def extract_piece(a, size=200, cuts=2):

    """
       extracts each piece of the puzzle and returns
    """

    cut_len = size // cuts

    if cuts == 3:
        a = np.array([a[:, 0:cut_len, :], a[:, cut_len:cut_len * 2, :], a[:, cut_len * 2:cut_len * 3, :]])
        a = np.concatenate(
            (a[:, 0:cut_len, :, :], a[:, cut_len:cut_len * 2, :, :], a[:, cut_len * 2:cut_len * 3, :, :]))
    if cuts == 2:
        a = np.array([a[:, 0:cut_len, :], a[:, cut_len:, :]])
        a = np.concatenate((a[:, 0:cut_len, :, :], a[:, cut_len:, :, :]))

    return a


def load_data(base_path, path, cuts=2):

    """
        loads and returns data
    """

    data = pd.read_csv(base_path + '{}.csv'.format(path))
    path = base_path + path + '/'

    x = []
    y = []
    total = len(data)
    for i in range(total):

        im = Image.open(path + data.iloc[i]['image'])
        im = np.array(im).astype('float16')
        im = im / 255 - 0.5

        if path.split('/')[-2] == 'test':
            x.append(im)
        else:
            x.append(extract_piece(im))

        label = data.iloc[i]['label']
        label = [int(i) for i in label.split()]
        y.append(label)

    return (np.array(x), np.expand_dims(np.array(y), axis=-1))

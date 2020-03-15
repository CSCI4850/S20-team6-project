# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm

HEIGHT = 2
WIDTH = 2


def encode(cover_name, hidden_name, output_name):
    cover_im = Image.open(cover_name)
    cover_im = cover_im.resize((HEIGHT, WIDTH), Image.ANTIALIAS)
    hidden_im = Image.open(hidden_name)
    hidden_im = hidden_im.resize((HEIGHT, WIDTH), Image.ANTIALIAS)
    cover_pix = cover_im.load()
    hidden_pix = hidden_im.load()
    output_image = np.empty((HEIGHT, WIDTH, 3))
    for x in range(WIDTH):
        for y in range(HEIGHT):
            cover_R = '{0:08b}'.format(cover_pix[x,y][0])
            cover_G = '{0:08b}'.format(cover_pix[x,y][1])
            cover_B = '{0:08b}'.format(cover_pix[x,y][2])
            hidden_R = '{0:08b}'.format(hidden_pix[x,y][0])
            hidden_G = '{0:08b}'.format(hidden_pix[x,y][1])
            hidden_B = '{0:08b}'.format(hidden_pix[x,y][2])
            output_R = int(cover_R[:5] + hidden_R[:3], 2) / 255.0
            output_G = int(cover_G[:5] + hidden_G[:3], 2) / 255.0
            output_B = int(cover_B[:5] + hidden_B[:3], 2) / 255.0
            output_image[y][x] = (output_R, output_G, output_B)
            print('{0:08b}'.format(int(output_R * 255)))
    matplotlib.image.imsave(output_name, output_image)


def decode(hidden_name, output_name):
    hidden_im = Image.open(hidden_name)
    hidden_pix = hidden_im.load()
    output_image = np.empty((HEIGHT, WIDTH, 3))
    for x in range(WIDTH):
        for y in range(HEIGHT):
            hidden_R = '{0:08b}'.format(hidden_pix[x,y][0])
            recoverd_R = hidden_R[5:] + '00000'
            hidden_G = '{0:08b}'.format(hidden_pix[x,y][1])
            recoverd_G = hidden_G[5:] + '00000'
            hidden_B = '{0:08b}'.format(hidden_pix[x,y][2])
            recoverd_B = hidden_B[5:] + '00000'
            output_R = int(recoverd_R, 2) / 255.0
            output_G = int(recoverd_G, 2) / 255.0
            output_B = int(recoverd_B, 2) / 255.0
            output_image[y][x] = (output_R, output_G, output_B)
            print(hidden_R)
    matplotlib.image.imsave(output_name, output_image)


encode("cover.jpg", "hidden.jpg", "covered.jpg")

decode("covered.jpg", "uncovered.jpg")


